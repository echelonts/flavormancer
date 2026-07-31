"""
predict.py — the unified flavor read the workbench screen renders.

One molecule in, one dict out, combining whatever heads exist in taste_models/:
  aroma          : DEFERRED — honest 'not available' (no clean public data; see docs/AROMA.md)
  sweet/bitter/umami : probabilities 0-1   (trained heads, if present)
  sweet_intensity    : ~relative-to-sucrose estimate (if regressor present)
  sour           : bool + which acid group  (RULE — acidic groups)
  salty          : bool + reason            (RULE — inorganic alkali salt, anion-guarded)
  safety         : disclaimer + scope + structural alerts + GRAS status + TTC hint
                   (DEFENSIVE, caution-only — never a safety clearance)
  physchem       : logP/MW/TPSA/HBD/HBA (computed) + solubility (ESOL estimate)
                   + aroma-volatility tier + ionizable-group pKa ranges (qualitative)
  stability      : oxidation / hydrolysis / photodegradation watch-flags (qualitative)
  chemesthesis   : trigeminal class flags — cooling / pungent / astringent (qualitative)

Each physchem value is tagged computed / estimate / qualitative so confidence is
explicit and nothing reads as more precise than it is.

  labeling       : EU declarable fragrance-allergen flag (regulatory lookup)

For formulations, check_mixture(ingredients, processes=[...]) flags documented
food hazards (benzene, nitrosamine, ethyl carbamate, acrylamide, furan, 3-MCPD,
4-MEI, biogenic amines), gated on the process (high_heat/refining/fermentation)
that causes them — active vs conditional. Curated, NOT a reaction predictor.

The taste heads load dynamically: whatever train_taste.py produced shows up
here automatically, so adding an umami/sour model later needs no edit.

Sour note: sourness is a solution/pH property, not a per-molecule ML target,
so we flag acidic functional groups as an honest proxy. True sour balance is
formulation-level (titratable acidity / pH), which their data teaches later.

Salty note: saltiness is an ionic effect, not a molecular-shape one, so it can't
be a trained head either. But it IS partly structure-readable: a simple
inorganic alkali/ammonium salt (NaCl, KCl, NH4Cl...) is reliably salty. The trap
is sodium-bearing organics — MSG (umami), sodium saccharin (sweet), sodium
benzoate (preservative) — where the organic ANION drives taste and the cation is
incidental. So the rule fires only on alkali/ammonium + a simple INORGANIC anion,
and defers to the anion's taste whenever the anion carries carbon. That mirrors
the sour rule's spirit while refusing the naive "has sodium -> salty" mistake.
Hard ceiling: it nails simple salts and honestly can't reach salt-enhancer
peptides or non-ionic salty compounds (little data, weak structure-activity).
"""

import contextlib
import os
import threading as _threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    DataStructs,
    Descriptors,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

# Where the trained artifacts live. Defaults to the working directory, which is how the systemd
# deployment has always run (code and models share one directory). Setting FLAVORMANCER_HOME lets
# a container bake the CODE into the image while MOUNTING the ~1 GB of models and parquet tables —
# without it, any bind mount that reached the artifacts would also shadow app.py.
HOME = Path(os.environ.get("FLAVORMANCER_HOME") or ".")


def artifact(name):
    """Resolve one trained artifact (model directory, parquet or csv) under FLAVORMANCER_HOME."""
    return HOME / name


FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
TASTE = artifact("taste_models")

ACID_SMARTS = {
    # Match both protonated (-OH) and deprotonated (-O-) forms — sour compounds are
    # routinely drawn as carboxylate/sulfonate/phosphate anions or zwitterions.
    # (Lifted the rule's recall on labeled-sour from 0.57 to 0.93.)
    "carboxylic acid / carboxylate": "[CX3](=O)[OX2H1,OX1-]",
    "sulfonic / sulfonate": "[SX4](=O)(=O)[OX2H1,OX1-]",
    "phosphoric / phosphonic (+ anion)": "[PX4](=O)[OX2H1,OX1-]",
}
_ACID = {k: Chem.MolFromSmarts(v) for k, v in ACID_SMARTS.items()}

# Salty rule: alkali metals (Li, Na, K, Rb, Cs) + ammonium are the salt-forming
# cations. Saltiness fires only when one of these pairs with a simple INORGANIC
# anion; an organic (carbon-bearing) anion means the anion drives taste instead.
_ALKALI_Z = {3, 11, 19, 37, 55}
_SYM = {3: "Li", 11: "Na", 19: "K", 37: "Rb", 55: "Cs"}

# ── SAFETY (all defensive, caution-only — never a clearance) ──────────────────
SAFETY_DISCLAIMER = (
    "Taste/aroma prediction only. This is NOT a safety, toxicity, GRAS, "
    "regulatory, or chemical-stability determination. Every formulation must be "
    "validated by qualified toxicology and regulatory review before use."
)

# A SMALL, curated set of high-signal structural alerts. These are PROMPTS FOR
# REVIEW, not toxicity verdicts — some safe compounds share these motifs. Kept
# deliberately to groups that are rare in the GRAS flavor palette to avoid alert
# fatigue (e.g. we do NOT flag aldehydes or Michael acceptors — too many GRAS
# flavor compounds like vanillin or cinnamaldehyde carry them).
TOX_ALERT_SMARTS = {
    "aromatic nitro": "[c][$([NX3](=O)=O),$([N+](=O)[O-])]",
    "N-nitroso (nitrosamine)": "[NX3][NX2]=O",
    "aromatic azo": "[c][NX2]=[NX2][c]",
    "epoxide": "[#6]1[#6]O1",
}
_TOX = {k: Chem.MolFromSmarts(v) for k, v in TOX_ALERT_SMARTS.items()}

# Mixture / process hazards: documented, curated — NOT a general reaction predictor.
# Roles are detected per-molecule; each hazard fires on roles (+ optionally a
# declared process condition that actually causes it). Detectors:
_BENZOATE = Chem.MolFromSmarts("[#6;a]C(=O)[OX2H1,OX1-]")  # benzoic acid / benzoate
_NITRITE = Chem.MolFromSmarts("[NX2](=O)[OX1-,OX2H1]")      # nitrite / nitrous
_SEC_AMINE = Chem.MolFromSmarts("[NX3;H1;!$(N-C=O)]([#6])[#6]")  # secondary amine, not amide
_UREA = Chem.MolFromSmarts("[NX3][CX3](=O)[NX3]")           # urea / carbamide
_CHLORIDE = Chem.MolFromSmarts("[Cl-]")                     # ionic chloride source
_AMMONIUM_ION = Chem.MolFromSmarts("[NX4H4+]")              # ammonium
_GLYCEROL_BB = Chem.MolFromSmarts("[CH2X4]([OX2])[CHX4]([OX2])[CH2X4][OX2]")  # glycerol backbone
_ACYL_ESTER = Chem.MolFromSmarts("[OX2][CX3]=O")           # ester linkage


def _ik1(*smiles):
    """InChIKey first-blocks (skeleton hashes) computed from SMILES, so the
    reference keys are correct by construction rather than hand-typed."""
    s = set()
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            s.add(Chem.MolToInchiKey(m).split("-")[0])
    return s


_ASCORBATE_IKS = _ik1("OCC(O)C1OC(=O)C(O)=C1O", "[Na+].OCC(O)C1OC(=O)C(O)=C1[O-]")
_ETHANOL_IKS = _ik1("CCO")
_ASPARAGINE_IKS = _ik1("NC(=O)CC(N)C(=O)O")
_CITRULLINE_IKS = _ik1("NC(=O)NCCCC(N)C(=O)O")
_HISTIDINE_IKS = _ik1("NC(Cc1cnc[nH]1)C(=O)O")
_TYROSINE_IKS = _ik1("NC(Cc1ccc(O)cc1)C(=O)O")
_REDUCING_SUGAR_IKS = _ik1("OCC1OC(O)C(O)C(O)C1O", "OCC1(O)OCC(O)C(O)C1O")  # glucose, fructose

# (roles required, process required or None, byproduct, note). Process tags:
# "high_heat", "refining", "fermentation". A None process = forms without a
# special step. Process-gated rules with no declared process surface as CONDITIONAL.
_HAZARDS = [
    ({"benzoate", "ascorbate"}, None,
     "benzene (a carcinogen), favored by heat/light",
     "Documented in soft drinks; FDA-investigated."),
    ({"nitrite", "secondary_amine"}, None,
     "N-nitrosamines (carcinogenic)",
     "Classic cured-food chemistry."),
    ({"ethanol", "urea"}, None,
     "ethyl carbamate / urethane (probable carcinogen)",
     "Relevant to spirits / fermented products."),
    ({"ethanol", "citrulline"}, None,
     "ethyl carbamate (probable carcinogen)",
     "Citrulline route — stone-fruit spirits especially."),
    ({"asparagine", "reducing_sugar"}, {"high_heat"},
     "acrylamide (probable carcinogen)",
     "Maillard route, >120C. Precursor co-occurrence, not a yield prediction."),
    ({"reducing_sugar"}, {"high_heat"},
     "furan / furfural (possible carcinogen)",
     "Sugar pyrolysis under heat (also in heat-processed/canned products)."),
    ({"ascorbate"}, {"high_heat"},
     "furan (possible carcinogen)",
     "Ascorbic-acid thermal degradation."),
    ({"glyceride", "chloride"}, {"high_heat", "refining"},
     "3-MCPD / glycidyl esters (process contaminants)",
     "Acylglycerol + chloride at high heat / oil refining."),
    ({"ammonium", "reducing_sugar"}, {"high_heat"},
     "4-methylimidazole (in ammonia caramel colours)",
     "Caramelisation with an ammonia source."),
    ({"histidine"}, {"fermentation"},
     "histamine (biogenic amine)",
     "Amino-acid decarboxylation in fermentation / spoilage."),
    ({"tyrosine"}, {"fermentation"},
     "tyramine (biogenic amine)",
     "Amino-acid decarboxylation in fermentation / spoilage."),
]

# EU declarable fragrance/flavor allergens (a regulatory labeling list — a clean
# lookup). A curated subset of the classic 26; expand from the official annex.
_EU_ALLERGEN_IKS = {}
for _nm, _smi in {
    "limonene": "CC(=C)C1CCC(C)=CC1", "linalool": "CC(C)=CCCC(C)(O)C=C",
    "citronellol": "CC(CCC=C(C)C)CCO", "geraniol": "CC(C)=CCC/C(C)=C/CO",
    "eugenol": "C=CCc1ccc(O)c(OC)c1", "isoeugenol": "CC=Cc1ccc(O)c(OC)c1",
    "cinnamaldehyde": "O=C/C=C/c1ccccc1", "cinnamyl alcohol": "OC/C=C/c1ccccc1",
    "coumarin": "O=c1ccc2ccccc2o1", "citral": "CC(=CCCC(=CC=O)C)C",
    "benzyl alcohol": "OCc1ccccc1", "farnesol": "CC(C)=CCC/C(C)=C/CC/C(C)=C/CO",
}.items():
    _EU_ALLERGEN_IKS.update({k: _nm for k in _ik1(_smi)})

def _load_rf(path):
    """Load a joblib RF head and pin n_jobs=1. These forests were trained with n_jobs=-1, which
    makes a SINGLE-sample predict_proba spawn a joblib thread pool on every call — ~150 ms of
    pure overhead that dwarfs the actual work and thrashes all cores under any concurrency.
    Single-threaded C tree traversal is far faster for our one-row inference, and it releases
    the GIL, so the app can parallelize a level up (e.g. per-ingredient in the Formulation
    Studio) instead of fighting joblib. Measured: ~3.8 s -> ~1.2 s per 24-head aroma read."""
    mdl = joblib.load(path)
    if hasattr(mdl, "n_jobs"):
        with contextlib.suppress(Exception):  # some wrapped estimators reject the set; harmless
            mdl.n_jobs = 1
    return mdl


# Model heads are loaded on a BACKGROUND THREAD at import (see _load_all_models below) so that
# `import predict` returns immediately and the web server can bind its port right away, showing a
# friendly "warming up" page while the ~195 forests (~700 MB) load — instead of a 50 s startup 502.
# The load is SERIAL, and this comment used to claim the opposite ("fanned out across cores,
# joblib.load releases the GIL"). Both halves of that were wrong: unpickling is GIL-bound, so
# threads made it ~2.5x SLOWER, and a process pool deadlocks because this runs during module
# import. See _load_all_models for the measurements and #225 for the real fix.
_CLASSIFIERS = {}          # sweet/bitter/umami/... taste heads
_INTENSITY = None          # sweet-intensity regressor
_TASTE_META = {}           # taste -> {auroc, ...} from taste_models/manifest.json (held-out score)
_TOX_MODELS = {}           # Tox21 caution-only assay heads (INDICATIVE, never a determination)
_TOX_META = {}             # assay -> {auroc, n_pos, ...} from tox_models/manifest.json (held-out CV)
_TOX_DIR = artifact("tox_models")
_AROMA_MODELS = {}         # HSDB odor-descriptor heads (presence/absence; NOT intensity)
_AROMA_META = {}
_AROMA_DIR = artifact("aroma_models")
_MOUTHFEEL_MODELS = {}     # trigeminal/chemesthesis heads (warming/astringent/tingling), own modality
_MOUTHFEEL_META = {}
_MOUTHFEEL_DIR = artifact("mouthfeel_models")

MODELS_READY = _threading.Event()  # set once every head is loaded; the app gates requests on this
_INFER_POOL = None  # shared thread pool for fanning a novel-molecule read across cores (lazy)


def _infer_pool():
    """A process-wide thread pool for parallel head inference on novel molecules. Sized to ~3/4 of
    the box (env FLAVORMANCER_INFER_WORKERS overrides) so a fresh 195-head read rips across cores
    (~40 s -> a couple of seconds). Shared, so many concurrent novel reads share one bounded pool
    instead of each spawning its own — in-corpus reads never touch it (they hit the index)."""
    global _INFER_POOL
    if _INFER_POOL is None:
        env = os.environ.get("FLAVORMANCER_INFER_WORKERS")
        # ~3/4 of the box's cores (leaving headroom for the web server), derived from the actual
        # cpu count — scales from a 4-core laptop (3 workers) to a 32-core server (24), no fixed cap.
        workers = int(env) if env else max(2, (os.cpu_count() or 4) * 3 // 4)
        _INFER_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="infer")
    return _INFER_POOL
# live progress for the warming-up page: how many heads are loaded, and the phase label
LOAD_PROGRESS = {"loaded": 0, "total": 0, "phase": "starting", "ready": False, "started": None}


def load_status():
    """Snapshot of the model-load progress for the warming-up page: loaded/total heads, phase,
    ready flag, and seconds elapsed since loading began (for a client-side ETA)."""
    with _LOAD_LOCK:
        s = dict(LOAD_PROGRESS)
    started = s.pop("started", None)
    s["elapsed"] = round(_time.monotonic() - started, 1) if started else 0.0
    s["ready"] = MODELS_READY.is_set()
    return s
_LOAD_LOCK = _threading.Lock()


def _load_all_models():
    """Discover and load every trained head (taste + tox + aroma), updating LOAD_PROGRESS as each
    lands, then set MODELS_READY. Runs on a daemon thread from import so the port binds instantly.

    Loading is SERIAL, and BOTH ways of parallelising it have now been measured and rejected:

      threads    WORSE than serial (~34 s serial vs ~86 s across 14 threads). joblib.load is
                 dominated by GIL-bound Python unpickling, so threads only add contention.
      processes  3.1x FASTER in isolation — 40 files take 11.5 s serial and 3.7 s across a
                 ProcessPoolExecutor, transfer of the deserialised forests included. But this
                 function runs DURING MODULE IMPORT, and forking while the interpreter holds the
                 import lock deadlocks: the children inherit a locked import machinery they can
                 never acquire. Tried it; the parent and every worker hung indefinitely.

    The process pool is the right answer, but only once loading is deferred out of import time
    (a FastAPI startup hook, or an explicit warm() the server calls) — see #225. Until then serial
    is correct, and the warming page makes the ~50 s visible rather than mysterious.

    The win from cores comes at INFERENCE time instead (predict_proba releases the GIL) — see
    _aroma_scores_canon."""
    jobs = []  # (kind, name, path)
    if TASTE.exists():
        jobs += [("taste", p.stem.replace("_rf", ""), p) for p in TASTE.glob("*_rf.joblib")]
    if _TOX_DIR.exists():
        jobs += [("tox", p.stem.replace("_rf", ""), p) for p in _TOX_DIR.glob("*_rf.joblib")]
    if _AROMA_DIR.exists():
        jobs += [("aroma", p.stem.replace("_clf", ""), p) for p in _AROMA_DIR.glob("*_clf.joblib")]
    if _MOUTHFEEL_DIR.exists():
        jobs += [("mouthfeel", p.stem.replace("_clf", ""), p) for p in _MOUTHFEEL_DIR.glob("*_clf.joblib")]
    with _LOAD_LOCK:
        LOAD_PROGRESS["total"] = len(jobs)
        LOAD_PROGRESS["phase"] = "loading models"
        LOAD_PROGRESS["started"] = _time.monotonic()

    for kind, name, p in jobs:
        mdl = _load_rf(p)
        if kind == "taste":
            if name == "sweet_intensity":
                globals()["_INTENSITY"] = mdl
            else:
                _CLASSIFIERS[name] = mdl
        elif kind == "tox":
            _TOX_MODELS[name] = mdl
        elif kind == "mouthfeel":
            _MOUTHFEEL_MODELS[name] = mdl
        else:
            _AROMA_MODELS[name] = mdl
        with _LOAD_LOCK:
            LOAD_PROGRESS["loaded"] += 1
    # manifests (small JSON, load after the heads)
    import json as _json
    _tm = TASTE / "manifest.json"
    if _tm.exists():
        globals()["_TASTE_META"] = _json.loads(_tm.read_text())
    _mf = _AROMA_DIR / "manifest.json"
    if _mf.exists():
        globals()["_AROMA_META"] = _json.loads(_mf.read_text()).get("descriptors", {})
    _txf = _TOX_DIR / "manifest.json"
    if _txf.exists():
        globals()["_TOX_META"] = _json.loads(_txf.read_text()).get("assays", {})
    _mff = _MOUTHFEEL_DIR / "manifest.json"
    if _mff.exists():
        globals()["_MOUTHFEEL_META"] = _json.loads(_mff.read_text()).get("descriptors", {})
    with _LOAD_LOCK:
        LOAD_PROGRESS["phase"] = "ready"
        LOAD_PROGRESS["ready"] = True
    MODELS_READY.set()


# Kick off loading in the background. Set FLAVORMANCER_BLOCKING_LOAD=1 (tests, CLI, batch jobs) to
# load synchronously instead. Set FLAVORMANCER_NO_MODELS=1 to skip loading entirely — for tools that
# only need featurization (_feat / _MORGAN), like the parallel index builder, which loads each head
# in its own worker process rather than in this parent.
if os.environ.get("FLAVORMANCER_NO_MODELS") == "1":
    # Skip loading, but still mark READY. Leaving the event unset meant the app's warming gate
    # returned 503 to every request FOREVER — a models-less install was not "degraded", it was
    # dead, which is the opposite of what the docs promised. Structure-derived answers (physchem,
    # the sour/salty rules, applicability, substructure) need no heads at all and should be served.
    with _LOAD_LOCK:
        LOAD_PROGRESS["phase"] = "ready (no models)"
        LOAD_PROGRESS["ready"] = True
    MODELS_READY.set()
elif os.environ.get("FLAVORMANCER_BLOCKING_LOAD") == "1":
    _load_all_models()
else:
    _threading.Thread(target=_load_all_models, name="model-loader", daemon=True).start()

# Known-label lookup: ground truth for molecules we actually have data on. This
# is how the salty/sour data works as a FLAG without a model — if a queried
# molecule is in our labeled set, we report the verified fact instead of a guess.
_KNOWN = {}  # inchikey -> {taste: 1}
_MASTER = artifact("taste_master.parquet")
# The neighbor / substitute reference set: the FULL molecule universe (every structure we know,
# ~8.8k) so structural neighbors and profile substitutes can surface ANY molecule — e.g. ethyl
# vanillin as the top vanillin substitute — not just the taste-labelled subset. Falls back to
# taste_master when the enrichment table hasn't been built yet.
_UNIVERSE = artifact("master_enrichment.parquet")
if _MASTER.exists():
    import pandas as pd
    _m = pd.read_parquet(_MASTER)
    _basic = [t for t in ("sweet", "bitter", "umami", "sour", "salty") if t in _m.columns]
    for _, _r in _m.iterrows():
        _labels = {t: 1 for t in _basic if _r[t] == 1}
        if _labels:
            _KNOWN[_r["inchikey"]] = _labels

# Optional GRAS / approved-flavor reference. The strongest *defensive* signal is
# not a tox model but "is this a recognized food ingredient at all?". Drop a
# reference list (e.g. the FEMA GRAS list) at gras_reference.parquet with an
# 'inchikey' column and we cross-check against it; absent the file we say so
# honestly rather than guessing.
_GRAS = set()
_GRAS_FILE = artifact("gras_reference.parquet")
if _GRAS_FILE.exists():
    import pandas as pd
    _g = pd.read_parquet(_GRAS_FILE)
    if "inchikey" in _g.columns:
        _GRAS = {str(k).split("-")[0] for k in _g["inchikey"].dropna()}

# Curated food-clearance supplement (food_safe_supplement.csv) — a few molecules that carry an
# aroma head but are NOT in the FDA SAF crawl, each backed by an OPEN-GOVERNMENT register only:
# the EU Union List (Reg. 1334/2008 Annex I, via data.food.gov.uk under the Open Government
# Licence) or US 21 CFR (public-domain law). Regulatory facts are non-copyrightable (Feist);
# no commercial compilation is used. Union into the same defensive "recognized food ingredient?"
# signal so these read as food-cleared everywhere the SAF list does.
def _foodsafe_label(fl, cfr):
    """Compose an accurate, specific food-use label + jurisdiction from the open-gov citations.
    Distinguishes true GRAS (21 CFR 182/184), FEMA GRAS (a FEMA number), an approved food additive
    (21 CFR 172), and an EU-authorised flavouring (EU FL) — never a blanket 'GRAS'."""
    fl, cfr = (fl or "").strip(), (cfr or "").strip()
    refs = [r for r in (f"EU FL {fl}" if fl else "", cfr) if r]
    juris = "US + EU" if (fl and cfr) else ("EU" if fl else ("US" if cfr else ""))
    low = cfr.lower()
    if "fema" in low:
        term = "FEMA GRAS"
    elif "182" in cfr or "184" in cfr:
        term = "GRAS"
    elif "172" in cfr:
        term = "approved food additive"
    elif fl:
        term = "EU-authorised flavouring"
    else:
        term = "authorised food ingredient"
    tag = f" ({juris} only)" if juris in ("US", "EU") else (f" ({juris})" if juris else "")
    return f"{term} — {' & '.join(refs)}{tag}" if refs else term


_FOODSAFE_FILE = artifact("food_safe_supplement.csv")
_FOODSAFE_BASIS = {}   # skeleton -> specific open-gov label (term + refs + jurisdiction)
if _FOODSAFE_FILE.exists():
    import pandas as pd
    # dtype=str + keep_default_na=False so FL numbers keep leading zeros ("07.142", not 7.142)
    # and empty cells read as "" rather than NaN.
    _fs = pd.read_csv(_FOODSAFE_FILE, dtype=str, keep_default_na=False)
    if "inchikey" in _fs.columns:
        _GRAS |= {str(k).split("-")[0] for k in _fs["inchikey"].dropna()}
        def _clean(v):
            s = str(v).strip()
            return "" if s.lower() in ("", "nan", "none") else s
        for _, _r in _fs.iterrows():
            _sk = str(_r.get("inchikey", "")).split("-")[0]
            _fl, _cfr = _clean(_r.get("eu_fl")), _clean(_r.get("us_cfr"))
            if _sk and (_fl or _cfr):
                _FOODSAFE_BASIS[_sk] = _foodsafe_label(_fl, _cfr)

# Bulk EU/GB flavourings Union List (gb_union_list.csv, ~2,200 authorised entries). The full
# authorisation register from data.food.gov.uk (Open Government Licence v3 — commercial reuse
# permitted); every AUTHORISED row is a food-cleared flavouring cited by its FL number. Union into
# the food-use reference so the whole authorised list reads food-listed with a specific citation.
# The FILE is a private data asset (gitignored); this LOADER is open framework.
_GB_FILE = artifact("gb_union_list.csv")
if _GB_FILE.exists():
    import pandas as pd
    _gb = pd.read_csv(_GB_FILE, dtype=str, keep_default_na=False)
    for _, _r in _gb.iterrows():
        if str(_r.get("status", "")).strip().lower() != "authorised":
            continue
        _sk = str(_r.get("inchikey", "")).split("-")[0]
        _fl = str(_r.get("fl", "")).strip()
        if not _sk or not _fl:
            continue
        _GRAS.add(_sk)
        _FOODSAFE_BASIS.setdefault(_sk, _foodsafe_label(_fl, ""))   # curated citation wins if present

# Optional measured-property + dosing table. Data-gated like GRAS. Drop
# properties.(parquet|csv) with an 'inchikey' column and any of:
# odor_threshold_ppm, fema_use_max_ppm, boiling_point_c, vapor_pressure_pa.
# We use MEASURED values (lookup) rather than structure estimates for these,
# because structure-based volatility (e.g. Joback) is too inaccurate for flavor
# molecules to report as a number — benzaldehyde misses by ~90 C.
_PROPS = {}
_PROP_COLS = ("odor_threshold_ppm", "fema_use_max_ppm", "boiling_point_c",
              "boiling_point_pressure_mmhg", "vapor_pressure_pa", "melting_point_c")
for _ext in ("properties.parquet", "properties.csv"):
    _pf = Path(_ext)
    if _pf.exists():
        import pandas as pd
        _pp = pd.read_parquet(_pf) if _ext.endswith("parquet") else pd.read_csv(_pf)
        if "inchikey" in _pp.columns:
            for _, _r in _pp.iterrows():
                vals = {c: float(_r[c]) for c in _PROP_COLS if c in _pp.columns and pd.notna(_r.get(c))}
                if vals:
                    _PROPS[str(_r["inchikey"]).split("-")[0]] = vals
        break


def _measured(mol):
    return _PROPS.get(Chem.MolToInchiKey(mol).split("-")[0], {})


def _fp(mol):
    bv = _MORGAN.GetFingerprint(mol)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.reshape(1, -1)


def _feat(mol):
    """Model input for the taste & aroma heads: the Morgan fingerprint PLUS the shared
    physicochemical descriptor block (see chemfeatures.py) — identical to how they were trained.
    Tox stays on the pure fingerprint; similarity/UMAP also keep the pure bits."""
    from chemfeatures import descriptors as _desc
    return np.hstack([_fp(mol).astype(np.float32), _desc(mol).reshape(1, -1)])


def _sour(mol):
    hits = [n for n, pat in _ACID.items() if pat is not None and mol.HasSubstructMatch(pat)]
    return {"sour": bool(hits), "sour_reason": hits}


def _is_salt_cation(frag):
    """A lone alkali-metal atom, or an ammonium (NH4+) — the salt-forming cations."""
    heavy = [a for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) != 1:
        return None
    a = heavy[0]
    if a.GetAtomicNum() in _ALKALI_Z:
        return _SYM[a.GetAtomicNum()]
    # ammonium: a single N(+) carrying 4 H and no heavy neighbors
    if (a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1
            and a.GetTotalNumHs() == 4):
        return "NH4"
    return None


def _has_carbon(frag):
    return any(a.GetAtomicNum() == 6 for a in frag.GetAtoms())


def _salty(mol):
    """Fire only for simple inorganic alkali/ammonium salts; defer on organic anions.

    Mirrors the sour rule, but cation-aware: NaCl/KCl/NH4Cl -> salty; MSG /
    Na-saccharin / Na-benzoate -> NOT salty (organic anion owns the taste).
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) < 2:
        return {"salty": False, "salty_reason": "no alkali-salt structure"}
    cations, others = [], []
    for f in frags:
        sym = _is_salt_cation(f)
        (cations if sym else others).append(sym or f)
    cations = [c for c in cations if c]
    if not cations or not others:
        return {"salty": False, "salty_reason": "no alkali-salt structure"}
    if any(_has_carbon(f) for f in others):
        # cation present, but a carbon-bearing anion drives the percept
        return {"salty": False, "salty_reason": "organic anion dominates (defer to anion taste)"}
    return {"salty": True, "salty_reason": f"inorganic {'/'.join(sorted(set(cations)))} salt"}


def _tox_alerts(mol):
    """Caution-only structural alerts. NOT toxicity verdicts — prompts for review."""
    return [n for n, pat in _TOX.items() if pat is not None and mol.HasSubstructMatch(pat)]


def _gras_status(mol):
    """Defensive 'is this even a recognized food ingredient?' check — NOT a GRAS or safety
    determination. The reference is a union of FDA's *Substances Added to Food* (SAF) inventory
    (public domain; broader than GRAS) and a curated EU/GB flavourings + 21 CFR set
    (`food_safe_supplement.csv`, open-government). We report *listing*, with the specific
    authority where known, and never claim a molecule is "GRAS" unless it actually is."""
    if not _GRAS:
        return "no food-use reference loaded — not checked"
    ik = Chem.MolToInchiKey(mol).split("-")[0]
    if ik in _FOODSAFE_BASIS:
        return _FOODSAFE_BASIS[ik]                # e.g. "FEMA GRAS — FDA SAF (FEMA 3434) (US only)"
    if ik in _GRAS:
        return "listed in the FDA Substances-Added-to-Food food-use reference (US)"
    return "not in the food-use reference — unverified for food use"


def _safety(mol):
    alerts = _tox_alerts(mol)
    return {
        "disclaimer": SAFETY_DISCLAIMER,
        "scope": "Taste/aroma only — not a safety/toxicity/GRAS/stability determination.",
        "structural_alerts": alerts,            # caution prompts, may be empty
        "gras_status": _gras_status(mol),
        "review_required": True,
    }


def _roles(mol):
    """Detect the reactive 'roles' used by the hazard screen."""
    r = set()
    if _BENZOATE is not None and mol.HasSubstructMatch(_BENZOATE):
        r.add("benzoate")
    if _NITRITE is not None and mol.HasSubstructMatch(_NITRITE):
        r.add("nitrite")
    if _SEC_AMINE is not None and mol.HasSubstructMatch(_SEC_AMINE):
        r.add("secondary_amine")
    if _UREA is not None and mol.HasSubstructMatch(_UREA):
        r.add("urea")
    if _CHLORIDE is not None and mol.HasSubstructMatch(_CHLORIDE):
        r.add("chloride")
    if _AMMONIUM_ION is not None and mol.HasSubstructMatch(_AMMONIUM_ION):
        r.add("ammonium")
    if (_GLYCEROL_BB is not None and _ACYL_ESTER is not None
            and mol.HasSubstructMatch(_GLYCEROL_BB) and mol.HasSubstructMatch(_ACYL_ESTER)):
        r.add("glyceride")
    ik = Chem.MolToInchiKey(mol).split("-")[0]
    for tag, ikset in (("ascorbate", _ASCORBATE_IKS), ("ethanol", _ETHANOL_IKS),
                       ("asparagine", _ASPARAGINE_IKS), ("citrulline", _CITRULLINE_IKS),
                       ("histidine", _HISTIDINE_IKS), ("tyrosine", _TYROSINE_IKS),
                       ("reducing_sugar", _REDUCING_SUGAR_IKS)):
        if ik in ikset:
            r.add(tag)
    return r


def check_mixture(ingredients, processes=None) -> dict:
    """Flag DOCUMENTED food hazards in a formulation. Curated, NOT a reaction predictor.

    ingredients: list of SMILES strings, or list of {"smiles": ...} dicts.
    processes:   optional set/list of process tags the product undergoes —
                 "high_heat", "refining", "fermentation". Hazards that require a
                 process surface as ACTIVE when the process is declared, or as
                 CONDITIONAL ("would form if ...") when it isn't.
    """
    procs = set(processes or [])
    present, parsed = set(), []
    for ing in ingredients:
        smi = ing["smiles"] if isinstance(ing, dict) else ing
        m = Chem.MolFromSmiles(smi or "")
        if m is not None:
            parsed.append(Chem.MolToSmiles(m))
            present |= _roles(m)
    active, conditional = [], []
    for roles, need_proc, product, note in _HAZARDS:
        if not roles <= present:
            continue
        entry = {"precursors": sorted(roles), "possible_product": product, "note": note}
        if need_proc is None or (procs & need_proc):
            active.append(entry)
        else:
            entry["requires_process"] = sorted(need_proc)
            conditional.append(entry)
    return {
        "ingredients_parsed": parsed,
        "processes_declared": sorted(procs),
        "active_hazards": active,
        "conditional_hazards": conditional,
        "scope_note": "Documented precursor/process hazards only — NOT a general reaction "
                      "predictor and NOT a yield or stability assay.",
        "disclaimer": SAFETY_DISCLAIMER,
    }


def labeling(mol):
    """Regulatory labeling flags — currently EU declarable fragrance/flavor allergens (lookup)."""
    name = _EU_ALLERGEN_IKS.get(Chem.MolToInchiKey(mol).split("-")[0])
    return {"eu_declarable_allergen": bool(name),
            "allergen_name": name,
            "note": "EU fragrance-allergen labeling list (curated subset) — a regulatory lookup"}


# ── Physicochemical pack: how the molecule behaves in a beverage ──────────────
# computed = exact from structure; estimate = published QSPR w/ error; qualitative = a class flag
_OXIDIZABLE = {
    "phenol/catechol": "[OX2H][c]",
    "thiol": "[SX2H]",
    "aldehyde": "[CX3H1]=O",
    "1,3-diene (autoxidation)": "[CX3]=[CX3][CX3]=[CX3]",
}
_HYDROLYZABLE = {
    "ester": "[CX3](=O)[OX2H0][#6;!$([CX3]=O)]",
    "lactone (cyclic ester)": "[CX3;R](=O)[OX2H0;R]",
    "acetal/glycoside": "[CX4]([OX2H0])[OX2H0]",
    "amide (slow)": "[CX3](=O)[NX3]",
}
_PHOTOLABILE = {
    "extended polyene": "[CX3]=[CX3][CX3]=[CX3][CX3]=[CX3]",
    "aryl ketone": "[c][CX3](=O)[#6]",
    "nitroaromatic": "[c][$([NX3](=O)=O),$([N+](=O)[O-])]",
}
_IONIZABLE = [  # (name, SMARTS, typical pKa, character)
    ("sulfonic acid", "[SX4](=O)(=O)[OX2H1]", "~ -1 to 2", "strong acid"),
    ("carboxylic acid", "[CX3](=O)[OX2H1]", "~3-5", "acid"),
    ("phenol", "[OX2H][c]", "~9-10", "weak acid"),
    ("aromatic amine (aniline)", "[NX3;H2,H1][c]", "~4-5 (conj. acid)", "weak base"),
    ("aliphatic amine", "[NX3;H2,H1;!$(N[#6]=[O,N,S]);!$(N[c])]", "~9-11 (conj. acid)", "base"),
]
_OX = {k: Chem.MolFromSmarts(v) for k, v in _OXIDIZABLE.items()}
_HY = {k: Chem.MolFromSmarts(v) for k, v in _HYDROLYZABLE.items()}
_PH = {k: Chem.MolFromSmarts(v) for k, v in _PHOTOLABILE.items()}
_ION = [(n, Chem.MolFromSmarts(s), p, c) for n, s, p, c in _IONIZABLE]
_PHENOL = Chem.MolFromSmarts("[OX2H][c]")

# Chemesthetic / trigeminal classes (curated, qualitative)
_ISOTHIOCYANATE = Chem.MolFromSmarts("[NX2]=[CX2]=[SX1]")  # pungent (mustard/wasabi)
_COOLING_IKS = _ik1("CC(C)C1CCC(C)CC1O")  # menthol  (expand: WS-3/WS-23, etc.)
_PUNGENT_IKS = _ik1("CC(C)/C=C/CCCCC(=O)NCc1ccc(O)c(OC)c1",  # capsaicin
                    "C1CCN(CC1)C(=O)/C=C/C=C/c1ccc2c(c1)OCO2")  # piperine


def physchem(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy = mol.GetNumHeavyAtoms()
    ap = (sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy) if heavy else 0.0
    # ESOL (Delaney 2004): log mol/L water solubility — estimate, ~0.7 log RMSE
    logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot - 0.74 * ap
    if mw < 250 and hbd <= 1 and tpsa < 60:
        vol = "high (likely top/volatile note)"
    elif mw < 400 and tpsa < 100:
        vol = "moderate (middle note)"
    else:
        vol = "low (base note / largely non-volatile)"
    ions = [{"group": n, "typical_pKa": p, "character": c}
            for n, pat, p, c in _ION if pat is not None and mol.HasSubstructMatch(pat)]
    result = {
        "computed": {
            "mol_weight": round(mw, 2), "logP": round(logp, 2), "tpsa": round(tpsa, 1),
            "h_bond_donors": hbd, "h_bond_acceptors": hba,
            "rotatable_bonds": rot, "aromatic_rings": arom, "heavy_atoms": heavy,
        },
        "estimate": {
            "water_solubility_logS": round(logS, 2),
            "note": "ESOL estimate (log mol/L), ~0.7 log RMSE",
        },
        "qualitative": {
            "aroma_volatility": vol,
            "volatility_note": "heuristic from size/polarity. Quantitative BP/vapor pressure "
                               "is a MEASURED lookup, not estimated (Joback too inaccurate here).",
            "ionizable_groups": ions,
            "pKa_note": "typical group ranges — NOT a computed per-molecule pKa",
        },
    }
    meas = _measured(mol)
    if meas:
        result["measured"] = {**{k: meas[k] for k in meas}, "source": "loaded property table"}
    # Flavorist formulation hints: carrier-solvent need (from estimated water solubility) and
    # room-temperature phase (from measured melting/boiling point, when the table has them).
    form = {}
    if logS <= -3:
        form["carrier"] = (f"poorly water-soluble (logS {logS:.1f}) — needs a carrier solvent "
                           "(propylene glycol, ethanol, or triacetin) to disperse in a water-based product")
    elif logS <= -1.5:
        form["carrier"] = (f"limited water solubility (logS {logS:.1f}) — a little propylene glycol "
                           "or ethanol helps it dissolve in water")
    else:
        form["carrier"] = f"reasonably water-soluble (logS {logS:.1f}) — usually no carrier needed"
    bp = (meas or {}).get("boiling_point_c")
    mp = (meas or {}).get("melting_point_c")
    if bp is not None and bp < 25:
        form["phase_at_rt"] = "gas"
    elif mp is not None:
        form["phase_at_rt"] = "solid" if mp > 25 else "liquid"
    result["formulation"] = form
    return result


def chirality(mol):
    """Stereo flag: enantiomers can taste/smell differently (R-carvone spearmint vs S caraway).
    We detect chiral centers (assigned or potential) and flag it honestly — the current models
    are achiral, so this is a caveat, not an enantiomer-specific prediction (see docs/AROMA.md)."""
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    if not centers:
        return {"is_chiral": False}
    assigned = [c for c in centers if c[1] not in ("?", "u")]
    return {"is_chiral": True, "n_centers": len(centers),
            "specified": len(assigned) == len(centers),
            "note": ("chiral — enantiomers can differ in taste/aroma; the current read is the same "
                     "for both mirror images (achiral model). Draw the stereochemistry (isomeric "
                     "SMILES) for the specific enantiomer's documented odor where PubChem has it.")}


def _stereo_label(mol):
    """A compact stereo-descriptor for one fully-specified isomer: tetrahedral R/S per center
    (with atom map) and E/Z per double bond — e.g. '(R)', '(2R,3S)', '(E)', '(1Z,2R)'. Covers
    ALL stereochemistry RDKit tracks, not just a single R/S center."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    parts = []
    # read double-bond E/Z FIRST — FindMolChiralCenters re-perceives stereo and clears bond flags
    for b in mol.GetBonds():
        s = b.GetStereo()
        if s == Chem.BondStereo.STEREOE:
            parts.append((b.GetBeginAtomIdx(), "E"))
        elif s == Chem.BondStereo.STEREOZ:
            parts.append((b.GetBeginAtomIdx(), "Z"))
    for idx, code in Chem.FindMolChiralCenters(mol, includeUnassigned=True,
                                               useLegacyImplementation=False):
        parts.append((idx, code))
    parts.sort()
    codes = [c for _, c in parts]
    return "(" + ",".join(codes) + ")" if codes else "(achiral)"


def stereoisomers(smiles, max_isomers=24):
    """Enumerate EVERY stereoisomer of a structure — all tetrahedral (R/S) and double-bond (E/Z)
    combinations, not just one R/S pair. Returns a list of {smiles (isomeric), inchikey, label,
    n_stereo}, capped at max_isomers so a molecule with many centers can't blow up. Empty when the
    molecule has no stereochemistry to vary."""
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    ez = sum(1 for b in mol.GetBonds() if b.GetStereo() != Chem.BondStereo.STEREONONE
             or (b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
                 and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1))
    if not centers and ez == 0:
        return []
    # onlyUnassigned=False -> flip ALL centers/bonds (every isomer), unique to dedupe meso forms
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=True, maxIsomers=max_isomers)
    out, seen = [], set()
    for iso in EnumerateStereoisomers(mol, options=opts):
        Chem.AssignStereochemistry(iso, cleanIt=True, force=True)
        smi = Chem.MolToSmiles(iso)
        ik = Chem.MolToInchiKey(iso)
        if ik in seen:
            continue
        seen.add(ik)
        out.append({"smiles": smi, "inchikey": ik, "label": _stereo_label(iso),
                    "n_stereo": len(centers) + ez})
    out.sort(key=lambda r: r["label"])
    return out


def stability(mol):
    def hits(d):
        return [n for n, p in d.items() if p is not None and mol.HasSubstructMatch(p)]
    return {
        "oxidation_watch": hits(_OX),
        "hydrolysis_watch": hits(_HY),
        "photodegradation_watch": hits(_PH),
        "note": "qualitative 'watch for' flags from reactive motifs — not a shelf-life prediction",
    }


def chemesthesis(mol):
    """Trigeminal/chemesthetic class flags (cooling/pungent/astringent) — qualitative."""
    classes = []
    if _ISOTHIOCYANATE is not None and mol.HasSubstructMatch(_ISOTHIOCYANATE):
        classes.append("pungent (isothiocyanate — mustard/wasabi type)")
    if _PHENOL is not None and len(mol.GetSubstructMatches(_PHENOL)) >= 3:
        classes.append("astringent (polyphenol/tannin-like)")
    ik = Chem.MolToInchiKey(mol).split("-")[0]
    if ik in _COOLING_IKS:
        classes.append("cooling (TRPM8 — menthol type)")
    if ik in _PUNGENT_IKS:
        classes.append("pungent/warming (TRPV1/TRPA1 — capsaicin/piperine type)")
    return {"classes": classes,
            "note": "curated structural / known-compound class flags, qualitative"}


def ttc_hint(mol):
    """PRELIMINARY toxicological-concern tier — NOT validated Cramer classification.

    Conservative heuristic (errs toward higher concern). Use Toxtree for the real
    Cramer/TTC call; this is a first-glance indicator only.
    """
    alerts = _tox_alerts(mol)
    elements = {a.GetSymbol() for a in mol.GetAtoms()}
    uncommon = elements - {"C", "H", "O", "N", "S", "P", "Cl", "Na", "K"}
    if alerts or uncommon:
        tier = "III — higher concern (structural alert or uncommon element)"
    elif elements <= {"C", "H", "O"} and Descriptors.MolWt(mol) < 200:
        tier = "I — lower concern (simple, common-element structure)"
    else:
        tier = "II — intermediate (review)"
    return {"preliminary_tier": tier,
            "drivers": {"alerts": alerts, "uncommon_elements": sorted(uncommon)},
            "note": "PRELIMINARY heuristic, not validated Cramer/TTC — use Toxtree for the real call"}


def retention_index(mol):
    """GC-MS Kovats retention index — a trained-QSPR task (solid on public NIST
    data). Hook for a loaded model; honest stub until one is wired in."""
    return {"kovats_ri": None,
            "note": "needs a trained RI QSPR (public data exists) — not estimated here"}


def analyze_balance(ingredients):
    """Rank a formulation by aroma impact and flag overbearing components.

    ingredients: list of {"smiles": str, "ppm": float (optional), "name": str (optional)}

    Quantitative when odor thresholds are loaded — odor activity value
    OAV = concentration / detection threshold; the highest-OAV component
    dominates the blend. Falls back to a qualitative volatility ranking when no
    thresholds are loaded. Also flags any dose above a loaded FEMA max use level.
    This ranks SINGLE-MOLECULE impact; it does NOT predict finished-blend
    perception (suppression/synergy need panel data — see the paid pilot).
    """
    rows = []
    for ing in ingredients:
        m = Chem.MolFromSmiles(ing.get("smiles", ""))
        if m is None:
            rows.append({"input": ing, "error": "unparseable SMILES"})
            continue
        meas = _measured(m)
        ppm = ing.get("ppm")
        thr = meas.get("odor_threshold_ppm")
        oav = (ppm / thr) if (ppm is not None and thr) else None
        over = (ppm > meas["fema_use_max_ppm"]) if (ppm is not None and meas.get("fema_use_max_ppm")) else None
        rows.append({
            "name": ing.get("name"), "smiles": Chem.MolToSmiles(m), "ppm": ppm,
            "odor_threshold_ppm": thr, "OAV": round(oav, 2) if oav is not None else None,
            "volatility": physchem(m)["qualitative"]["aroma_volatility"],
            "over_fema_max": over,
        })
    warnings = []
    have = [r for r in rows if r.get("OAV")]
    if have:
        have.sort(key=lambda r: r["OAV"], reverse=True)
        total = sum(r["OAV"] for r in have)
        top = have[0]
        if total > 0 and top["OAV"] / total > 0.6:
            warnings.append(
                f"{top['name'] or top['smiles']} dominates (~{round(100 * top['OAV'] / total)}% "
                "of total odor activity) — likely overbearing")
        ranking = [{"name": r["name"] or r["smiles"], "OAV": r["OAV"]} for r in have]
        basis = "quantitative (OAV = ppm / odor threshold)"
    else:
        order = {"high": 0, "moderate": 1, "low": 2}
        sr = sorted((r for r in rows if "volatility" in r),
                    key=lambda r: order.get(r["volatility"].split()[0], 3))
        ranking = [{"name": r["name"] or r["smiles"], "volatility": r["volatility"]} for r in sr]
        basis = "qualitative (volatility tier — load odor thresholds for quantitative OAV)"
    for r in rows:
        if r.get("over_fema_max"):
            warnings.append(f"{r['name'] or r['smiles']}: {r['ppm']} ppm exceeds loaded FEMA max use level")
    return {
        "per_ingredient": rows,
        "impact_ranking": ranking,
        "basis": basis,
        "balance_warnings": warnings,
        "scope_note": "Ranks single-molecule odor impact; does NOT predict finished-blend "
                      "perception (suppression/synergy need panel data).",
        "disclaimer": SAFETY_DISCLAIMER,
    }


# Canonical one-line odor-descriptor blurbs — the SINGLE source for the "what does this
# note smell like" help text shown under each aroma bar (workbench), in /api/aroma &
# /api/predict, and in the MCP server / skill output. Keep one entry per shipped head.
AROMA_DESC = {
    "odorless": "odorless — no documented smell (water, salts, most sugars, involatile solids); the aroma parallel to tasteless",
    "pungent": "sharp / irritating — acrid bite",
    "sweet": "sweet-smelling — indicative; clears the bar with odorless negatives, weaker at sweet-vs-other-sweet-odors",
    "ammoniacal": "ammonia / amine — sharp, pungent",
    "fruity": "ripe fruit — esters & lactones",
    "ethereal": "light & volatile — fresh, solvent-like",
    "phenolic": "phenolic / carbolic — phenol, cresols & alkylphenols",
    "sulfurous": "eggy / alliaceous — sulfur volatiles",
    "floral": "flowery — rose, jasmine, violet character",
    "acidic": "acidic / vinegar — short-chain carboxylic acids (acetic family)",
    "garlic": "allium — pungent sulfur",
    "fishy": "amine / trimethylamine — marine",
    "camphor": "camphoraceous — cooling, penetrating",
    "fatty": "oily / tallowy — long-chain aldehydes & acids",
    "almond": "marzipan — benzaldehyde, nutty-sweet",
    "minty": "cooling mint — menthol / carvone family",
    "spicy": "warm spice — eugenol / cinnamaldehyde / piperine / cuminaldehyde family",
    "petroleum": "solvent / naphtha — hydrocarbon character",
    "cherry": "cherry — benzaldehyde / almond-fruity aromatics",
    "grassy": "grassy — mown hay, cis-3-hexenol",
    "fresh": "clean / airy — light aldehydes & dihydromyrcenol",
    "grape": "grape / foxy — anthranilate esters (methyl anthranilate)",
    "berry": "berry — strawberry / raspberry furanones & esters",
    "putrid": "putrid — decay / rotten off-note",
    "orange": "sweet orange — limonene, decanal & orange esters",
    "muguet": "muguet / lily-of-the-valley — hydroxycitronellal & floral aldehydes",
    "waxy": "waxy / fatty — long-chain aldehydes, acids & alcohols",
    "alcoholic": "boozy / ethanolic — spirituous",
    "citrus": "lemon / orange peel — bright, zesty terpenes",
    "herbal": "green-herb / medicinal — thymol, carvacrol, cineole",
    "earthy": "soil / beetroot — geosmin-like",
    "meaty": "savoury / cooked-meat — sulfur volatiles (furanthiols, thiazoles, methional)",
    "musky": "musk — macrocyclic ketones / lactones",
    "green": "green — fresh-cut leaf, grassy aldehydes",
    "woody": "woody — cedar / sandalwood character",
    "pine": "pine / resin — coniferous terpenes (α-pinene)",
    "winey": "fermented / vinous — ethyl esters & lactate",
    "burnt": "burnt / roasted — pyrolysis furanones & roast pyrazines",
    "tropical": "tropical — pineapple / mango / passionfruit esters & thioesters",
    "wintergreen": "wintergreen / teaberry — methyl salicylate & salicylate esters",
    "lavender": "lavender — linalool & linalyl esters",
    "rose": "rosy floral — geraniol / phenylethanol",
    "nutty": "roasted nut / hazelnut — alkylpyrazines",
    "anise": "anise / licorice — anethole & anisyl aromatics",
    "cheesy": "cheesy / fermented — short & branched fatty acids",
    "creamy": "creamy / milky — γ & δ dairy lactones",
    "soapy": "soapy — C10–C12 fatty aldehydes & ketones",
    "jasmine": "jasmine — jasmonoids (hedione, cis-jasmone) & floral esters",
    "neroli": "orange-blossom / neroli — anthranilates & indole over a terpene-alcohol base",
    "vanilla": "vanilla — vanillin & guaiacol-derived phenolic aldehydes",
    "balsamic": "sweet-resinous balsam — benzyl / cinnamyl esters",
    "smoky": "smoke / phenolic — guaiacol & alkylphenols",
    "banana": "ripe banana — isoamyl acetate & branched esters",
    "vegetable": "green vegetable — methoxypyrazines & sulfides",
    "bready": "bread / toasted — pyrazines, pyrrolines & furfural",
    "melon": "melon / cucumber — (E,Z)-nonadienals",
    "cassis": "blackcurrant / cassis — sulfury cassis thiol & berry esters",
    "fennel": "fennel / anise-spice — anethole & terpene spice",
    "buttery": "butter / cream — vicinal diketones (diacetyl, acetoin)",
    "coconut": "creamy coconut — γ / δ lactones (nonalactone, decalactone)",
    "apple": "apple — green-fruity esters (ethyl 2-methylbutyrate, hexyl acetate)",
    "coffee": "roasted coffee — furfurylthiols & roast pyrazines",
    "peach": "peach — γ / δ lactones (undecalactone) & fruity esters",
    "violet": "violet / orris — ionones & irones",
    "ginger": "ginger — gingerol / zingerone / zingiberene",
    "hay": "new-mown hay — dihydrocoumarin & hay lactones",
    "tonka": "tonka / coumarinic — dihydrocoumarin, sweet-hay",
    "caramel": "caramel — maltol / furaneol / cyclotene sugar-pyrolysis",
    "rancid": "rancid — oxidized fat, stale off-note",
    "onion": "alliaceous onion — di/propyl disulfides",
    "medicinal": "phenolic / clove — antiseptic edge",
    "honey": "sweet honey / beeswax — phenylacetic acid & esters",
    "clove": "clove — eugenol / isoeugenol / alkyl-guaiacol phenylpropanoids",
    "cinnamon": "cinnamon — cinnamaldehyde & esters",
    "tarry": "tar / creosote — heavy smoky phenols (cresols, catechol)",
    "pear": "pear — ethyl 2,4-decadienoate & light fruity esters",
    "apricot": "apricot — γ-lactones & fruity esters",
    "malty": "malty — Strecker aldehydes (2/3-methylbutanal)",
    "marine": "marine / oceanic — sulfur volatiles & cucumber aldehydes",
    "mushroom": "mushroom — C8 volatiles (1-octen-3-ol / -one)",
    "cardamom": "cardamom — cineole & terpinyl spice terpenoids",
    "plum": "plum / prune — dried-fruit lactones & esters",
    "tea": "tea — linalool / ionone / damascenone leaf notes",
    "elderflower": "elderflower — linalool / rose-oxide floral-green",
    "cocoa": "cocoa / chocolate — pyrazines & malty Strecker aldehydes",
    "aldehydic": "aldehydic — fatty-aldehyde sparkle (C8–C13 aldehydes)",
    "animalic": "animalic — indolic / civet / castoreum (indole, skatole, muscone)",
    "amber": "amber / ambergris — labdane & woody-amber (ambroxide, sclareolide)",
    "leathery": "leathery — quinolines & alkylphenols (suede / birch-tar)",
    "powdery": "powdery — soft orris / cosmetic (ionones, heliotropin, coumarin, musks)",
    "resinous": "resinous / incense — sesquiterpenes & resin acids (labdanum, olibanum)",
    "terpenic": "terpenic / turpentine — monoterpene hydrocarbons (pinenes, myrcene)",
    "ozonic": "ozonic / aquatic — fresh sea-air (Calone, helional, melonal)",
    "hyacinth": "hyacinth — green floral (phenylacetaldehyde family)",
    "cooling": "cooling — physiological coolants (menthol, menthyl lactate, WS-agents)",
    "metallic": "metallic — sharp blood / tin note (1-octen-3-one, epoxy-decenal)",
    "lactonic": "lactonic — creamy γ/δ-lactones (structural class; overlaps coconut/peach/creamy)",
    "popcorn": "popcorn / roasted-cereal — pyrazines & 2-acetylpyrroline",
    "celery": "celery / lovage — phthalides (butylphthalide, sedanolide)",
    "maple": "maple / fenugreek — sotolon & furanones (sweet-curry)",
    "chamomile": "chamomile — fruity-herbal angelate & tiglate esters",
    "tobacco": "tobacco — dry cured (megastigmatrienone, damascones)",
    "carnation": "carnation — spicy clove-floral (eugenol / isoeugenol over floral)",
    "mimosa": "mimosa — powdery-green anisic floral (anisaldehyde, heliotropin)",
    "magnolia": "magnolia — fresh lemony-floral (linalool, citral, dihydromyrcenol)",
    "narcissus": "narcissus — green-animalic floral (indole, p-cresol, cinnamic esters)",
    "saffron": "saffron — safranal & carotenoid-degradation ketones (oxophorones)",
    "gardenia": "gardenia — green-creamy white floral (styralyl acetate, cis-3-hexenyl esters)",
    "ylang": "ylang-ylang — narcotic phenolic-floral (p-cresyl ethers, benzyl benzoate)",
    "lilac": "lilac — soft floral (lilac aldehydes, terpineol, anisaldehyde)",
    "osmanthus": "osmanthus — apricot-floral (ionones, damascones, lactones)",
    "geranium": "geranium — rosy-green (geraniol, citronellol, rose oxide)",
    "eucalyptus": "eucalyptus — cineole / camphoraceous terpenoids",
    "corky": "corky / cork-taint — haloanisoles (TCA) & moldy phenols",
    "violetleaf": "violet leaf — green-cucumber (methyl octine carbonate, 2,6-nonadienal); distinct from powdery violet",
    "fir": "fir / pine-needle — bornyl acetate & conifer terpenes",
    "oakmoss": "oakmoss — mossy orcinol / orsellinate esters (fragrance; atranol restricted)",
    "vetiver": "vetiver — earthy-woody sesquiterpenes (vetivones, khusimol)",
    "patchouli": "patchouli — camphoraceous-woody (patchoulol, patchoulenes)",
    "sandalwood": "sandalwood — creamy-woody santalols (narrow single-scaffold — AUROC inflated by small n)",
    "cedarwood": "cedarwood — dry cedar (cedrol, cedrenes, thujopsene)",
    "cognac": "cognac — fruity-fermented ethyl esters (heptanoate–laurate)",
    "grapefruit": "grapefruit — citrus-thiol & nootkatone over limonene",
    "bergamot": "bergamot — linalyl acetate / linalool citrus",
    "mandarin": "mandarin / tangerine — sinensals over citrus terpenes",
    "lime": "lime — citral & terpinen-4-ol citrus",
    "yuzu": "yuzu — terpene-rich citrus (phellandrene, terpinolene)",
    "honeysuckle": "honeysuckle — nectar floral (hotrienol, jasmine lactone, lilac aldehyde)",
    "freesia": "freesia — soft ionone-green floral",
    "cyclamen": "cyclamen — aquatic-green floral (cyclamen aldehyde)",
    "linden": "linden / lime-blossom — honeyed floral (farnesol, decadienal); marginal (0.81)",
    "coriander": "coriander / cilantro — linalool over fatty (2E)-alkenals",
    "cumin": "cumin — cuminaldehyde & cymene terpenes",
    "passionfruit": "passionfruit — tropical thiols & esters (mercaptohexanol, oxathiane)",
    "rosemary": "rosemary — cineole / camphor / verbenone herb",
    "blackpepper": "black pepper — rotundone & peppery sesquiterpenes",
    "nutmeg": "nutmeg / mace — myristicin & terpene spice",
    "sage": "sage — thujones, camphor & cineole",
    "thyme": "thyme — thymol / carvacrol phenolic herb",
    "oregano": "oregano / marjoram — carvacrol & thymol",
    "mustard": "mustard / horseradish / wasabi — pungent isothiocyanates",
    "pineapple": "pineapple — allyl & methylthio esters (tropical)",
    "strawberry": "strawberry — furaneol & fruity esters; marginal (0.78)",
    "raspberry": "raspberry — raspberry ketone, ionones & damascones",
    "myrrh": "myrrh — furanosesquiterpenes (narrow single-scaffold — AUROC inflated by small n)",
    "frankincense": "frankincense / olibanum — incensole & resin terpenes",
    "fig": "fig — green-lactonic fruit (hexenals, decalactones); marginal (0.72)",
    "mango": "mango — tropical (car-3-ene, terpinolene, lactones & esters)",
    "turmeric": "turmeric — ar-turmerone & curcuma sesquiterpenes",
    "davana": "davana — davanone & davana ether (fruity-balsamic; narrow scaffold)",
    "costus": "costus — costunolide & sesquiterpene lactones",
    "truffle": "truffle — dimethyl-polysulfides & dithiapentane (savoury-sulfurous)",
    "clarysage": "clary sage — sclareol / linalyl acetate (amber-herbal)",
    "elemi": "elemi — elemol & elemicin (lemon-resinous)",
    "labdanum": "labdanum / cistus — labdane amber-resins",
    "styrax": "styrax / storax — cinnamate esters (sweet-balsamic)",
    "opoponax": "opoponax — bisabolene & santalol resins",
    "champaca": "champaca — magnolia-type floral (methyl anthranilate, ionones)",
    "blueberry": "blueberry — fruity esters & cinnamates (ethyl 2-methylbutanoate)",
    "guava": "guava — tropical sulfur-esters (3-sulfanylhexyl acetate) & green",
    "tomato": "tomato — green-vegetal (cis-3-hexenal, 2-isobutylthiazole)",
    "juniper": "juniper / gin — piney terpenes (pinene, myrcene, terpinen-4-ol)",
    "hazelnut": "hazelnut — filbertone & roasted pyrazines (narrow — AUROC inflated by small n)",
    "allspice": "allspice / pimento — eugenol & clove-spice terpenes",
    "dill": "dill — carvone & dill ether",
}


@lru_cache(maxsize=8192)
def _aroma_scores_canon(canon):
    """Run all 172 descriptor forests for a CANONICAL SMILES and return {head: score}."""
    m = Chem.MolFromSmiles(canon)
    if m is None or not _AROMA_MODELS:
        return None
    fp = _feat(m)
    # Fan the 172 forests across cores — each predict_proba releases the GIL, so this turns the
    # ~40 s serial read (the only remaining cost, for genuinely novel/out-of-corpus molecules) into
    # a couple of seconds. In-corpus molecules never reach here (they read the precomputed index row).
    def _score(it):
        name, clf = it
        return name, round(float(clf.predict_proba(fp)[0][1]), 3)

    return dict(_infer_pool().map(_score, list(_AROMA_MODELS.items())))


def _aroma_scores(smiles):
    """The expensive part of the aroma read: all 172 descriptor forests → {head: score}. Keyed on
    the CANONICAL SMILES (not threshold/top_k, not the raw string) so every caller shares one
    computation per molecule regardless of how they spelled it — predict_aroma, _query_profile
    (substitutes) and the /api/aroma endpoint all collapse to the same cache entry instead of each
    re-running 172 forests (that double/mismatched inference was the ~6 s /api/substitutes).

    In-corpus molecules skip the forests entirely: their 172 scores are read straight off the
    precomputed profile index (built at startup) — the same numbers, ~40 s cheaper on a cold hit."""
    m = Chem.MolFromSmiles(smiles)
    if m is None or not _AROMA_MODELS:
        return None
    row = _index_row(m)
    if row is not None:
        profiles, dims = _SUB_INDEX[4], _SUB_INDEX[5]
        return {d.split(":", 1)[1]: round(float(profiles[row][j]), 3)
                for j, d in enumerate(dims) if d.startswith("aroma:")}
    return _aroma_scores_canon(Chem.MolToSmiles(m))


def _head_threshold(meta, name, override=None):
    """The probability at or above which a head counts as FIRING.

    Not a flat 0.5. Each head carries its own threshold, fitted on out-of-fold predictions at
    training time (train_aroma._calibrate) and stored in its manifest. The thin heads need this:
    with 13 positives against 2400 negatives a forest hedges, so a genuine pine match can land at
    0.42 and a flat cut-off would silently withhold it — while a head with 800 positives has no
    such problem and keeps a threshold near 0.5.

    This decides only whether a descriptor is marked *confident*. The raw probability is returned
    and displayed either way, so nothing is hidden and nothing is inflated.
    """
    if override is not None:
        return override
    t = meta.get(name, {}).get("threshold")
    return float(t) if isinstance(t, (int, float)) else 0.5


def _head_capable(meta, name):
    """Whether this head may be presented as making a CONFIDENT call.

    False for heads that never reach 50% out-of-fold precision at any threshold — they are right
    less than half the time when they fire, so calling them confident would be a lie no matter
    where the cut-off sits. Such a head keeps its score and its place in the profile (it is still
    real evidence, and often far better than the base rate); it is reported as INDICATIVE instead.
    Heads trained before calibration shipped have no flag, and are trusted as before.
    """
    return meta.get(name, {}).get("confident_capable", True)


@lru_cache(maxsize=8192)
def predict_aroma(smiles, top_k=8, threshold=None):
    """Predicted odor descriptors from RandomForest heads trained on the PUBLIC-DOMAIN HSDB
    odor corpus (see docs/AROMA.md). Returns the descriptors the model scores above threshold,
    each with its probability and the head's CV-AUROC. This is PRESENCE/ABSENCE (the free-text
    corpus carries no intensity), not a scored intensity map — honest about that ceiling; a
    stronger intensity model needs licensed (PMP 2001) or customer panel data. Returns
    available:False until the heads are trained into aroma_models/ (train_aroma.py)."""
    if Chem.MolFromSmiles(smiles) is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    scores = _aroma_scores(smiles)
    if scores is None:
        return {"available": False,
                "note": "aroma model not trained here — build with train_aroma.py"}
    preds = []
    for name, p in scores.items():
        thr = _head_threshold(_AROMA_META, name, threshold)
        fires, capable = p >= thr, _head_capable(_AROMA_META, name)
        preds.append({"odor": name, "score": p, "threshold": thr,
                      # a head that never reaches 50% out-of-fold precision fires as INDICATIVE,
                      # never as confident — see _head_capable
                      "confident": fires and capable,
                      "indicative": fires and not capable,
                      "precision": _AROMA_META.get(name, {}).get("cv_precision"),
                      "auroc": _AROMA_META.get(name, {}).get("auroc"),
                      "desc": AROMA_DESC.get(name)})
    preds.sort(key=lambda d: -d["score"])
    # Return EVERY head (like the taste meters list every taste), ranked, each flagged confident
    # or not — so the read shows the full aroma profile across all trained descriptor models, not
    # just the ones that fired. `top` is the confident shortlist for compact tag uses elsewhere.
    confident = [d for d in preds if d["confident"]]
    indicative = [d for d in preds if d["indicative"]]
    return {"available": True, "predicted": True, "descriptors": preds,
            "top": (confident or preds[:3]), "any_confident": bool(confident),
            "indicative": indicative,
            "note": "presence/absence model on public-domain HSDB odor text; not intensity"}


# Plain-language meaning of each mouthfeel / chemesthesis head (trigeminal sensations).
_MOUTHFEEL_DESC = {
    "cooling": "cooling — TRPM8 coolants (menthol, WS-agents); a physiological cool, not just a cool smell",
    "pungent": "pungent — TRPV1 / mustard-oil heat & bite (capsaicinoids, isothiocyanates, allium sulfur)",
    "warming": "warming — capsaicinoid & warm-spice heat (chili, pepper, ginger, cinnamon)",
    "astringent": "astringent — tannins & polyphenols; the puckering, mouth-drying sensation",
    "tingling": "tingling — paresthesia alkylamides (Sichuan-pepper sanshools, jambu spilanthol)",
}


def predict_mouthfeel(mol):
    """Predicted MOUTHFEEL / chemesthesis descriptors from the trained mouthfeel heads (trigeminal
    sensations — cooling, pungent, warming, astringent, tingling). Same presence/absence stack as
    taste & aroma, on curated public-domain agents; each with its score + CV-AUROC. available:False
    until trained into mouthfeel_models/ (train_mouthfeel.py)."""
    if not _MOUTHFEEL_MODELS:
        return {"available": False, "note": "mouthfeel heads not trained — run train_mouthfeel.py"}
    x = _feat(mol)
    preds = []
    for name, clf in sorted(_MOUTHFEEL_MODELS.items()):
        p = round(float(clf.predict_proba(x)[0, 1]), 3)
        thr = _head_threshold(_MOUTHFEEL_META, name)
        fires, capable = p >= thr, _head_capable(_MOUTHFEEL_META, name)
        preds.append({"sensation": name, "score": p, "threshold": thr,
                      "confident": fires and capable, "indicative": fires and not capable,
                      "precision": _MOUTHFEEL_META.get(name, {}).get("cv_precision"),
                      "auroc": _MOUTHFEEL_META.get(name, {}).get("auroc"),
                      "desc": _MOUTHFEEL_DESC.get(name)})
    preds.sort(key=lambda d: -d["score"])
    confident = [d for d in preds if d["confident"]]
    return {"available": True, "descriptors": preds, "top": confident,
            "any_confident": bool(confident),
            "note": "trigeminal/chemesthesis heads on curated public-domain agents; presence/absence"}


# Plain-language meaning of each Tox21 assay, for caution context.
_TOX_MEANING = {
    "NR-AhR": "aryl-hydrocarbon receptor (xenobiotic / dioxin-like activity)",
    "NR-AR": "androgen receptor", "NR-AR-LBD": "androgen receptor (LBD)",
    "NR-Aromatase": "aromatase (estrogen synthesis)",
    "NR-ER": "estrogen receptor", "NR-ER-LBD": "estrogen receptor (LBD)",
    "NR-PPAR-gamma": "PPAR-γ (metabolic)",
    "SR-ARE": "oxidative-stress response (ARE)",
    "SR-ATAD5": "ATAD5 — genotoxicity / DNA damage",
    "SR-HSE": "heat-shock response", "SR-MMP": "mitochondrial toxicity",
    "SR-p53": "p53 — DNA-damage response (genotoxic stress)",
}


def predict_tox(mol, threshold=None):
    """Caution-only in-vitro tox-assay activity (Tox21 models). INDICATIVE flags for
    review — NEVER a toxicity/safety determination. Honest/empty if heads untrained.

    Each assay fires at its OWN calibrated threshold, and calibration matters more here than
    anywhere else in the app: assay actives are rare, so a flat 0.5 made several heads
    over-flag. Every one of the twelve calibrated UPWARD (NR-AR to 0.69, NR-ER to 0.63) — the
    opposite direction from the thin aroma heads. A caution flag that cries wolf is worse than
    no flag, because it teaches people to ignore the ones that matter.
    """
    if not _TOX_MODELS:
        return {"available": False,
                "note": "tox heads not trained — run train_tox.py (Tox21, public domain)"}
    x = _fp(mol)
    assays = []
    for name, clf in sorted(_TOX_MODELS.items()):
        p = round(float(clf.predict_proba(x)[0, 1]), 3)
        thr = _head_threshold(_TOX_META, name, threshold)
        assays.append({"assay": name, "meaning": _TOX_MEANING.get(name, name), "probability": p,
                       "threshold": thr, "flagged": p >= thr,
                       "precision": _TOX_META.get(name, {}).get("cv_precision"),
                       "auroc": _TOX_META.get(name, {}).get("auroc")})
    flags = [a["assay"] for a in assays if a["flagged"]]
    return {"available": True, "assays": assays, "flags": flags,
            "note": "INDICATIVE in-vitro tox-assay activity (Tox21 RandomForest heads) — "
                    "caution-only, NOT a toxicity/safety determination; confirm with a toxicologist."}


def _taste_profile(out):
    """Trained taste heads ranked by probability (descending) — the 'order of
    dominance' view. Sour is a small-data indicative head; the deterministic
    sour/salty rules remain separate flags (out['sour'], out['salty'])."""
    ranked = []
    for t in ("sweet", "bitter", "umami"):
        v = out.get(t)
        if isinstance(v, (int, float)):
            ranked.append({"taste": t, "probability": round(float(v), 3), "basis": "trained"})
    sp = out.get("sour_predicted")
    if isinstance(sp, (int, float)):
        ranked.append({"taste": "sour", "probability": round(float(sp), 3),
                       "basis": "trained (indicative)"})
    ranked.sort(key=lambda e: e["probability"], reverse=True)
    return ranked


# --- Substitution search (issue #22) -------------------------------------------
# "Find me a molecule that behaves like X." Nearest-neighbor search over our
# labeled molecules by Morgan/Tanimoto similarity — the reformulation / cost-down
# tool (swap an expensive or supply-constrained ingredient for a close analogue,
# with its known tastes shown). This is the clean Track-A core; the product
# (Track B, #22) mirrors it as a pgvector ANN query over the same fingerprints.
# lazily built: (fps, smiles, known_tastes, predicted_aromas, profiles, profile_dims)
#   profiles: an (N x D) float32 matrix of predicted head SCORES — taste heads then aroma heads —
#   the "flavor profile" vector used for profile-based substitutes (vs the fingerprint fps used for
#   structural neighbors). profile_dims labels the columns.
_SUB_INDEX = None
_SUB_LOCK = _threading.Lock()  # guards the one-time index build against concurrent callers


def _profile_heads():
    """The ordered head list backing a flavor-profile vector: taste, then aroma, then mouthfeel —
    each sorted by name so the order is deterministic (the parallel index builder derives the same
    order straight from the head filenames without loading a single model)."""
    return sorted(_CLASSIFIERS), sorted(_AROMA_MODELS), sorted(_MOUTHFEEL_MODELS)


# Aroma heads to ALSO surface under mouthfeel. Now empty: cooling & pungent have their own dedicated
# mouthfeel_models heads (the SENSATION, trained on TRPM8/TRPV1 agents), separate from the aroma
# odour-descriptor heads of the same name — exactly as taste:sweet is separate from aroma:sweet.
_MOUTHFEEL_HEADS = set()


def head_catalog():
    """Every model head grouped by category (taste / aroma / mouthfeel / safety) with its held-out
    AUROC where known. Categories are TAGS, not buckets — a head can appear in more than one (e.g.
    cooling is aroma + mouthfeel). Powers the modal Heads card and the library category pickers."""
    taste_heads, aroma_heads, mouthfeel_heads = _profile_heads()

    def _taste_auroc(t):
        meta = _TASTE_META.get(t) if isinstance(_TASTE_META, dict) else None
        return meta.get("auroc") if isinstance(meta, dict) else None

    def _cal(meta, h):
        """The head's published calibration: where its bar sits, how precise it is there, and
        whether it may be shown as confident. Surfaced so the catalog is auditable — a head that
        fires at 0.16 and is right 6% of the time should say so on its own row, not only inside a
        molecule read."""
        m = meta.get(h, {})
        return {"threshold": m.get("threshold"), "precision": m.get("cv_precision"),
                "recall": m.get("cv_recall"), "n_pos": m.get("n_pos"),
                "confident_capable": m.get("confident_capable", True)}

    def _aroma(a):
        return {"head": a, "auroc": _AROMA_META.get(a, {}).get("auroc"),
                "desc": AROMA_DESC.get(a), **_cal(_AROMA_META, a)}

    # mouthfeel = the aroma heads tagged mouthfeel (cooling/pungent) + the dedicated mouthfeel heads
    mouthfeel = [_aroma(a) for a in aroma_heads if a in _MOUTHFEEL_HEADS]
    mouthfeel += [{"head": h, "auroc": _MOUTHFEEL_META.get(h, {}).get("auroc"),
                   "desc": AROMA_DESC.get(h), **_cal(_MOUTHFEEL_META, h)}
                  for h in mouthfeel_heads]
    return {
        "taste": [{"head": t, "auroc": _taste_auroc(t), **_cal(_TASTE_META, t)}
                  for t in taste_heads],
        "aroma": [_aroma(a) for a in aroma_heads],
        "mouthfeel": mouthfeel,
        "safety": [{"head": t, "auroc": _TOX_META.get(t, {}).get("auroc"),
                    "meaning": _TOX_MEANING.get(t, t), **_cal(_TOX_META, t)}
                   for t in sorted(_TOX_MODELS)],
    }


def _build_sub_index():
    global _SUB_INDEX
    import numpy as np
    # Fast path: load the precomputed profile index (build_profile_index.py). The 183-dim
    # inference over ~8.8k molecules is slow (~3 min); the cache makes startup instant. We only
    # rebuild the cheap Morgan fingerprints from SMILES on load.
    cache = artifact("profile_index.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        smis = [str(s) for s in z["smiles"]]
        tastes = [[t for t in str(s).split(",") if t.strip()] for s in z["taste_documented"]]
        aromas = [[a for a in str(s).split(",") if a.strip()] for s in z["aromas"]]
        fps = [_MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]
        _SUB_INDEX = (fps, smis, tastes, aromas, z["profiles"], list(z["dims"]))
        return
    fps, smis, tastes, feats = [], [], [], []
    profiles, profile_dims = None, []
    src = _UNIVERSE if _UNIVERSE.exists() else _MASTER
    if src.exists():
        import pandas as pd
        m = pd.read_parquet(src)
        basic = [t for t in ("sweet", "bitter", "umami", "sour", "salty") if t in m.columns]
        has_documented = "taste_documented" in m.columns  # enrichment: comma-separated string
        seen_skel = set()
        for _, r in m.iterrows():
            mol = Chem.MolFromSmiles(str(r["smiles"]))
            if mol is None:
                continue
            skel = Chem.MolToInchiKey(mol).split("-")[0]
            if skel in seen_skel:  # dedupe by connectivity so the universe doesn't repeat a molecule
                continue
            seen_skel.add(skel)
            fps.append(_MORGAN.GetFingerprint(mol))
            smis.append(Chem.MolToSmiles(mol))
            if has_documented:
                tastes.append([t for t in str(r.get("taste_documented") or "").split(",") if t.strip()])
            else:
                tastes.append([t for t in basic if r.get(t) == 1])
            feats.append(_feat(mol)[0])
        # One-time batched inference over the whole labeled set. We keep BOTH:
        #   - the thresholded confident-aroma NAME list per molecule (for display / palette match)
        #   - the full head-SCORE matrix (taste heads + aroma heads) for profile-based substitutes.
        # Kept at n_jobs=1: a single big predict_proba is already vectorized C; joblib fan-out here
        # thrashed under concurrency. Reusing this index is what makes the endpoints fast.
        aromas = [[] for _ in smis]
        if feats:
            X = np.vstack(feats)
            taste_heads, aroma_heads, mouthfeel_heads = _profile_heads()
            cols = [_CLASSIFIERS[t].predict_proba(X)[:, 1] for t in taste_heads]
            for name in aroma_heads:
                col = _AROMA_MODELS[name].predict_proba(X)[:, 1]
                cols.append(col)
                # per-head threshold, same as a live read — otherwise the precomputed chip list
                # and the on-the-fly one disagree for exactly the thin heads this fixes
                thr = _head_threshold(_AROMA_META, name)
                for i in range(len(smis)):
                    if col[i] >= thr:
                        aromas[i].append(name)
            cols += [_MOUTHFEEL_MODELS[h].predict_proba(X)[:, 1] for h in mouthfeel_heads]
            profile_dims = ([f"taste:{t}" for t in taste_heads] + [f"aroma:{a}" for a in aroma_heads]
                            + [f"mouthfeel:{h}" for h in mouthfeel_heads])
            profiles = np.column_stack(cols).astype("float32") if cols else None
    else:
        aromas = []
    _SUB_INDEX = (fps, smis, tastes, aromas, profiles, profile_dims)


def _ensure_sub_index():
    if _SUB_INDEX is None:  # double-checked lock: a request during the startup build waits for
        with _SUB_LOCK:      # that one build instead of kicking off a second (which would thrash cores)
            if _SUB_INDEX is None:
                _build_sub_index()


_PN_CACHE = {}
_SKEL2ROW = {}


def _index_row(mol):
    """Row of `mol` in the profile index (matched by connectivity skeleton), or None if the
    molecule isn't in the reference corpus. In-corpus molecules can reuse their PRECOMPUTED
    183-dim profile (built once at index build / startup) instead of re-running 172 forests at
    query time — that inference is ~40 s cold on a novel molecule and was the real /api/substitutes
    and include_aroma cost. The precomputed row is the SAME model output, just paid up front."""
    _ensure_sub_index()
    smis, profiles = _SUB_INDEX[1], _SUB_INDEX[4]
    if profiles is None or not smis:
        return None
    key = id(smis)
    table = _SKEL2ROW.get(key)
    if table is None:
        table = {}
        for i, s in enumerate(smis):
            mi = Chem.MolFromSmiles(s)
            if mi is not None:
                table.setdefault(Chem.MolToInchiKey(mi).split("-")[0], i)
        _SKEL2ROW.clear()  # one live index at a time; don't leak on rebuild
        _SKEL2ROW[key] = table
    return table.get(Chem.MolToInchiKey(mol).split("-")[0])


def is_precomputed(smiles):
    """True if this molecule's full taste+aroma profile is already in the index (an instant read),
    False if it's out-of-corpus and the 183 profile heads have to run fresh (the slower path). Used by the
    UI to decide whether to show the 'conjuring a fresh reading' note while a read brews."""
    mol = Chem.MolFromSmiles(smiles or "")
    return mol is not None and _index_row(mol) is not None


def _profiles_unit(profiles):
    """L2-normalized rows of the reference profile matrix, computed once and reused (the query
    cosine is then just a matmul). Keyed on the matrix's identity so it rebuilds only if the
    index is rebuilt — paying this at index build / prewarm, not on every /api/substitutes."""
    import numpy as np
    key = id(profiles)
    pn = _PN_CACHE.get(key)
    if pn is None:
        pn = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-9)
        _PN_CACHE.clear()  # only ever one live index; don't leak on rebuild
        _PN_CACHE[key] = pn
    return pn


def _query_profile(mol):
    """The taste + aroma + mouthfeel profile vector for a query molecule, in the index column order.
    In-corpus molecules read their precomputed row (instant); novel molecules run the heads (taste +
    the memoized aroma core shared with /api/aroma + the small mouthfeel set)."""
    import numpy as np
    row = _index_row(mol)
    if row is not None:
        return np.asarray(_SUB_INDEX[4][row], dtype="float32")  # precomputed profile row
    taste_heads, aroma_heads, mouthfeel_heads = _profile_heads()
    x = _feat(mol)
    tvals = [float(_CLASSIFIERS[t].predict_proba(x)[0, 1]) for t in taste_heads]  # 6 heads, cheap
    ascore = _aroma_scores(Chem.MolToSmiles(mol)) or {}  # memoized aroma core, shared with /api/aroma
    avals = [float(ascore.get(a, 0.0)) for a in aroma_heads]
    mvals = [float(_MOUTHFEEL_MODELS[h].predict_proba(x)[0, 1]) for h in mouthfeel_heads]  # few heads
    return np.array(tvals + avals + mvals, dtype="float32")


def _predicted_tastes_at(profiles, i):
    """The PREDICTED taste read for reference-set row i, from the profile matrix — the top few
    taste heads by score (not a hard >=0.5 cut, which mostly surfaced only 'bitter'). There are
    only 6 taste heads, so we return the top 3 that carry any signal (>= 0.2), highest first, to
    give a fuller taste picture. The taste columns are the first len(_CLASSIFIERS) of the vector."""
    if profiles is None:
        return []
    taste_heads = sorted(_CLASSIFIERS)
    row = profiles[i]
    scored = sorted(((float(row[j]), t) for j, t in enumerate(taste_heads)), reverse=True)
    return [t for s, t in scored[:3] if s >= 0.2]


def _predicted_mouthfeel_at(profiles, i):
    """The MOUTHFEEL read for reference-set row i, straight off the profile matrix — no extra
    inference, the columns are already there. Each sensation must clear its OWN calibrated
    threshold (and be confident-capable), so a card never shows a sensation the modal would
    call indicative. Returns the firing sensations, strongest first."""
    if profiles is None:
        return []
    taste_heads, aroma_heads, mouth_heads = _profile_heads()
    base = len(taste_heads) + len(aroma_heads)   # mouthfeel columns follow taste then aroma
    row, out = profiles[i], []
    for j, name in enumerate(mouth_heads):
        col = base + j
        if col >= len(row):
            break
        score = float(row[col])
        if score >= _head_threshold(_MOUTHFEEL_META, name) and _head_capable(_MOUTHFEEL_META, name):
            out.append((score, name))
    return [n for _, n in sorted(out, reverse=True)]


def structural_neighbors(smiles: str, k: int = 8, min_similarity: float = 0.0) -> dict:
    """STRUCTURAL neighbors: the k labeled molecules most structurally similar to the query
    (Tanimoto over Morgan fingerprints), each with its known tastes. Structural look-alikes —
    contrast with profile-based `substitutes` (taste/aroma-alikes). Returns {'neighbors': [...]}
    ranked by similarity (self excluded)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    _ensure_sub_index()
    fps, smis, tastes, _aromas, profiles, _dims = _SUB_INDEX
    if not fps:
        return {"neighbors": [], "note": "no reference set loaded (taste_master.parquet absent)"}
    q = _MORGAN.GetFingerprint(mol)
    self_smi = Chem.MolToSmiles(mol)
    # exclude self by InChIKey SKELETON (connectivity), not the canonical-SMILES string: the query's
    # canonicalization can differ from the stored form, which let the identical molecule slip back in
    # at 100%. The Morgan fingerprint is achiral, so a same-skeleton neighbor is structurally the
    # query (or a stereoisomer it can't distinguish) — either way not a substitute.
    self_skel = Chem.MolToInchiKey(mol).split("-")[0]
    sims = DataStructs.BulkTanimotoSimilarity(q, fps)
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    neighbors = []
    for i in order:
        if sims[i] < min_similarity:
            continue
        ni = Chem.MolFromSmiles(smis[i])
        if ni is None or Chem.MolToInchiKey(ni).split("-")[0] == self_skel:
            continue
        neighbors.append({"smiles": smis[i], "similarity": round(float(sims[i]), 3),
                          "known_tastes": tastes[i],
                          "predicted_tastes": _predicted_tastes_at(profiles, i),
                          "mouthfeel": _predicted_mouthfeel_at(profiles, i),
                          # confident aromas precomputed once in the index — reused so the
                          # endpoint never re-runs the 24 aroma heads per neighbor (8x ~1.3s saved)
                          "aromas": _aromas[i] if i < len(_aromas) else []})
        if len(neighbors) >= k:
            break
    return {"query": self_smi, "neighbors": neighbors,
            "basis": "structural — Tanimoto / Morgan r2 2048-bit over labeled molecules"}


# back-compat alias: the endpoint / callers historically called this `substitute`
substitute = structural_neighbors


def substitutes(smiles: str, k: int = 8, min_match: float = 0.0) -> dict:
    """PROFILE-based substitutes: the molecules whose predicted FLAVOR profile (taste + aroma
    head scores) is closest to the query's — the drop-in reformulation list. A molecule that
    *tastes and smells* like the target is a likely substitute regardless of its structure, so
    this ranks by cosine similarity over the head-score vectors (not fingerprint distance).
    Returns every match with profile_match >= min_match (ranked, self excluded), capped at k so
    callers can scroll the qualifying set rather than a fixed count."""
    import numpy as np
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    _ensure_sub_index()
    _fps, smis, tastes, aromas, profiles, _dims = _SUB_INDEX
    if profiles is None or not len(smis):
        return {"substitutes": [], "note": "no reference set / models loaded"}
    qv = _query_profile(mol)
    qn = qv / (float(np.linalg.norm(qv)) + 1e-9)
    sims = _profiles_unit(profiles) @ qn
    self_skel = Chem.MolToInchiKey(mol).split("-")[0]
    subs = []
    for i in np.argsort(-sims):
        if sims[i] < min_match:  # sorted descending — nothing further qualifies
            break
        ni = Chem.MolFromSmiles(smis[i])
        if ni is None or Chem.MolToInchiKey(ni).split("-")[0] == self_skel:
            continue
        subs.append({"smiles": smis[i], "profile_match": round(float(sims[i]), 3),
                     "known_tastes": tastes[i], "predicted_tastes": _predicted_tastes_at(profiles, i),
                     "mouthfeel": _predicted_mouthfeel_at(profiles, i),
                     "aromas": aromas[i] if i < len(aromas) else []})
        if len(subs) >= k:
            break
    return {"query": Chem.MolToSmiles(mol), "substitutes": subs,
            "basis": "profile — cosine over predicted taste + aroma head scores"}


def mixture_to_molecule(smiles_list: list, weights: list | None = None, k: int = 6) -> dict:
    """Collapse a blend into a single equivalent molecule: average the components' taste+aroma
    profile vectors (dose-weighted if weights given) into one target profile, then return the k
    molecules whose own profile is closest — 'one molecule that tastes and smells like the whole
    blend'. The inverse of a recipe: instead of many ingredients, find the single closest match."""
    import numpy as np
    _ensure_sub_index()
    _fps, smis, tastes, aromas, profiles, _dims = _SUB_INDEX
    if profiles is None or not len(smis):
        return {"error": "no profile index / reference set"}
    comps = []
    for smi in smiles_list or []:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        comps.append((Chem.MolToSmiles(m), _query_profile(m)))
    if not comps:
        return {"error": "no parseable components"}
    w = np.array((weights or [1.0] * len(comps))[:len(comps)], dtype="float32")
    w = w / (float(w.sum()) + 1e-9)
    target = np.average(np.vstack([v for _, v in comps]), axis=0, weights=w).astype("float32")
    in_skel = {Chem.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0] for s, _ in comps}
    qn = target / (float(np.linalg.norm(target)) + 1e-9)
    pn = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-9)
    sims = pn @ qn
    out = []
    for i in np.argsort(-sims):
        ni = Chem.MolFromSmiles(smis[i])
        if ni is None or Chem.MolToInchiKey(ni).split("-")[0] in in_skel:
            continue
        out.append({"smiles": smis[i], "profile_match": round(float(sims[i]), 3),
                    "known_tastes": tastes[i], "predicted_tastes": _predicted_tastes_at(profiles, i),
                     "mouthfeel": _predicted_mouthfeel_at(profiles, i),
                    "aromas": aromas[i] if i < len(aromas) else []})
        if len(out) >= k:
            break
    return {"components": [s for s, _ in comps], "equivalents": out,
            "basis": "cosine of each candidate to the dose-weighted mean blend profile"}


def palette_match(tastes, aromas=None, k=5):
    """Single molecules that best resemble a target flavor PALETTE — taste labels AND aroma
    descriptors — scored by the mean of taste-Jaccard and aroma-Jaccard over the labeled set
    (whichever targets you give). NOT a blend-perception model (a blend's palette isn't the
    union of its parts — suppression/synergy); an honest label-set approximation for the 'one
    molecule like this mixture' view. Taste is documented ground truth; aroma is the model's
    predicted descriptors."""
    if _SUB_INDEX is None:
        _build_sub_index()
    _, smis, tlist, alist, _profiles, _dims = _SUB_INDEX
    t_target, a_target = set(tastes or []), set(aromas or [])
    if not (t_target or a_target) or not smis:
        return {"target": {"tastes": sorted(t_target), "aromas": sorted(a_target)}, "matches": []}
    scored = []
    for smi, ts, ars in zip(smis, tlist, alist):
        s, a = set(ts), set(ars)
        parts = []
        if t_target:
            parts.append(len(t_target & s) / len(t_target | s) if (t_target | s) else 0.0)
        if a_target:
            parts.append(len(a_target & a) / len(a_target | a) if (a_target | a) else 0.0)
        score = sum(parts) / len(parts) if parts else 0.0
        if score > 0:
            scored.append((score, smi, sorted(s), sorted(a)))
    scored.sort(key=lambda e: -e[0])
    return {"target": {"tastes": sorted(t_target), "aromas": sorted(a_target)},
            "matches": [{"smiles": sm, "tastes": ts, "aromas": ar, "match": round(sc, 2)}
                        for sc, sm, ts, ar in scored[:k]]}


# --- Reaction-template augmentation for the mixture screen -------------------------------------
# INDICATIVE forward reaction templates (RDKit reaction SMARTS) for the flavor-relevant
# condensations that can occur when ingredients sit together — esters (fruity notes forming),
# Schiff bases (the Maillard first step), (hemi)acetals, hemithioacetals. This AUGMENTS the
# documented-hazard lookup (it never replaces it): "these two could plausibly react to form X",
# not "this is safe". A template firing means the functional groups are present, not that the
# reaction proceeds under any given condition.
_RXN_TEMPLATES = [
    ("esterification — acid + alcohol → ester (a fruity note forms)",
     "[CX3:1](=[OX1:2])[OX2H1].[OX2H1][#6:3]>>[C:1](=[O:2])O[#6:3]"),
    ("Schiff base — aldehyde + primary amine → imine (Maillard first step)",
     "[CX3H1:1]=[OX1:2].[NX3;H2:3][#6:4]>>[C:1]=[N:3][#6:4]"),
    ("hemiacetal — aldehyde + alcohol",
     "[CX3H1:1]=[OX1:2].[OX2H1:3][#6:4]>>[C:1]([OX2H1])[O:3][#6:4]"),
    ("hemithioacetal — aldehyde + thiol (savory/allium precursors)",
     "[CX3H1:1]=[OX1:2].[SX2H1:3][#6:4]>>[C:1]([OX2H1])[S:3][#6:4]"),
]
_RXNS = None


def _rxns():
    global _RXNS
    if _RXNS is None:
        from rdkit.Chem import AllChem
        out = []
        for name, sm in _RXN_TEMPLATES:
            with contextlib.suppress(Exception):  # a template that won't parse is just skipped
                out.append((name, AllChem.ReactionFromSmarts(sm)))
        _RXNS = out
    return _RXNS


def reaction_products(smiles_list, max_products=12):
    """Plausible template products of combining the given molecules (pairwise). INDICATIVE only —
    'the groups to form this are present', not 'it happens'. Returns [{smiles, reaction, reactants}]."""
    mols = [(s, Chem.MolFromSmiles(s)) for s in smiles_list]
    mols = [(s, m) for s, m in mols if m is not None]
    seen, out = set(), []
    for name, rxn in _rxns():
        for si, mi in mols:
            for sj, mj in mols:
                if si == sj:
                    continue
                runs = None
                with contextlib.suppress(Exception):
                    runs = rxn.RunReactants((mi, mj))
                if runs is None:
                    continue
                for prods in runs:
                    for p in prods:
                        smi = None
                        with contextlib.suppress(Exception):  # template made an invalid product
                            Chem.SanitizeMol(p)
                            smi = Chem.MolToSmiles(p)
                        if smi is None:
                            continue
                        pm = Chem.MolFromSmiles(smi)
                        if pm is None:
                            continue
                        ik = Chem.MolToInchiKey(pm)
                        if ik in seen or ik in {Chem.MolToInchiKey(m) for _, m in mols}:
                            continue  # skip dups and products identical to an input
                        seen.add(ik)
                        out.append({"smiles": smi, "reaction": name,
                                    "reactants": [Chem.MolToSmiles(mi), Chem.MolToSmiles(mj)]})
                        if len(out) >= max_products:
                            return out
    return out


def predict(smiles: str, include_aroma: bool = False) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    x = _feat(mol)  # taste heads + intensity regressor use fingerprint + physicochemical block
    out = {"smiles": Chem.MolToSmiles(mol)}
    # Applicability domain: the trained taste/tox heads are fit on ORGANIC tastants.
    # For inorganic / carbon-free molecules (water, O2, N2, NaCl, ...) their output is
    # meaningless, so flag it; the demo suppresses the trained heads for these.
    out["applicability"] = {
        "in_domain": _has_carbon(mol),
        "note": ("Trained taste/tox heads are fit on organic molecules; predictions for "
                 "inorganic / carbon-free structures are outside their domain — the rules "
                 "(sour/salty), structure, and computed properties remain valid."),
    }
    for name, clf in sorted(_CLASSIFIERS.items()):
        out[name] = round(float(clf.predict_proba(x)[0, 1]), 3)
    # per-head held-out CV-AUROC (from taste_models/manifest.json) so the UI can show how
    # trustworthy each trained taste head is, the same way the aroma descriptors carry theirs
    out["taste_meta"] = {t: {"auroc": _TASTE_META[t]["auroc"]}
                         for t in _CLASSIFIERS if t in _TASTE_META and "auroc" in _TASTE_META[t]}
    # Sour AND salty train as INDICATIVE heads, but their boolean stays the RULE's call
    # below — keep the model probabilities separately as sour_predicted / salty_predicted.
    if "sour" in out:
        out["sour_predicted"] = out.pop("sour")
    if "salty" in out:
        out["salty_predicted"] = out.pop("salty")
    if _INTENSITY is not None:
        out["sweet_intensity"] = round(float(_INTENSITY.predict(x)[0]), 2)
    out.update(_sour(mol))
    out.update(_salty(mol))
    # Known dataset labels are ground truth: if our data says salty, trust it over
    # the rule (and mark it so the UI shows verified-fact, not prediction).
    known = _KNOWN.get(Chem.MolToInchiKey(mol), {})
    if known.get("salty"):
        out["salty"] = True
        out["salty_reason"] = "verified (dataset label)"
    if known:
        out["known_tastes"] = sorted(known)
    # If two+ taste heads both fire high, surface that as a complex-taste note —
    # the model-side echo of ChemTastesDB's 'multitaste' class.
    strong = [t for t in ("sweet", "bitter", "umami")
              if isinstance(out.get(t), float)
              and out[t] >= _head_threshold(_TASTE_META, t)]
    out["multitaste"] = len(strong) >= 2
    out["taste_profile"] = _taste_profile(out)
    out["physchem"] = physchem(mol)
    out["stability"] = stability(mol)
    out["chemesthesis"] = chemesthesis(mol)
    out["mouthfeel"] = predict_mouthfeel(mol)  # trained trigeminal heads (cooling/pungent/warming/…)
    out["chirality"] = chirality(mol)
    out["analytical"] = {"retention_index": retention_index(mol)}
    out["labeling"] = labeling(mol)
    out["safety"] = _safety(mol)
    out["safety"]["ttc_hint"] = ttc_hint(mol)
    out["safety"]["tox_screen"] = predict_tox(mol)
    if include_aroma:
        out["aroma"] = predict_aroma(smiles)
    return out


if __name__ == "__main__":
    import json
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "OC(=O)CC(O)(CC(=O)O)C(=O)O"  # citric acid
    print(json.dumps(predict(s), indent=2))
