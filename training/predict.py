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

from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, DataStructs, Descriptors, rdFingerprintGenerator, rdMolDescriptors

FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
TASTE = Path("taste_models")

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

# load whatever classifier heads exist (sweet/bitter/umami...) + intensity
_CLASSIFIERS = {}
_INTENSITY = None
if TASTE.exists():
    for p in TASTE.glob("*_rf.joblib"):
        name = p.stem.replace("_rf", "")
        if name == "sweet_intensity":
            _INTENSITY = joblib.load(p)
        else:
            _CLASSIFIERS[name] = joblib.load(p)

# Caution-only toxicity-assay heads (Tox21, public domain). Loaded if trained.
# INDICATIVE in-vitro signals — never a toxicity determination.
_TOX_MODELS = {}
_TOX_DIR = Path("tox_models")
if _TOX_DIR.exists():
    for p in _TOX_DIR.glob("*_rf.joblib"):
        _TOX_MODELS[p.stem.replace("_rf", "")] = joblib.load(p)

# Known-label lookup: ground truth for molecules we actually have data on. This
# is how the salty/sour data works as a FLAG without a model — if a queried
# molecule is in our labeled set, we report the verified fact instead of a guess.
_KNOWN = {}  # inchikey -> {taste: 1}
_MASTER = Path("taste_master.parquet")
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
_GRAS_FILE = Path("gras_reference.parquet")
if _GRAS_FILE.exists():
    import pandas as pd  # noqa: F811
    _g = pd.read_parquet(_GRAS_FILE)
    if "inchikey" in _g.columns:
        _GRAS = {str(k).split("-")[0] for k in _g["inchikey"].dropna()}

# Optional measured-property + dosing table. Data-gated like GRAS. Drop
# properties.(parquet|csv) with an 'inchikey' column and any of:
# odor_threshold_ppm, fema_use_max_ppm, boiling_point_c, vapor_pressure_pa.
# We use MEASURED values (lookup) rather than structure estimates for these,
# because structure-based volatility (e.g. Joback) is too inaccurate for flavor
# molecules to report as a number — benzaldehyde misses by ~90 C.
_PROPS = {}
_PROP_COLS = ("odor_threshold_ppm", "fema_use_max_ppm", "boiling_point_c",
              "boiling_point_pressure_mmhg", "vapor_pressure_pa")
for _ext in ("properties.parquet", "properties.csv"):
    _pf = Path(_ext)
    if _pf.exists():
        import pandas as pd  # noqa: F811
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
    """Defensive 'is this even a recognized food ingredient?' check."""
    if not _GRAS:
        return "no GRAS reference loaded — not checked"
    ik = Chem.MolToInchiKey(mol).split("-")[0]
    return "in GRAS/flavor reference" if ik in _GRAS else "NOT in reference — unverified for food use"


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
    return result


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


def predict_aroma(smiles, top_k=8):
    """Aroma is deferred. No commercially-clean *public* odor data yields a working
    model (see docs/AROMA.md), so rather than fabricate smells we return an honest
    'not available'. A real head gets trained on licensed (PMP 2001) or customer
    odor data — OpenPOM's MIT architecture for large sets, RandomForest for small —
    and wired in here then. predict() never calls this unless include_aroma=True."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    return {"available": False,
            "note": "aroma deferred — needs licensed/customer odor data; see docs/AROMA.md"}


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


def predict_tox(mol, threshold=0.5):
    """Caution-only in-vitro tox-assay activity (Tox21 models). INDICATIVE flags for
    review — NEVER a toxicity/safety determination. Honest/empty if heads untrained."""
    if not _TOX_MODELS:
        return {"available": False,
                "note": "tox heads not trained — run train_tox.py (Tox21, public domain)"}
    x = _fp(mol)
    assays = []
    for name, clf in sorted(_TOX_MODELS.items()):
        p = round(float(clf.predict_proba(x)[0, 1]), 3)
        assays.append({"assay": name, "meaning": _TOX_MEANING.get(name, name), "probability": p})
    flags = [a["assay"] for a in assays if a["probability"] >= threshold]
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
_SUB_INDEX = None  # lazily built: (list[bitvect], list[smiles], list[known_tastes])


def _build_sub_index():
    global _SUB_INDEX
    fps, smis, tastes = [], [], []
    if _MASTER.exists():
        import pandas as pd  # noqa: F811
        m = pd.read_parquet(_MASTER)
        basic = [t for t in ("sweet", "bitter", "umami", "sour", "salty") if t in m.columns]
        for _, r in m.iterrows():
            mol = Chem.MolFromSmiles(str(r["smiles"]))
            if mol is None:
                continue
            fps.append(_MORGAN.GetFingerprint(mol))
            smis.append(Chem.MolToSmiles(mol))
            tastes.append([t for t in basic if r.get(t) == 1])
    _SUB_INDEX = (fps, smis, tastes)


def substitute(smiles: str, k: int = 8, min_similarity: float = 0.0) -> dict:
    """Nearest-neighbor substitution: the k labeled molecules most structurally
    similar to the query (Tanimoto over Morgan fingerprints), each with its known
    tastes. The reformulation / cost-down tool — swap an ingredient for a close
    analogue. Returns {'neighbors': [...]} ranked by similarity (self excluded)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    if _SUB_INDEX is None:
        _build_sub_index()
    fps, smis, tastes = _SUB_INDEX
    if not fps:
        return {"neighbors": [], "note": "no reference set loaded (taste_master.parquet absent)"}
    q = _MORGAN.GetFingerprint(mol)
    self_smi = Chem.MolToSmiles(mol)
    sims = DataStructs.BulkTanimotoSimilarity(q, fps)
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    neighbors = []
    for i in order:
        if smis[i] == self_smi or sims[i] < min_similarity:
            continue
        neighbors.append({"smiles": smis[i], "similarity": round(float(sims[i]), 3),
                          "known_tastes": tastes[i]})
        if len(neighbors) >= k:
            break
    return {"query": self_smi, "neighbors": neighbors,
            "basis": "Tanimoto / Morgan r2 2048-bit over labeled molecules"}


def palette_match(tastes, k=5):
    """Single molecules whose KNOWN taste-label set best matches a target taste set
    (Jaccard over sweet/bitter/umami/sour/salty). NOT a blend-perception model — a
    label-set similarity over the labeled molecules, for the 'one molecule like this
    mixture' view. A blend's actual palette isn't the union of its parts (suppression /
    synergy); this is an honest structural-label approximation."""
    if _SUB_INDEX is None:
        _build_sub_index()
    _, smis, tlist = _SUB_INDEX
    target = set(tastes)
    if not target or not smis:
        return {"target": sorted(target), "matches": []}
    scored = []
    for smi, ts in zip(smis, tlist):
        s = set(ts)
        if not s:
            continue
        j = len(target & s) / len(target | s)
        if j > 0:
            scored.append((j, smi, sorted(s)))
    scored.sort(key=lambda e: -e[0])
    return {"target": sorted(target),
            "matches": [{"smiles": sm, "tastes": ts, "match": round(j, 2)} for j, sm, ts in scored[:k]]}


def predict(smiles: str, include_aroma: bool = False) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": f"unparseable SMILES: {smiles}"}
    x = _fp(mol)
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
    # Sour trains as a small-data INDICATIVE head, but its boolean stays the rule's
    # call below — keep the model probability separately as sour_predicted.
    if "sour" in out:
        out["sour_predicted"] = out.pop("sour")
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
              if isinstance(out.get(t), float) and out[t] >= 0.5]
    out["multitaste"] = len(strong) >= 2
    out["taste_profile"] = _taste_profile(out)
    out["physchem"] = physchem(mol)
    out["stability"] = stability(mol)
    out["chemesthesis"] = chemesthesis(mol)
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
