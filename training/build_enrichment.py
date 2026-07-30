"""
build_enrichment.py — the master enrichment table (one rich row per unique molecule).

Assembles EVERYTHING we know about every molecule that has a structure into one table
(master_enrichment.parquet) so the workbench can browse the whole universe as a sortable,
searchable grid: identity (name, SMILES), taste (documented + predicted), dominant aroma,
computed physicochemistry (MW, logP, TPSA, H-bond donors/acceptors, rotatable bonds, rings),
measured properties from the PubChem crawl (melting/boiling point), and GRAS status.

Model inference is BATCHED (one predict_proba per head over the whole matrix). Reads the same
feature block the heads were trained on (chemfeatures) via predict._feat. Re-run it any time the
crawl fills in more names / MP / BP — the table just gets richer.

Usage: python build_enrichment.py            # -> master_enrichment.parquet
"""
import contextlib
import glob
import re

import numpy as np
import pandas as pd
import predict as P
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
_BASIC = ["sweet", "bitter", "umami", "sour", "salty"]


def _all_structures():
    out = {}
    for f in sorted(glob.glob("*.parquet")):
        if f in ("master_enrichment.parquet", "flavor_map.parquet"):
            continue
        d = None
        with contextlib.suppress(Exception):
            d = pd.read_parquet(f)
        if d is None:
            continue
        if "smiles" not in d.columns:
            continue
        for s in d["smiles"].dropna().unique():
            m = Chem.MolFromSmiles(str(s))
            if m is not None:
                out.setdefault(Chem.MolToInchiKey(m).split("-")[0], Chem.MolToSmiles(m))
    return out


def _by_skel(path, cols):
    """{skeleton -> {col: val}} from a parquet keyed by full inchikey."""
    out = {}
    try:
        d = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 — table absent; return what we have
        return out
    if "inchikey" not in d.columns:
        return out
    have = [c for c in cols if c in d.columns]
    for _, r in d.iterrows():
        ik = r["inchikey"]
        if isinstance(ik, str):
            out[ik.split("-")[0]] = {c: r[c] for c in have}
    return out


def _documented_isomer_rows(name_by_skel):
    """First-class rows for the stereoisomers that genuinely DIFFER in documented odor/taste
    (R- vs S-carvone, (-)- vs (+)-menthol…). Most stereoisomers share their parent's data and
    stay folded into one skeleton; these few carry their own cited notes, so they earn their own
    row — searchable, sortable, openable like any molecule."""
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
    try:
        import build_aroma_dataset as BA  # odor text -> descriptor tag
    except Exception:  # noqa: BLE001 — optional module; degrade without odor tags
        BA = None
    docs = {}  # skeleton -> {full_ik: {smiles, odor?, taste?}}
    for path, col in (("odor_notes.parquet", "odor"), ("taste_notes.parquet", "taste")):
        d = None
        with contextlib.suppress(Exception):
            d = pd.read_parquet(path)
        if d is None:
            continue
        if col not in d.columns:
            continue
        for ik, smi, txt in zip(d["inchikey"], d["smiles"], d[col]):
            if isinstance(ik, str) and isinstance(smi, str) and isinstance(txt, str) and txt.strip():
                rec = docs.setdefault(ik.split("-")[0], {}).setdefault(ik, {"smiles": smi})
                rec[col] = txt.strip().split("\n")[0][:40]
    rows = []
    for skel, isos in docs.items():
        distinct = {(v.get("odor"), v.get("taste")) for v in isos.values()}
        if len(isos) < 2 or len(distinct) < 2:            # need >=2 isomers with DIFFERENT notes
            continue
        parent = name_by_skel.get(skel)
        for full_ik, v in isos.items():
            smi = v["smiles"]
            if not any(c in smi for c in "@/\\"):          # must actually carry stereochemistry
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            label = P._stereo_label(m)
            odor, taste = v.get("odor", ""), v.get("taste", "")
            aroma_top = ""
            if BA and odor:
                tags = sorted(BA.tag(odor))
                aroma_top = tags[0] if tags else ""
            rows.append({
                "inchikey_skel": full_ik, "smiles": Chem.MolToSmiles(m),
                "name": (f"{label} {parent}" if parent else f"{label} isomer"),
                "mw": round(float(Descriptors.MolWt(m)), 1),
                "logp": round(float(Crippen.MolLogP(m)), 2),
                "tpsa": round(float(rdMolDescriptors.CalcTPSA(m)), 1),
                "hbd": int(rdMolDescriptors.CalcNumHBD(m)), "hba": int(rdMolDescriptors.CalcNumHBA(m)),
                "rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(m)),
                "rings": int(rdMolDescriptors.CalcNumRings(m)),
                "melting_point_c": None, "boiling_point_c": None, "gras": skel in P._GRAS,
                "taste_documented": taste, "taste_predicted": "", "aroma_top": aroma_top,
                "p_sweet": 0.0, "p_bitter": 0.0, "p_umami": 0.0, "is_isomer": True,
            })
    return rows


def _pick_name(pr):
    """Best display name from a properties row: common name if there is one, else the IUPAC name.

    Written the long way ON PURPOSE. The obvious `pr.get("common_name") or pr.get("iupac_name")`
    is a real bug here: a missing pandas value is NaN, NaN is TRUTHY, so the `or` returns NaN and
    never falls through to the IUPAC name. That silently left ~500 molecules unnamed even though a
    perfectly good name sat in the table — they rendered as raw SMILES in the grid and on cards."""
    for key in ("common_name", "iupac_name"):
        v = pr.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _fallback_label(mol):
    """A readable label for a molecule that genuinely has no name anywhere.

    770 rows reach this. They are not flavour molecules we failed to look up — a full PubChem
    crawl on Title AND IUPACName resolved only 41 of 784, and an InChIKey lookup did worse. The
    remainder are the long tail that arrived with the broad food/safety universe: 159
    multi-component salts (PubChem indexes the components, not the mixture), 197 very large
    structures, 100 peptide-like, 13 carbon-free. No crawl will name those, and most are not
    flavour-relevant.

    Showing a raw SMILES for them is the worst option — it is unreadable and looks broken. A
    molecular formula is honest, compact, and tells a chemist something real, so that is the floor.
    Multi-component structures are labelled as such rather than pretending to be one substance.
    """
    if mol is None:
        return None
    parts = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    if len(parts) > 1:
        bits = [rdMolDescriptors.CalcMolFormula(x) for x in parts]
        return f"{formula} (mixture: {' + '.join(bits[:3])}{' + …' if len(bits) > 3 else ''})"
    return formula


_CAS_INVERTED = re.compile(r"^([A-Za-z0-9\-\[\]\(\)']+), ([0-9A-Za-z\-\(\),+\u00b1\s]+?)-?$")


def _uninvert_cas(name):
    """Un-invert CAS-style index names so they read forwards.

    PubChem/CAS list many compounds parent-first — "Carvone, (+-)-", "Limonene, (-)-",
    "Cyclohexanol, 5-methyl-2-(1-methylethenyl)-". That ordering exists for alphabetised print
    indexes and reads backwards to everyone else. Three cases, because they resolve differently:
      stereo descriptor   "Carvone, (+-)-"          -> "(+-)-Carvone"       (parent keeps its case)
      derivative suffix   "Linalool, oxide"          -> "Linalool oxide"
                          "2-Hexen-1-ol, 1-acetate"  -> "2-Hexen-1-ol 1-acetate"
      substituent prefix  "Cyclohexanol, 5-methyl-"  -> "5-methyl-cyclohexanol"
    Anything that doesn't match is returned untouched — a wrong "fix" is worse than none."""
    if not isinstance(name, str) or ", " not in name:
        return name
    m = _CAS_INVERTED.match(name.strip())
    if not m:
        return name
    parent, mod = m.group(1), m.group(2).strip().rstrip("-").strip()
    if not mod:
        return parent
    # 1. pure stereo / optical descriptor -> prefix, keep the parent's capitalisation
    if re.fullmatch(r"[\(\[][^)\]]*[\)\]]", mod) or mod.lower() in ("cis", "trans", "d", "l", "dl"):
        return f"{mod}-{parent}"
    # 2. a derivative/functional word -> it's a suffix, not a substituent
    if re.search(r"(acetate|oxide|ester|ether|hydrate|hydrochloride|anhydride|lactone|"
                 r"[a-z]+oate|[a-z]+ate|alcohol|aldehyde|ketone|acid)$", mod, re.IGNORECASE):
        return f"{parent} {mod}"
    # 3. otherwise a substituent prefix -> lowercase the parent, which is now mid-name
    return f"{mod}-{parent[0].lower()}{parent[1:]}"


def _curated_names():
    """{skeleton -> human name} from the curated CSVs (flavors + aroma/mouthfeel supplements).
    These are the names a flavorist actually uses — "gamma-nonalactone", "hydroxy-alpha-sanshool" —
    and for the molecules we hand-added they are usually the ONLY good name available."""
    import csv
    out = {}
    for path in ("flavors.csv", "aroma_supplement.csv", "mouthfeel_supplement.csv"):
        with contextlib.suppress(Exception), open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                nm = (r.get("molecule") or "").strip()
                m = Chem.MolFromSmiles((r.get("smiles") or "").strip())
                if nm and m is not None:
                    out.setdefault(Chem.MolToInchiKey(m).split("-")[0], nm)
    return out


def _taste_by_skel():
    out = {}
    try:
        tm = pd.read_parquet("taste_master.parquet")
    except Exception:  # noqa: BLE001 — table absent; return what we have
        return out
    for _, r in tm.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"]))
        if m is None:
            continue
        doc = [t for t in _BASIC if r.get(t) == 1]
        if doc:
            out[Chem.MolToInchiKey(m).split("-")[0]] = doc
    return out


if __name__ == "__main__":
    P.MODELS_READY.wait()  # predict loads heads on a background thread now — wait before we run inference
    structs = _all_structures()
    # fold in curated supplement molecules (aroma + mouthfeel + flavors) — CSVs the parquet glob
    # misses — so EVERY unique molecule we train on is accounted for in the universe (map/index/pickers)
    for _csv in ("aroma_supplement.csv", "mouthfeel_supplement.csv", "flavors.csv"):
        with contextlib.suppress(Exception):
            for _smi in pd.read_csv(_csv)["smiles"].dropna():
                _m = Chem.MolFromSmiles(str(_smi))
                if _m is not None:
                    structs.setdefault(Chem.MolToInchiKey(_m).split("-")[0], Chem.MolToSmiles(_m))
    props = _by_skel("properties.parquet",
                     ["common_name", "iupac_name", "melting_point_c", "boiling_point_c"])
    taste_doc = _taste_by_skel()
    curated = _curated_names()   # human names for the molecules we hand-curated
    print(f"{len(structs)} molecules; {len(props)} with crawl properties; "
          f"{len(taste_doc)} with documented taste; {len(curated)} curated names", flush=True)

    skels, smis, rows, feats = [], [], [], []
    for skel, smi in structs.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        pr = props.get(skel, {})
        # Curated FIRST: those names were hand-picked as what a flavorist calls the molecule,
        # and PubChem's "common_name" is often systematic anyway (spilanthol's is
        # "N-(2-Methylpropyl)-2,6,8-decatrienamide"). Only fall through to PubChem when we
        # haven't named it ourselves.
        name = _uninvert_cas(curated.get(skel) or _pick_name(pr))
        rows.append({
            "inchikey_skel": skel, "smiles": smi,
            # never emit a bare SMILES as a name: fall back to the molecular formula
            "name": name if isinstance(name, str) else _fallback_label(m),
            "mw": round(float(Descriptors.MolWt(m)), 1),
            "logp": round(float(Crippen.MolLogP(m)), 2),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(m)), 1),
            "hbd": int(rdMolDescriptors.CalcNumHBD(m)),
            "hba": int(rdMolDescriptors.CalcNumHBA(m)),
            "rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(m)),
            "rings": int(rdMolDescriptors.CalcNumRings(m)),
            "melting_point_c": pr.get("melting_point_c"),
            "boiling_point_c": pr.get("boiling_point_c"),
            "gras": skel in P._GRAS,
            "taste_documented": ",".join(taste_doc.get(skel, [])),
            "is_isomer": False,
        })
        skels.append(skel)
        smis.append(smi)
        feats.append(P._feat(m)[0])

    X = np.vstack(feats)
    print(f"running batched inference over {len(X)} molecules "
          f"({len(P._CLASSIFIERS)} taste + {len(P._AROMA_MODELS)} aroma heads)...", flush=True)

    # predicted taste (sweet/bitter/umami) — probability, and a compact 'taste_predicted' summary
    for t in ("sweet", "bitter", "umami"):
        clf = P._CLASSIFIERS.get(t)
        col = clf.predict_proba(X)[:, 1] if clf is not None else np.zeros(len(X))
        for i, r in enumerate(rows):
            r[f"p_{t}"] = round(float(col[i]), 3)
    # dominant predicted aroma (highest-prob head that clears 0.5, else '')
    best, bp = [""] * len(rows), [0.5] * len(rows)
    for name, clf in P._AROMA_MODELS.items():
        col = clf.predict_proba(X)[:, 1]
        for i in range(len(rows)):
            if col[i] >= bp[i]:
                bp[i], best[i] = col[i], name
    for i, r in enumerate(rows):
        r["aroma_top"] = best[i]
        r["taste_predicted"] = ",".join(t for t in ("sweet", "bitter", "umami")
                                        if r[f"p_{t}"] >= 0.5)

    # predicted mouthfeel (chemesthesis) + tox — stored per molecule so the browsable universe grid is
    # complete (taste + aroma + mouthfeel + safety). tox is CAUTION-ONLY (in-vitro assay activity), and
    # NEITHER mouthfeel here nor tox is part of the substitute-match profile vector — display only.
    for name, clf in sorted(P._MOUTHFEEL_MODELS.items()):
        col = clf.predict_proba(X)[:, 1]
        for i, r in enumerate(rows):
            r[f"mf_{name}"] = round(float(col[i]), 3)
    # tox heads were trained on the BARE 2048-bit Morgan fingerprint (_fp), not the _feat block the
    # taste/aroma/mouthfeel heads use — build a separate matrix so the feature width matches.
    xtox = np.vstack([P._fp(Chem.MolFromSmiles(s))[0] for s in smis])
    tox_cols = {}
    for name, clf in sorted(P._TOX_MODELS.items()):
        tox_cols[name] = clf.predict_proba(xtox)[:, 1]
    for i, r in enumerate(rows):
        for name, col in tox_cols.items():
            r[f"tox_{name}"] = round(float(col[i]), 3)
        r["tox_flags"] = ",".join(n for n, col in tox_cols.items() if col[i] >= 0.5)  # assays firing >=0.5

    # first-class rows for stereoisomers that differ in documented odor/taste
    name_by_skel = {sk: _uninvert_cas(curated.get(sk) or _pick_name(props.get(sk, {})))
                    for sk in structs}  # isomer labels only — a formula parent reads badly there
    iso_rows = _documented_isomer_rows(name_by_skel)
    if iso_rows:
        rows.extend(iso_rows)
    out = pd.DataFrame(rows)
    out.to_parquet("master_enrichment.parquet")
    named = int(out["name"].notna().sum())
    print(f"master_enrichment.parquet: {len(out)} molecules ({len(iso_rows)} distinct-documented "
          f"isomer rows), {named} named, {int((out['aroma_top'] != '').sum())} with a dominant "
          f"aroma, {int(out['gras'].sum())} GRAS")
