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
import glob

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit import RDLogger

import predict as P

RDLogger.DisableLog("rdApp.*")
_BASIC = ["sweet", "bitter", "umami", "sour", "salty"]


def _all_structures():
    out = {}
    for f in sorted(glob.glob("*.parquet")):
        if f in ("master_enrichment.parquet", "flavor_map.parquet"):
            continue
        try:
            d = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
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
    except Exception:  # noqa: BLE001
        BA = None
    docs = {}  # skeleton -> {full_ik: {smiles, odor?, taste?}}
    for path, col in (("odor_notes.parquet", "odor"), ("taste_notes.parquet", "taste")):
        try:
            d = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
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


def _taste_by_skel():
    out = {}
    try:
        tm = pd.read_parquet("taste_master.parquet")
    except Exception:  # noqa: BLE001
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
    structs = _all_structures()
    props = _by_skel("properties.parquet",
                     ["common_name", "iupac_name", "melting_point_c", "boiling_point_c"])
    taste_doc = _taste_by_skel()
    print(f"{len(structs)} molecules; {len(props)} with crawl properties; "
          f"{len(taste_doc)} with documented taste", flush=True)

    skels, smis, rows, feats = [], [], [], []
    for skel, smi in structs.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        pr = props.get(skel, {})
        name = pr.get("common_name") or pr.get("iupac_name")
        rows.append({
            "inchikey_skel": skel, "smiles": smi,
            "name": name if isinstance(name, str) else None,
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

    # first-class rows for stereoisomers that differ in documented odor/taste
    name_by_skel = {sk: (props.get(sk, {}).get("common_name") or props.get(sk, {}).get("iupac_name"))
                    for sk in structs}
    iso_rows = _documented_isomer_rows(name_by_skel)
    if iso_rows:
        rows.extend(iso_rows)
    out = pd.DataFrame(rows)
    out.to_parquet("master_enrichment.parquet")
    named = int(out["name"].notna().sum())
    print(f"master_enrichment.parquet: {len(out)} molecules ({len(iso_rows)} distinct-documented "
          f"isomer rows), {named} named, {int((out['aroma_top'] != '').sum())} with a dominant "
          f"aroma, {int(out['gras'].sum())} GRAS")
