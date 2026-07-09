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

    out = pd.DataFrame(rows)
    out.to_parquet("master_enrichment.parquet")
    named = int(out["name"].notna().sum())
    print(f"master_enrichment.parquet: {len(out)} molecules, {named} named, "
          f"{int((out['aroma_top'] != '').sum())} with a dominant aroma, "
          f"{int(out['gras'].sum())} GRAS")
