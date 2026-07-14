"""
build_flavor_map.py — a flavor-space embedding of the labeled molecules (2D + 3D).

Projects each molecule's 2048-bit Morgan fingerprint into 2D and 3D with UMAP (Jaccard metric,
which is the right similarity for binary fingerprints — same family as Tanimoto), and labels it
by its dominant known taste. Structurally similar molecules land near each other and the taste
classes separate visually. Output flavor_map.parquet (smiles, label, x, y, x3, y3, z3); the app
normalizes + serves it as an interactive scatter (2D zoom/pan, or a rotatable 3D point cloud).

Needs `pip install umap-learn` (a build-time dependency only — the server just reads the
resulting parquet, so UMAP is not required to run the demo).

Usage: python build_flavor_map.py            # taste_master.parquet -> flavor_map.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

TASTES = ["sweet", "bitter", "umami", "sour", "salty", "tasteless"]
# _BASIC is only the tastes that ChemTastesDB (taste_master) actually columns. tasteless is a
# real taste head (it trains — see train_taste.BASIC) but taste_master carries no tasteless
# column, so on the map it's OVERLAID from documented-tasteless data below, not read from a column.
_BASIC = ["sweet", "bitter", "umami", "sour", "salty"]
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _tasteless_structs():
    """{InChIKey-skeleton -> canonical SMILES} for molecules documented as tasteless
    (PubChem/HSDB Taste). Used to LABEL and to ADD tasteless points — most documented-tasteless
    molecules aren't in ChemTastesDB (taste_master), so without adding them the class is empty."""
    import re
    try:
        tn = pd.read_parquet("taste_notes.parquet")
    except Exception:  # noqa: BLE001 — no documented-taste table
        return {}
    neg = re.compile(r"\b(tasteless|no taste|without taste)", re.I)
    out = {}
    for smi, txt in zip(tn["smiles"], tn["taste"].astype(str)):
        if neg.search(txt):
            m = Chem.MolFromSmiles(str(smi))
            if m is not None:
                out[Chem.MolToInchiKey(m).split("-")[0]] = Chem.MolToSmiles(m)
    return out


def _fp(m):
    bv = _MORGAN.GetFingerprint(m)
    a = np.zeros((2048,), dtype=np.float32)  # float for UMAP's jaccard metric
    DataStructs.ConvertToNumpyArray(bv, a)
    return a


def _umap(X, n):
    return umap.UMAP(n_components=n, n_neighbors=15, min_dist=0.15,
                     metric="jaccard", random_state=42).fit_transform(X)


def _taste_labels(path="taste_master.parquet"):
    """{InChIKey-skeleton -> dominant basic taste} from ChemTastesDB, so any molecule that also
    appears in other sets still gets its taste color on the map."""
    out = {}
    if not Path(path).exists():
        return out
    df = pd.read_parquet(path)
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"]))
        if m is None:
            continue
        lbl = next((t for t in _BASIC if r.get(t) == 1), None)
        if lbl:
            out[Chem.MolToInchiKey(m).split("-")[0]] = lbl
    return out


def _all_structures():
    """{InChIKey-skeleton -> canonical SMILES} for EVERY molecule with a structure across all
    datasets — so the map is the whole embeddable universe, not just the taste-labeled subset."""
    import glob
    out = {}
    for f in sorted(glob.glob("*.parquet")):
        if f in ("flavor_map.parquet",):
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


if __name__ == "__main__":
    if not Path("taste_master.parquet").exists():
        print("taste_master.parquet not found")
        sys.exit(1)
    taste = _taste_labels()
    tasteless = _tasteless_structs()
    structs = _all_structures()
    for skel, smi in tasteless.items():  # fold documented-tasteless in (many aren't elsewhere)
        structs.setdefault(skel, smi)
    smiles, labels, feats = [], [], []
    for skel, smi in structs.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        lbl = taste.get(skel) or ("tasteless" if skel in tasteless else "other")
        smiles.append(smi)
        labels.append(lbl)
        feats.append(_fp(m))
    X = np.array(feats)
    print(f"embedding {len(X)} molecules with UMAP (Jaccard) — 2D then 3D...", flush=True)
    xy = _umap(X, 2)
    xyz = _umap(X, 3)
    # real physicochemical coordinates too, so the map can offer an INTERPRETABLE-AXES view:
    # MW (x) vs logP (y) in 2D, and MW vs logP vs TPSA (z) in 3D — actual, labelled units.
    from rdkit.Chem import Crippen, Descriptors
    mw, logp, tpsa = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(str(s))
        mw.append(round(float(Descriptors.MolWt(m)), 1) if m else None)
        logp.append(round(float(Crippen.MolLogP(m)), 2) if m else None)
        tpsa.append(round(float(Descriptors.TPSA(m)), 1) if m else None)
    out = pd.DataFrame({"smiles": smiles, "label": labels,
                        "x": xy[:, 0].astype(float), "y": xy[:, 1].astype(float),
                        "x3": xyz[:, 0].astype(float), "y3": xyz[:, 1].astype(float),
                        "z3": xyz[:, 2].astype(float),
                        "mw": mw, "logp": logp, "tpsa": tpsa})
    # dominant aroma descriptor per molecule (for the map's "color by aroma" mode) — guarded so
    # the map still builds without the aroma heads
    try:
        import predict as P
        if P._AROMA_MODELS:
            heads = list(P._AROMA_MODELS.keys())
            # the aroma heads take fingerprint + physicochemical features (chemfeatures), not the
            # bare fp used for the Jaccard UMAP — build that feature matrix here
            mols = [Chem.MolFromSmiles(str(s)) for s in smiles]
            Xf = np.vstack([P._feat(m)[0] for m in mols])
            prob = {name: clf.predict_proba(Xf)[:, 1] for name, clf in P._AROMA_MODELS.items()}
            # DOCUMENTED aroma positives per molecule (skeleton -> shipped heads it's labelled with),
            # from the training set — so the molecules a head was TRAINED on surface as that head,
            # instead of being outshone by a commoner predicted aroma (why rare heads had no dots).
            doc = {}
            try:
                at = pd.read_parquet("aroma_train.parquet")
                hc = [h for h in heads if h in at.columns]
                for _, r in at.iterrows():
                    ik = str(r.get("inchikey", "")).split("-")[0]
                    pos = [h for h in hc if int(r.get(h, 0) or 0) == 1]
                    if ik and pos:
                        doc[ik] = pos
            except Exception:  # noqa: BLE001 — no training labels; predicted-only
                pass
            best = ["other"] * len(out)
            for i, m in enumerate(mols):
                ik = Chem.MolToInchiKey(m).split("-")[0] if m is not None else ""
                dp = doc.get(ik)
                if dp:                                   # documented aroma wins (tie-break by model)
                    best[i] = max(dp, key=lambda h: prob[h][i])
                else:                                    # else the strongest predicted head >= 0.5
                    bp = 0.5
                    for name in heads:
                        if prob[name][i] >= bp:
                            bp, best[i] = prob[name][i], name
            out["aroma_label"] = best
    except Exception:  # noqa: BLE001 — no aroma models; taste-only map
        pass
    out.to_parquet("flavor_map.parquet")
    print(f"flavor_map.parquet: {len(out)} points  "
          f"({', '.join(f'{t}={int((out.label == t).sum())}' for t in TASTES)}, "
          f"other={int((out.label == 'other').sum())})")
