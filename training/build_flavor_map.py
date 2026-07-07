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


if __name__ == "__main__":
    src = "taste_master.parquet"
    if not Path(src).exists():
        print(f"{src} not found")
        sys.exit(1)
    df = pd.read_parquet(src)
    tasteless = _tasteless_structs()
    smiles, labels, feats, seen = [], [], [], set()
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"]))
        if m is None:
            continue
        skel = Chem.MolToInchiKey(m).split("-")[0]
        seen.add(skel)
        smiles.append(r["smiles"])
        lbl = next((t for t in _BASIC if r.get(t) == 1), None)
        if lbl is None and skel in tasteless:
            lbl = "tasteless"
        labels.append(lbl or "other")
        feats.append(_fp(m))
    # add documented-tasteless molecules that ChemTastesDB doesn't carry, so the class is real
    for skel, smi in tasteless.items():
        if skel in seen:
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        seen.add(skel)
        smiles.append(smi)
        labels.append("tasteless")
        feats.append(_fp(m))
    X = np.array(feats)
    print(f"embedding {len(X)} molecules with UMAP (Jaccard) — 2D then 3D...", flush=True)
    xy = _umap(X, 2)
    xyz = _umap(X, 3)
    out = pd.DataFrame({"smiles": smiles, "label": labels,
                        "x": xy[:, 0].astype(float), "y": xy[:, 1].astype(float),
                        "x3": xyz[:, 0].astype(float), "y3": xyz[:, 1].astype(float),
                        "z3": xyz[:, 2].astype(float)})
    # dominant aroma descriptor per molecule (for the map's "color by aroma" mode) — guarded so
    # the map still builds without the aroma heads
    try:
        import predict as P
        if P._AROMA_MODELS:
            best, bp = ["other"] * len(out), [0.0] * len(out)
            for name, clf in P._AROMA_MODELS.items():
                probs = clf.predict_proba(X)[:, 1]
                for i in range(len(out)):
                    if probs[i] >= 0.5 and probs[i] > bp[i]:
                        bp[i], best[i] = probs[i], name
            out["aroma_label"] = best
    except Exception:  # noqa: BLE001 — no aroma models; taste-only map
        pass
    out.to_parquet("flavor_map.parquet")
    print(f"flavor_map.parquet: {len(out)} points  "
          f"({', '.join(f'{t}={int((out.label == t).sum())}' for t in TASTES)}, "
          f"other={int((out.label == 'other').sum())})")
