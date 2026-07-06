"""
build_flavor_map.py — a 2D "flavor-space" embedding of the labeled molecules.

Projects each molecule's 2048-bit Morgan fingerprint to 2D (PCA -> t-SNE) and labels it by
its dominant known taste, so structurally similar molecules land near each other and the taste
classes separate visually. Output flavor_map.parquet (smiles, label, x, y); the app normalizes
+ serves it as an interactive scatter (hover = name, click = full read).

Usage: python build_flavor_map.py            # taste_master.parquet -> flavor_map.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TASTES = ["sweet", "bitter", "umami", "sour", "salty"]
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fp(m):
    bv = _MORGAN.GetFingerprint(m)
    a = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, a)
    return a


if __name__ == "__main__":
    src = "taste_master.parquet"
    if not Path(src).exists():
        print(f"{src} not found")
        sys.exit(1)
    df = pd.read_parquet(src)
    smiles, labels, feats = [], [], []
    for _, r in df.iterrows():
        m = Chem.MolFromSmiles(str(r["smiles"]))
        if m is None:
            continue
        smiles.append(r["smiles"])
        labels.append(next((t for t in TASTES if r.get(t) == 1), "other"))
        feats.append(_fp(m))
    X = np.array(feats)
    print(f"embedding {len(X)} molecules (PCA-50 -> t-SNE 2D)...", flush=True)
    x50 = PCA(n_components=50, random_state=42).fit_transform(X)
    xy = TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(x50)
    out = pd.DataFrame({"smiles": smiles, "label": labels,
                        "x": xy[:, 0].astype(float), "y": xy[:, 1].astype(float)})
    out.to_parquet("flavor_map.parquet")
    print(f"flavor_map.parquet: {len(out)} points  "
          f"({', '.join(f'{t}={int((out.label==t).sum())}' for t in TASTES)}, "
          f"other={int((out.label=='other').sum())})")
