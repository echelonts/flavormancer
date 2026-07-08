"""
train_aroma.py — multi-label odor-descriptor CLASSIFIERS on Morgan fingerprints.

Reads aroma_train.parquet (build_aroma_dataset.py: public-domain HSDB odor text keyword-
normalized to presence/absence descriptor labels). Trains one RandomForest classifier per
descriptor that has enough positives, reports HONEST 5-fold CV AUROC, and keeps only the
descriptors that clear a minimum AUROC. Saves the kept heads to aroma_models/ + a manifest.

This is the same modeling stack as taste (Morgan FP + RandomForest), but PRESENCE/ABSENCE
per descriptor (the free text carries no intensity). It's the commercial-clean, public-domain
aroma read; a stronger intensity model still needs licensed/customer panel data.

Usage: python train_aroma.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from chemfeatures import descriptors as _desc

FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
MIN_POS = 20       # need enough positives for a stable 5-fold estimate
MIN_AUROC = 0.70   # below this the descriptor isn't learnable from structure -> don't ship it
OUT = Path("aroma_models")
OUT.mkdir(exist_ok=True)
for s in OUT.glob("*_clf.joblib"):
    s.unlink()


def fp(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    bv = _MORGAN.GetFingerprint(m)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return np.concatenate([arr, _desc(m)])  # fingerprint + physicochemical block (chemfeatures.py)


df = pd.read_parquet("aroma_train.parquet")
descriptors = [c for c in df.columns if c not in ("inchikey", "smiles")]
feats, keep = [], []
for i, s in enumerate(df["smiles"]):
    f = fp(s)
    if f is not None:
        feats.append(f)
        keep.append(i)
X = np.array(feats)
df = df.iloc[keep].reset_index(drop=True)

print(f"training odor-descriptor classifiers on {len(df)} molecules "
      f"(min {MIN_POS} positives, keep CV-AUROC >= {MIN_AUROC}):")
manifest = {"fp_bits": FP_BITS, "fp_radius": FP_RADIUS, "descriptors": {}}
for d in sorted(descriptors, key=lambda c: -int(df[c].sum())):
    y = df[d].values.astype(int)
    npos = int(y.sum())
    if npos < MIN_POS:
        continue
    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42,
                                 class_weight="balanced")
    auroc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1).mean()
    flag = "kept" if auroc >= MIN_AUROC else "drop (not learnable)"
    if auroc >= MIN_AUROC:
        clf.fit(X, y)
        joblib.dump(clf, OUT / f"{d}_clf.joblib")
        manifest["descriptors"][d] = {"auroc": round(float(auroc), 3), "n_pos": npos}
    print(f"  {d:12s} n_pos={npos:4d}  CV-AUROC={auroc:.3f}  -> {flag}")

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nkept {len(manifest['descriptors'])} descriptor heads -> aroma_models/ (+ manifest.json)")
