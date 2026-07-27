"""
train_mouthfeel.py — MOUTHFEEL / chemesthesis CLASSIFIERS on Morgan fingerprints.

The mouthfeel analogue of train_aroma.py / train_taste.py: one RandomForest per trigeminal
descriptor (warming / astringent / tingling) trained on mouthfeel_train.parquet
(build_mouthfeel_dataset.py). Same honest bar — >=10 positives AND 5-fold CV-AUROC >= 0.70 — so a
head only ships if the sensation is actually learnable from structure. Kept heads + their CV-AUROC
go to mouthfeel_models/ (its own dir, independent of aroma/taste), loaded and tagged 'mouthfeel'
by predict.py. Same modeling stack as the other modalities; positives are curated public-domain
structure->sensation facts (see build_mouthfeel_supplement.py), not intensity.

Usage: python train_mouthfeel.py
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
MIN_POS = 10       # enough positives for a 5-fold estimate; still must clear the AUROC bar below
MIN_AUROC = 0.70   # below this the sensation isn't learnable from structure -> don't ship it
OUT = Path("mouthfeel_models")
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


df = pd.read_parquet("mouthfeel_train.parquet")
descriptors = [c for c in df.columns if c not in ("inchikey", "smiles")]
feats, keep = [], []
for i, s in enumerate(df["smiles"]):
    f = fp(s)
    if f is not None:
        feats.append(f)
        keep.append(i)
X = np.array(feats)
df = df.iloc[keep].reset_index(drop=True)

print(f"training mouthfeel/chemesthesis classifiers on {len(df)} molecules "
      f"(min {MIN_POS} positives, keep CV-AUROC >= {MIN_AUROC}):")
manifest = {"fp_bits": FP_BITS, "fp_radius": FP_RADIUS, "descriptors": {}}
for d in sorted(descriptors, key=lambda c: -int(df[c].sum())):
    y = df[d].values.astype(int)
    npos = int(y.sum())
    if npos < MIN_POS:
        print(f"  {d:12s} n_pos={npos:4d}  -> skip (below {MIN_POS})")
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
print(f"\nkept {len(manifest['descriptors'])} mouthfeel heads -> mouthfeel_models/ (+ manifest.json)")
