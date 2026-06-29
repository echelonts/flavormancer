"""
train_tox.py — caution-only toxicity-assay heads from Tox21 (public domain).

Tox21 (NIH/NCATS · EPA · FDA · NTP) screened ~8k compounds against 12 in-vitro assays
(nuclear-receptor + stress-response, incl. genotoxic-stress SR-p53 / SR-ATAD5 and AhR).
This trains one RandomForest per assay on Morgan fingerprints — **INDICATIVE, caution-only**
flags ("active in an in-vitro tox assay → review"), **NEVER a toxicity determination**.
Reports honest 5-fold CV AUROC; saves to tox_models/.

Data: tox21.csv — the MoleculeNet mirror of the NCATS public-domain Tox21 Challenge set.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
OUT = Path("tox_models")
OUT.mkdir(exist_ok=True)
for s in OUT.glob("*_rf.joblib"):
    s.unlink()


def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    bv = _MORGAN.GetFingerprint(m)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


df = pd.read_csv("tox21.csv")
feats, idx = [], []
for i, s in enumerate(df["smiles"]):
    f = fp(s)
    if f is not None:
        feats.append(f)
        idx.append(i)
X = np.array(feats)
df = df.iloc[idx].reset_index(drop=True)

print(f"training caution-only tox heads on {len(df)} molecules:")
kept = 0
for t in TASKS:
    y = df[t].values
    mask = ~np.isnan(y)
    Xd, yd = X[mask], y[mask].astype(int)
    if yd.sum() < 30:
        print(f"  {t:14s} too few positives — skip")
        continue
    clf_args = dict(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced")
    auc = cross_val_score(RandomForestClassifier(**clf_args), Xd, yd, cv=5, scoring="roc_auc").mean()
    joblib.dump(RandomForestClassifier(**clf_args).fit(Xd, yd), OUT / f"{t}_rf.joblib")
    kept += 1
    print(f"  {t:14s} n={int(mask.sum()):5d} pos={int(yd.sum()):4d}  CV-AUROC={auc:.3f}")
print(f"\nkept {kept}/{len(TASKS)} tox heads -> tox_models/  (caution-only, Tox21 public domain)")
