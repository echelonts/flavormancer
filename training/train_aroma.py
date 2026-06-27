"""
train_aroma.py — odor-descriptor regressors (one per descriptor) on Morgan fingerprints.

Reads aroma_master.parquet (keller_2016, CC-BY); trains a RandomForestRegressor per
descriptor predicting its 0-100 panel rating; reports HONEST 5-fold CV R2 (small,
noisy panel-mean data — expect modest, descriptor-dependent scores). Saves the ones
that clear a minimum CV-R2 to aroma_models/.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
MIN_R2 = 0.10  # below this a descriptor is too noisy to ship an honest head
OUT = Path("aroma_models")
OUT.mkdir(exist_ok=True)
for s in OUT.glob("*_reg.joblib"):
    s.unlink()

DESCRIPTORS = ["ACID", "AMMONIA/URINOUS", "BAKERY", "BURNT", "CHEMICAL", "COLD",
               "DECAYED", "EDIBLE", "FISH", "FLOWER", "FRUIT", "GARLIC", "GRASS",
               "MUSKY", "SOUR", "SPICES", "SWEATY", "SWEET", "WARM", "WOOD"]


def fp(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    bv = _MORGAN.GetFingerprint(m)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


df = pd.read_parquet("aroma_master.parquet")
feats, keep = [], []
for i, s in enumerate(df["smiles"]):
    f = fp(s)
    if f is not None:
        feats.append(f)
        keep.append(i)
X = np.array(feats)
df = df.iloc[keep].reset_index(drop=True)

print(f"training odor-descriptor regressors on {len(df)} molecules:")
kept = 0
for d in DESCRIPTORS:
    y = df[d].values
    mask = ~np.isnan(y)
    Xd, yd = X[mask], y[mask]
    r2 = cross_val_score(RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42),
                         Xd, yd, cv=5, scoring="r2").mean()
    flag = "kept" if r2 >= MIN_R2 else "drop (too noisy)"
    if r2 >= MIN_R2:
        reg = RandomForestRegressor(n_estimators=400, n_jobs=-1, random_state=42).fit(Xd, yd)
        joblib.dump(reg, OUT / f"{d.replace('/', '_')}_reg.joblib")
        kept += 1
    print(f"  {d:16s} n={int(mask.sum()):3d}  CV-R2={r2:+.2f}  -> {flag}")
print(f"\nkept {kept}/{len(DESCRIPTORS)} descriptor heads (CV-R2 >= {MIN_R2}) -> aroma_models/")
