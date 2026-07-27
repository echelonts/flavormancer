"""
train_aroma.py — multi-label odor-descriptor CLASSIFIERS on Morgan fingerprints (PARALLEL).

Reads aroma_train.parquet (build_aroma_dataset.py: public-domain HSDB odor text keyword-
normalized to presence/absence descriptor labels). Trains one RandomForest classifier per
descriptor that has enough positives, reports HONEST 5-fold CV AUROC, and keeps only the
descriptors that clear a minimum AUROC. Saves the kept heads to aroma_models/ + a manifest.

PARALLELISM: each head trains in its OWN process with n_jobs=1, so the ~164 heads train
concurrently across the box's cores instead of one-at-a-time (the same process-per-unit pattern
that made build_profile_index fast). Feature matrix is memmapped so workers share it. A head
either clears the bar and dumps its model, or is dropped — workers write disjoint files, no locks.

This is the same modeling stack as taste (Morgan FP + RandomForest), but PRESENCE/ABSENCE
per descriptor (the free text carries no intensity). Commercial-clean, public-domain aroma read.

Usage: python train_aroma.py           # uses ~all cores
"""
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
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
MIN_AUROC = 0.70   # below this the descriptor isn't learnable from structure -> don't ship it
OUT = Path("aroma_models")


def fp(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    bv = _MORGAN.GetFingerprint(m)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return np.concatenate([arr, _desc(m)])  # fingerprint + physicochemical block (chemfeatures.py)


def _train_head(args):
    """Worker: CV + fit ONE head at n_jobs=1. Dumps the model if it clears the bar (workers write
    disjoint files). Returns (name, auroc_or_None, n_pos, kept)."""
    name, x_path, y = args
    y = np.asarray(y, dtype=int)
    npos = int(y.sum())
    if npos < MIN_POS:
        return (name, None, npos, False)
    x = np.load(x_path, mmap_mode="r")
    clf = RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=42, class_weight="balanced")
    try:
        auroc = float(cross_val_score(clf, x, y, cv=5, scoring="roc_auc", n_jobs=1).mean())
    except Exception:  # noqa: BLE001 — degenerate fold split etc.; treat as not learnable
        return (name, None, npos, False)
    kept = auroc >= MIN_AUROC
    if kept:
        clf.fit(x, y)
        joblib.dump(clf, OUT / f"{name}_clf.joblib")
    return (name, round(auroc, 3), npos, kept)


def main():
    OUT.mkdir(exist_ok=True)
    for s in OUT.glob("*_clf.joblib"):
        s.unlink()
    df = pd.read_parquet("aroma_train.parquet")
    descriptors = [c for c in df.columns if c not in ("inchikey", "smiles")]
    feats, keep = [], []
    for i, s in enumerate(df["smiles"]):
        f = fp(s)
        if f is not None:
            feats.append(f)
            keep.append(i)
    x = np.array(feats)
    df = df.iloc[keep].reset_index(drop=True)

    workers = int(os.environ.get("FLAVORMANCER_TRAIN_WORKERS")
                  or max(2, min(len(descriptors), os.cpu_count() or 4)))
    print(f"training {len(descriptors)} odor heads on {len(df)} molecules across {workers} processes "
          f"(min {MIN_POS} positives, keep CV-AUROC >= {MIN_AUROC}):", flush=True)
    manifest = {"fp_bits": FP_BITS, "fp_radius": FP_RADIUS, "descriptors": {}}
    with tempfile.TemporaryDirectory() as tmp:
        x_path = str(Path(tmp) / "X.npy")
        np.save(x_path, x)
        args = [(d, x_path, df[d].values.astype(int)) for d in descriptors]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_train_head, args))

    for name, auroc, npos, kept in sorted(results, key=lambda r: -r[2]):
        if kept:
            manifest["descriptors"][name] = {"auroc": auroc, "n_pos": npos}
        flag = "kept" if kept else ("skip (<10)" if npos < MIN_POS else "drop (not learnable)")
        au = f"{auroc:.3f}" if auroc is not None else "  -  "
        print(f"  {name:14s} n_pos={npos:4d}  CV-AUROC={au}  -> {flag}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nkept {len(manifest['descriptors'])} descriptor heads -> aroma_models/ (+ manifest.json)")


if __name__ == "__main__":
    main()
