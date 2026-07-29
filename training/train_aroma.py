"""
train_aroma.py — multi-label odor-descriptor CLASSIFIERS on Morgan fingerprints (PARALLEL).

Reads aroma_train.parquet (build_aroma_dataset.py: public-domain HSDB odor text keyword-
normalized to presence/absence descriptor labels). Trains one RandomForest classifier per
descriptor that has enough positives, reports HONEST 5-fold CV AUROC, and keeps only the
descriptors that clear a minimum AUROC. Saves the kept heads to aroma_models/ + a manifest.

PARALLELISM: each head trains in its OWN process with n_jobs=1, so the ~167 heads train
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
from chemfeatures import descriptors as _desc
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score

FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
MIN_POS = 10       # enough positives for a 5-fold estimate; still must clear the AUROC bar below
MIN_AUROC = 0.70   # below this the descriptor isn't learnable from structure -> don't ship it
OUT = Path("aroma_models")

# Per-head decision threshold bounds. A head with 13 positives against 2400 negatives is
# calibrated conservatively even with balanced class weights, so a flat 0.5 silently withholds
# real matches from the thin heads (see docs/METHODS.md). We fit each head its own threshold
# instead — but bounded, so a degenerate fit can't produce a head that fires on everything or
# nothing. Both directions are allowed: an over-eager head should be able to move UP.
THR_LO, THR_HI = 0.15, 0.85
MIN_PRECISION = 0.50  # a call labelled "confident" must be right more often than not


def _calibrate(y, oof):
    """Fit a head's decision threshold on OUT-OF-FOLD probabilities.

    Three properties make this a calibration and not a thumb on the scale:

    1. `oof` comes from cross_val_predict, so every probability was produced by a model that did
       NOT see that molecule. Tuning on in-sample scores would just relabel the training set.
    2. The threshold must clear a PRECISION FLOOR. Maximising F1 alone is not enough and we
       learned this the hard way: on a head with 12 positives in 2400 molecules, F1 peaks in a
       low-precision regime because recall climbs faster than precision falls. `blackberry` tuned
       to 0.18, where 96% of its calls were wrong. A high AUROC does not protect against this —
       AUROC is insensitive to class imbalance and precision is not.
    3. The result is bounded, reported, and honest about failure. Where NO threshold reaches the
       floor, the head is marked `confident_capable: false` rather than given a flattering cut-off:
       it keeps its score and its place in the profile, but the UI must not present it as a
       confident call. Nothing is deleted — see docs/METHODS.md.

    Returns (threshold, precision, recall, f1, confident_capable).
    """
    npos = int((y == 1).sum())
    best = None          # best F1 among thresholds clearing the precision floor
    fallback = None      # highest-precision threshold, for heads that never clear it
    for t in np.arange(THR_LO, THR_HI + 1e-9, 0.01):
        pred = oof >= t
        nfire = int(pred.sum())
        tp = int((pred & (y == 1)).sum())
        if not tp:
            continue
        precision, recall = tp / nfire, tp / npos
        f1 = 2 * precision * recall / (precision + recall)
        row = (float(t), precision, recall, f1)
        if precision >= MIN_PRECISION and (best is None or f1 > best[3]):
            best = row
        if fallback is None or precision > fallback[1]:
            fallback = row
    if best is not None:
        t, p, r, f1 = best
        return round(t, 2), round(p, 3), round(r, 3), round(f1, 3), True
    if fallback is not None:
        t, p, r, f1 = fallback
        return round(t, 2), round(p, 3), round(r, 3), round(f1, 3), False
    return 0.5, None, None, None, False


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
    disjoint files). Returns (name, auroc_or_None, n_pos, kept, threshold, f1)."""
    name, x_path, y = args
    y = np.asarray(y, dtype=int)
    npos = int(y.sum())
    if npos < MIN_POS:
        return (name, None, npos, False, None)
    x = np.load(x_path, mmap_mode="r")
    clf = RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=42, class_weight="balanced")
    try:
        auroc = float(cross_val_score(clf, x, y, cv=5, scoring="roc_auc", n_jobs=1).mean())
    except Exception:  # noqa: BLE001 — degenerate fold split etc.; treat as not learnable
        return (name, None, npos, False, None)
    kept = auroc >= MIN_AUROC
    cal = None
    if kept:
        # Second CV pass, for the decision threshold only. AUROC deliberately still comes from
        # cross_val_score above so the reported score stays comparable with every earlier run —
        # pooling out-of-fold probabilities would shift it slightly for reasons unrelated to the
        # model getting better or worse.
        try:
            oof = cross_val_predict(clf, x, y, cv=5, method="predict_proba", n_jobs=1)[:, 1]
            cal = _calibrate(y, oof)
        except Exception:  # noqa: BLE001 — fall back to the flat default, never to no head
            cal = (0.5, None, None, None, False)
        clf.fit(x, y)
        joblib.dump(clf, OUT / f"{name}_clf.joblib")
    return (name, round(auroc, 3), npos, kept, cal)


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

    for name, auroc, npos, kept, cal in sorted(results, key=lambda r: -r[2]):
        note = ""
        if kept:
            thr, prec, rec, f1, capable = cal
            manifest["descriptors"][name] = {"auroc": auroc, "n_pos": npos, "threshold": thr,
                                             "cv_precision": prec, "cv_recall": rec, "cv_f1": f1,
                                             "confident_capable": capable}
            note = f"  thr={thr:.2f} prec={prec if prec is not None else float('nan'):.2f}"
            if not capable:
                note += "  INDICATIVE (never reaches 50% precision)"
        flag = "kept" if kept else ("skip (<10)" if npos < MIN_POS else "drop (not learnable)")
        au = f"{auroc:.3f}" if auroc is not None else "  -  "
        print(f"  {name:14s} n_pos={npos:4d}  CV-AUROC={au}{note}  -> {flag}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ds = manifest["descriptors"]
    capable = [n for n, d in ds.items() if d.get("confident_capable")]
    thrs = sorted(d["threshold"] for d in ds.values())
    print(f"\nkept {len(ds)} descriptor heads -> aroma_models/ (+ manifest.json)")
    print(f"{len(capable)}/{len(ds)} reach {MIN_PRECISION:.0%} out-of-fold precision and can make "
          f"CONFIDENT calls; the other {len(ds) - len(capable)} are INDICATIVE — they keep their "
          f"score and their place in the profile, but must not be shown as confident.")
    if thrs:
        print(f"thresholds: min {thrs[0]:.2f}, median {thrs[len(thrs) // 2]:.2f}, max {thrs[-1]:.2f}")


if __name__ == "__main__":
    main()
