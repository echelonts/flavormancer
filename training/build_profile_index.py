"""build_profile_index.py — precompute the neighbor / substitute reference index (PARALLEL).

The profile-based substitutes rank molecules by cosine similarity over their predicted taste +
aroma head SCORES. Running that inference over the whole ~8.8k-molecule universe is the one batch
job that used to be single-threaded (~3 min). Here we fan the heads across ALL cores as separate
PROCESSES — each worker loads only its shard of head models and scores them against the shared
feature matrix. joblib unpickling and predict_proba both run without fighting one GIL, so this
scales near-linearly (30 workers ≈ 30x on the inference) and finishes in seconds. Output is
identical to before; predict.py loads profile_index.npz instantly on startup.

Output profile_index.npz:
  smiles            (N,)  canonical SMILES, deduped by connectivity skeleton
  taste_documented  (N,)  comma-separated documented tastes ("" if none)
  aromas            (N,)  comma-separated confident aroma heads (score >= that head's calibrated threshold)
  profiles          (N,D) float32 head-score matrix: taste heads then aroma heads (sorted names)
  dims              (D,)  column labels ("taste:sweet", "aroma:citrus", ...)

Usage: python build_profile_index.py            # uses ~all cores
       FLAVORMANCER_INDEX_WORKERS=8 python build_profile_index.py
"""
import contextlib
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

# The parent process only needs featurization, so tell predict (imported lazily inside main) to
# skip loading models entirely; each worker loads its own shard of head models.
os.environ.setdefault("FLAVORMANCER_NO_MODELS", "1")

SRC = "master_enrichment.parquet"
OUT = "profile_index.npz"
TASTE_DIR = Path("taste_models")
AROMA_DIR = Path("aroma_models")
MOUTHFEEL_DIR = Path("mouthfeel_models")


def _score_shard(args):
    """Worker: load this shard's head models and score them against the memmapped feature matrix.
    Returns {head_name: float32 column of P(class=1) over all molecules}."""
    x_path, heads = args
    x = np.load(x_path, mmap_mode="r")
    out = {}
    for name, path in heads:
        clf = joblib.load(path)
        with contextlib.suppress(Exception):  # some wrapped estimators reject the n_jobs set; harmless
            clf.n_jobs = 1
        out[name] = clf.predict_proba(x)[:, 1].astype("float32")
    return out


def _heads():
    """(taste, aroma, mouthfeel names, ordered [(name, path), ...]) derived straight from the head
    files, in the SAME sorted order predict._profile_heads() uses — no models loaded here."""
    taste = sorted(p.stem[:-3] for p in TASTE_DIR.glob("*_rf.joblib") if p.stem != "sweet_intensity_rf")
    aroma = sorted(p.stem[:-4] for p in AROMA_DIR.glob("*_clf.joblib"))
    mouth = sorted(p.stem[:-4] for p in MOUTHFEEL_DIR.glob("*_clf.joblib")) if MOUTHFEEL_DIR.exists() else []
    # keys are DIM-qualified ("mouthfeel:cooling") because cooling/pungent exist as both an aroma
    # head and a mouthfeel head — a bare-name key would collide and overwrite one with the other.
    files = ([(f"taste:{n}", str(TASTE_DIR / f"{n}_rf.joblib")) for n in taste]
             + [(f"aroma:{n}", str(AROMA_DIR / f"{n}_clf.joblib")) for n in aroma]
             + [(f"mouthfeel:{n}", str(MOUTHFEEL_DIR / f"{n}_clf.joblib")) for n in mouth])
    return taste, aroma, mouth, files


def main():
    import predict as P  # NO_MODELS is set above, so this pulls in featurization without loading heads
    m = pd.read_parquet(SRC)
    smis, td, feats, seen = [], [], [], set()
    for _, r in m.iterrows():
        mol = Chem.MolFromSmiles(str(r["smiles"]))
        if mol is None:
            continue
        skel = Chem.MolToInchiKey(mol).split("-")[0]
        if skel in seen:
            continue
        seen.add(skel)
        smis.append(Chem.MolToSmiles(mol))
        td.append(str(r.get("taste_documented") or ""))
        feats.append(P._feat(mol)[0])
    x = np.vstack(feats).astype("float32")

    taste, aroma, mouth, head_files = _heads()
    # a batch job — saturate the box: one worker per core, but never more workers than there are
    # heads to score (extra processes would just sit idle). Scales 4-core -> 4, 32-core -> 32.
    n_cores = os.cpu_count() or 4
    workers = int(os.environ.get("FLAVORMANCER_INDEX_WORKERS") or max(2, min(len(head_files), n_cores)))
    print(f"{len(smis)} unique structures — {len(taste)}+{len(aroma)}+{len(mouth)} heads "
          f"across {workers} processes...", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        x_path = str(Path(tmp) / "X.npy")
        np.save(x_path, x)
        # round-robin the heads so each worker gets a mix (aroma heads dominate the count)
        shards = [head_files[i::workers] for i in range(workers)]
        scores = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(_score_shard, [(x_path, s) for s in shards if s]):
                scores.update(part)

    dims = ([f"taste:{t}" for t in taste] + [f"aroma:{a}" for a in aroma]
            + [f"mouthfeel:{h}" for h in mouth])
    profiles = np.column_stack([scores[k] for k in dims]).astype("float32")  # scores keyed by dim
    # Per-head calibrated thresholds, read straight from the manifest: this module runs under
    # NO_MODELS so predict's own _AROMA_META is empty, and defaulting to a flat 0.5 here would make
    # the precomputed chip list disagree with a live read for exactly the thin heads (#261).
    import json
    _mf = AROMA_DIR / "manifest.json"
    aroma_meta = json.loads(_mf.read_text()).get("descriptors", {}) if _mf.exists() else {}
    aromas = [[] for _ in smis]
    # Indicative heads (those that never reach 50% out-of-fold precision) are INCLUDED here on
    # purpose. Dropping them would silently delete 73 of 167 notes from chip search and palette
    # match — a real loss of reach to avoid a labelling problem. The honest fix is to mark them,
    # which the read does (`indicative` per descriptor), not to hide the molecules they find.
    for a in aroma:
        col = scores[f"aroma:{a}"]
        thr = P._head_threshold(aroma_meta, a)
        for i in range(len(smis)):
            if col[i] >= thr:
                aromas[i].append(a)
    tmp_out = "profile_index.building.npz"  # write then atomically replace so the live index is never half-written
    np.savez_compressed(
        tmp_out,
        smiles=np.array(smis, dtype=object),
        taste_documented=np.array(td, dtype=object),
        aromas=np.array([",".join(a) for a in aromas], dtype=object),
        profiles=profiles,
        dims=np.array(dims, dtype=object),
    )
    os.replace(tmp_out, OUT)
    print(f"wrote {OUT}: {profiles.shape[0]} molecules x {profiles.shape[1]} heads")


if __name__ == "__main__":
    main()
