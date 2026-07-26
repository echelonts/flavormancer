"""build_profile_index.py — precompute the neighbor / substitute reference index.

The profile-based substitutes rank molecules by cosine similarity over their predicted taste +
aroma head SCORES. Computing that 170-head inference over the whole ~8.8k-molecule universe at app
startup is slow (~3 min), so we precompute it here once and cache to profile_index.npz. predict.py
loads it instantly (rebuilding the cheap Morgan fingerprints from SMILES on load).

Output profile_index.npz:
  smiles            (N,)  canonical SMILES, deduped by connectivity skeleton
  taste_documented  (N,)  comma-separated documented tastes ("" if none)
  aromas            (N,)  comma-separated confident aroma heads (score >= 0.5)
  profiles          (N,D) float32 head-score matrix: taste heads then aroma heads
  dims              (D,)  column labels ("taste:sweet", "aroma:citrus", ...)

Usage: python build_profile_index.py
"""
import numpy as np
import pandas as pd
import predict as P
from rdkit import Chem

SRC = "master_enrichment.parquet"
OUT = "profile_index.npz"


def main():
    m = pd.read_parquet(SRC)
    taste_heads, aroma_heads = P._profile_heads()
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
    X = np.vstack(feats)
    print(f"{len(smis)} unique structures — running {len(taste_heads)}+{len(aroma_heads)} heads...", flush=True)
    cols, aromas = [], [[] for _ in smis]
    for t in taste_heads:
        cols.append(P._CLASSIFIERS[t].predict_proba(X)[:, 1])
    for a in aroma_heads:
        col = P._AROMA_MODELS[a].predict_proba(X)[:, 1]
        cols.append(col)
        for i in range(len(smis)):
            if col[i] >= 0.5:
                aromas[i].append(a)
    profiles = np.column_stack(cols).astype("float32")
    dims = [f"taste:{t}" for t in taste_heads] + [f"aroma:{a}" for a in aroma_heads]
    np.savez_compressed(
        OUT,
        smiles=np.array(smis, dtype=object),
        taste_documented=np.array(td, dtype=object),
        aromas=np.array([",".join(a) for a in aromas], dtype=object),
        profiles=profiles,
        dims=np.array(dims, dtype=object),
    )
    print(f"wrote {OUT}: {profiles.shape[0]} molecules x {profiles.shape[1]} heads")


if __name__ == "__main__":
    main()
