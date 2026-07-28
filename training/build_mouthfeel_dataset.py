"""build_mouthfeel_dataset.py — presence/absence training table for the MOUTHFEEL heads.

Mouthfeel / chemesthesis is its own modality (separate from taste and aroma), so it gets its own
dataset -> train_mouthfeel.py -> mouthfeel_models/, mirroring the independent taste subsystem.

Positives are the curated trigeminal agents from mouthfeel_supplement.csv (warming / astringent /
tingling — public-domain flavor-chemistry facts). Negatives are a broad RANDOM background sampled
from the molecule universe (a molecule with no documented chemesthesis sensation is a negative), so
each head learns "this sensation vs. everything else." Output mouthfeel_train.parquet
(inchikey, smiles, <one 0/1 column per descriptor>).

Usage: python build_mouthfeel_dataset.py     # needs mouthfeel_supplement.csv (run its builder first)
"""
import csv
from pathlib import Path

import pandas as pd
from rdkit import Chem

SUP = "mouthfeel_supplement.csv"
UNIVERSE = "master_enrichment.parquet"
OUT = "mouthfeel_train.parquet"
N_BACKGROUND = 2500   # random negatives so each head is "sensation vs. everything else"


def main():
    if not Path(SUP).exists():
        raise SystemExit(f"{SUP} missing — run build_mouthfeel_supplement.py first")
    pos = {}           # skeleton -> {"smiles": canonical, "d": set(descriptors)}
    descriptors = set()
    with open(SUP) as f:
        for r in csv.DictReader(f):
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            skel = Chem.MolToInchiKey(m).split("-")[0]
            descriptors.add(r["flavor"])
            pos.setdefault(skel, {"smiles": Chem.MolToSmiles(m), "d": set()})["d"].add(r["flavor"])
    descriptors = sorted(descriptors)

    rows = [{"inchikey": skel, "smiles": v["smiles"],
             **{d: (1 if d in v["d"] else 0) for d in descriptors}}
            for skel, v in pos.items()]

    # broad background negatives from the universe (deterministic sample; skip any that are positives)
    uni = pd.read_parquet(UNIVERSE)
    uni = uni.sample(min(len(uni), N_BACKGROUND * 2), random_state=42)  # oversample, then filter positives
    added = 0
    seen = set(pos)
    for smi in uni["smiles"]:
        if added >= N_BACKGROUND:
            break
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        skel = Chem.MolToInchiKey(m).split("-")[0]
        if skel in seen:
            continue
        seen.add(skel)
        rows.append({"inchikey": skel, "smiles": Chem.MolToSmiles(m), **{d: 0 for d in descriptors}})
        added += 1

    out = pd.DataFrame(rows)
    out.to_parquet(OUT)
    counts = ", ".join(f"{d}={int(out[d].sum())}" for d in descriptors)
    print(f"{OUT}: {len(out)} molecules ({len(pos)} positives + {added} background) — {counts}")


if __name__ == "__main__":
    main()
