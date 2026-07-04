"""
build_aroma_dataset.py — turn documented odor free-text into a multi-label descriptor dataset.

Reads odor_notes.parquet (public-domain HSDB/Haz-Map odor descriptions + SMILES, built by
build_odor_notes.py --pubchem-all) and normalizes each molecule's free-text odor into a
CONTROLLED descriptor vocabulary by word-boundary keyword matching. Output aroma_train.parquet
= (inchikey, smiles, <one 0/1 column per descriptor>) — a multi-label presence/absence dataset
for train_aroma.py.

Honest scope: PRESENCE/ABSENCE learned from free text, not intensity — HSDB text has no scored
descriptors. It's weak labeling (a missing keyword is treated as a negative), so noisier than
expert panel data (GS-LF), but real and commercial-clean (public domain). (The earlier
keller_2016 CC-BY approach was dropped — negative-R² on naive-subject ratings; this HSDB set is
larger and yes/no rather than noisy 0-100 scores.)

Usage: python build_aroma_dataset.py            # odor_notes.parquet -> aroma_train.parquet
"""
import re
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# controlled odor-descriptor vocabulary: descriptor -> keyword cues matched with word boundaries
VOCAB = {
    "fruity": ["fruity", "fruit"], "sweet": ["sweet"], "floral": ["floral", "flower", "flowery"],
    "citrus": ["citrus", "lemon", "orange", "lime", "grapefruit"], "green": ["green"],
    "minty": ["mint", "minty", "menthol", "peppermint", "spearmint"], "herbal": ["herb", "herbal"],
    "woody": ["wood", "woody"], "spicy": ["spicy", "spice"], "rose": ["rose", "rosy"],
    "almond": ["almond"], "vanilla": ["vanilla"], "caramel": ["caramel", "caramellic"],
    "nutty": ["nut", "nutty"], "buttery": ["butter", "buttery"], "fatty": ["fatty"],
    "rancid": ["rancid"], "sulfurous": ["sulfur", "sulphur", "sulfurous", "sulfury"],
    "garlic": ["garlic"], "onion": ["onion"], "fishy": ["fish", "fishy"],
    "earthy": ["earth", "earthy", "musty", "moldy"], "camphor": ["camphor", "camphoraceous"],
    "pine": ["pine", "piney"], "balsamic": ["balsam", "balsamic"],
    "medicinal": ["medicinal", "phenol", "phenolic"], "smoky": ["smoke", "smoky", "smoked"],
    "pungent": ["pungent", "sharp", "acrid"], "ethereal": ["ether", "ethereal", "solvent"],
    "winey": ["wine", "winey", "vinous"], "honey": ["honey"], "coconut": ["coconut"],
    "banana": ["banana"], "apple": ["apple"], "cherry": ["cherry"], "clove": ["clove"],
    "cinnamon": ["cinnamon", "cinnamic"], "anise": ["anise", "licorice", "aniseed"],
    "coffee": ["coffee"], "cocoa": ["cocoa", "chocolate"], "meaty": ["meat", "meaty"],
    "cheesy": ["cheese", "cheesy"], "creamy": ["cream", "creamy"], "waxy": ["wax", "waxy"],
    "fresh": ["fresh"], "grassy": ["grass", "grassy", "hay"],
    "burnt": ["burnt", "roasted", "toasted"], "tarry": ["tar", "tarry", "creosote"],
    "ammoniacal": ["ammonia", "ammoniacal"], "fecal": ["fecal", "feces", "faecal", "manure"],
}
PATS = {d: re.compile(r"\b(" + "|".join(k) + r")\b", re.I) for d, k in VOCAB.items()}


def tag(text):
    """Set of descriptors whose keywords appear in the odor text."""
    t = (text or "").replace("\n", " ")
    return {d for d, p in PATS.items() if p.search(t)}


if __name__ == "__main__":
    src = "odor_notes.parquet"
    if not Path(src).exists():
        print(f"{src} not found — run build_odor_notes.py --pubchem-all first")
        sys.exit(1)
    df = pd.read_parquet(src)
    df = df[df["odor"].notna() & df["smiles"].notna()].copy()
    rows = []
    for _, r in df.iterrows():
        if Chem.MolFromSmiles(str(r["smiles"])) is None:
            continue
        tags = tag(r["odor"])
        if not tags:  # no descriptor keyword -> uninformative ("characteristic odor"); skip to
            continue  # limit false-negative noise (absence-as-negative only among tagged mols)
        row = {"inchikey": r["inchikey"], "smiles": r["smiles"]}
        for d in VOCAB:
            row[d] = int(d in tags)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet("aroma_train.parquet")
    counts = {d: int(out[d].sum()) for d in VOCAB}
    print(f"aroma_train.parquet: {len(out)} molecules with >=1 descriptor "
          f"(of {len(df)} with odor text), {len(VOCAB)} descriptors")
    print("descriptor positives (sorted):")
    for d, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {d}")
