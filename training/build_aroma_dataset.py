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

# controlled odor-descriptor vocabulary: descriptor -> keyword cues matched with word boundaries.
# Broadened over the first pass to recover more of the ~2,258 HSDB odor texts (many phrased with
# a synonym the narrow list missed), plus a handful of new descriptors common in the corpus.
VOCAB = {
    "fruity": ["fruity", "fruit", "berry", "estery", "ester", "jammy"], "sweet": ["sweet", "sweetish"],
    "floral": ["floral", "flower", "flowery", "blossom"],
    "citrus": ["citrus", "citrusy", "lemon", "orange", "lime", "grapefruit", "bergamot"],
    "green": ["green", "leafy"],
    "minty": ["mint", "minty", "menthol", "peppermint", "spearmint", "cooling"],
    "herbal": ["herb", "herbal", "herbaceous"], "woody": ["wood", "woody", "cedar", "sandalwood"],
    "spicy": ["spicy", "spice", "peppery", "pepper"], "rose": ["rose", "rosy"],
    "almond": ["almond", "marzipan", "benzaldehyde"], "vanilla": ["vanilla", "vanillin"],
    "caramel": ["caramel", "caramellic"], "nutty": ["nut", "nutty"],
    "buttery": ["butter", "buttery", "diacetyl"], "fatty": ["fatty", "oily", "tallow"],
    "rancid": ["rancid"],
    "sulfurous": ["sulfur", "sulphur", "sulfurous", "sulfury", "rotten egg", "hydrogen sulfide", "eggy", "skunk"],
    "garlic": ["garlic", "alliaceous"], "onion": ["onion"], "fishy": ["fish", "fishy", "aminy"],
    "earthy": ["earth", "earthy", "musty", "moldy", "mushroom", "humus"],
    "camphor": ["camphor", "camphoraceous"], "pine": ["pine", "piney", "turpentine", "resinous"],
    "balsamic": ["balsam", "balsamic"],
    "medicinal": ["medicinal", "phenol", "phenolic", "carbolic", "antiseptic", "iodine"],
    "smoky": ["smoke", "smoky", "smoked", "smokey"],
    "pungent": ["pungent", "sharp", "acrid", "irritating", "choking"],
    "ethereal": ["ether", "ethereal", "solvent", "solventy", "chloroform"],
    "winey": ["wine", "winey", "vinous", "brandy", "cognac"], "honey": ["honey"],
    "coconut": ["coconut"], "banana": ["banana"], "apple": ["apple"], "cherry": ["cherry"],
    "clove": ["clove", "eugenol"], "cinnamon": ["cinnamon", "cinnamic"],
    "anise": ["anise", "licorice", "aniseed"], "coffee": ["coffee"],
    "cocoa": ["cocoa", "chocolate"], "meaty": ["meat", "meaty", "brothy", "savory"],
    "cheesy": ["cheese", "cheesy"], "creamy": ["cream", "creamy", "milky", "dairy"],
    "waxy": ["wax", "waxy"], "fresh": ["fresh"],
    "grassy": ["grass", "grassy", "hay", "coumarin"],
    "burnt": ["burnt", "roasted", "toasted", "scorched", "charred"],
    "tarry": ["tar", "tarry", "creosote"], "ammoniacal": ["ammonia", "ammoniacal", "amine"],
    "fecal": ["fecal", "feces", "faecal", "manure"],
    # new descriptors (each clears the training MIN_POS on this corpus)
    "musky": ["musk", "musky"], "alcoholic": ["alcohol", "alcoholic", "ethanol"],
    "putrid": ["putrid", "rotten", "rotting", "foul", "decayed", "decaying", "stench"],
    "soapy": ["soap", "soapy"], "vegetable": ["vegetable", "cabbage", "celery", "potato"],
    "bready": ["bread", "bready", "yeasty", "yeast", "doughy"],
    "petroleum": ["petroleum", "kerosene", "gasoline", "petrol", "naphtha"],
}
PATS = {d: re.compile(r"\b(" + "|".join(k) + r")\b", re.I) for d, k in VOCAB.items()}


def fold_flavors(rows, path="flavors.csv"):
    """Fold the curated flavor->character-molecule list in as extra AROMA labels: a molecule that
    IS the banana / cherry / clove note is a documented positive for that descriptor. Extends the
    weak-labeled HSDB set with hand-verified character-impact facts (only for flavors that are
    already descriptors in VOCAB). Returns (rows, n_labels_added, n_new_molecules)."""
    p = Path(path)
    if not p.exists():
        return rows, 0, 0
    fl = pd.read_csv(p)
    by_key = {r["inchikey_skel"]: r for r in rows}  # rows are dicts; index by skeleton
    added, new = 0, 0
    for flavor, smi in zip(fl["flavor"].astype(str).str.lower(), fl["smiles"]):
        if flavor not in VOCAB:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        skel = Chem.MolToInchiKey(mol).split("-")[0]
        row = by_key.get(skel)
        if row is None:
            row = {"inchikey": Chem.MolToInchiKey(mol), "smiles": Chem.MolToSmiles(mol),
                   "inchikey_skel": skel, **{d: 0 for d in VOCAB}}
            by_key[skel] = row
            rows.append(row)
            new += 1
        if not row.get(flavor):
            row[flavor] = 1
            added += 1
    return rows, added, new


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
        mol = Chem.MolFromSmiles(str(r["smiles"]))
        if mol is None:
            continue
        tags = tag(r["odor"])
        if not tags:  # no descriptor keyword -> uninformative ("characteristic odor"); skip to
            continue  # limit false-negative noise (absence-as-negative only among tagged mols)
        row = {"inchikey": r["inchikey"], "smiles": r["smiles"],
               "inchikey_skel": Chem.MolToInchiKey(mol).split("-")[0]}
        for d in VOCAB:
            row[d] = int(d in tags)
        rows.append(row)
    n_odor = len(rows)
    rows, n_lbl, n_new = fold_flavors(rows)  # curated character-impact molecules as extra positives
    # public-domain aroma supplement (build_aroma_supplement.py) — extra positives for the sparse
    # descriptors flagged in docs/AROMA-AUDIT.md, sourced from open flavor-chemistry facts
    rows, n_sup, n_sup_new = fold_flavors(rows, path="aroma_supplement.csv")
    n_lbl += n_sup
    n_new += n_sup_new
    out = pd.DataFrame(rows).drop(columns=["inchikey_skel"])
    out.to_parquet("aroma_train.parquet")
    counts = {d: int(out[d].sum()) for d in VOCAB}
    print(f"aroma_train.parquet: {len(out)} molecules with >=1 descriptor "
          f"({n_odor} from HSDB odor text of {len(df)}; +{n_lbl} curated flavor labels "
          f"across +{n_new} molecules), {len(VOCAB)} descriptors")
    print("descriptor positives (sorted):")
    for d, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {d}")
