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
    "citrus": ["citrus", "citrusy", "lemon", "lime", "grapefruit", "bergamot"],
    # fruit sub-notes split out of the broad fruity/citrus buckets, for finer resolution
    "orange": ["orange", "orangey"], "peach": ["peach", "peachy"], "pear": ["pear"],
    "tropical": ["tropical", "pineapple", "mango", "passionfruit", "passion fruit", "papaya", "guava"],
    "grape": ["grape", "grapey", "foxy", "labrusca"], "melon": ["melon", "melony", "honeydew", "cantaloupe"],
    "apricot": ["apricot"],
    # acidic / vinegar odour — the smell of short-chain carboxylic acids (a coherent, food-relevant class)
    "acidic": ["acetic", "vinegar", "vinegary", "acidic", "sour milk", "acid odor", "acid odour"],
    # wintergreen / teaberry / birch — methyl salicylate & the salicylate esters (root-beer-adjacent)
    "wintergreen": ["wintergreen", "teaberry", "gaultheria", "birch", "methyl salicylate"],
    # floral sub-notes split out for finer resolution (distinct scaffolds: ionones / jasmonoids / linalyl)
    "violet": ["violet", "ionone", "orris", "orrisroot"], "jasmine": ["jasmine", "jasmin", "jasmone", "jasmonate"],
    "lavender": ["lavender", "lavandin"], "muguet": ["muguet", "lily of the valley", "hyacinth"],
    # other distinct food descriptors
    "ginger": ["ginger", "gingery"], "berry": ["berry", "strawberry", "raspberry", "blackberry", "blueberry", "currant"],
    "malty": ["malt", "malty"], "marine": ["marine", "oceanic", "seaweed", "algae", "oyster"],
    "mushroom": ["mushroom", "fungal"], "hay": ["hay", "new-mown", "new mown"],
    # further distinct food descriptors (final sourcing pass)
    "cardamom": ["cardamom", "cardamon"], "plum": ["plum", "prune", "damson"],
    "tonka": ["tonka", "coumarinic", "coumarin-like", "hay-sweet"], "tea": ["tea", "black tea", "green tea"],
    "neroli": ["neroli", "orange blossom", "orange-blossom", "petitgrain"],
    # further distinct sub-notes (final niche pass)
    "cassis": ["cassis", "blackcurrant", "black currant", "currant bud", "buchu"],
    "elderflower": ["elderflower", "elder flower"], "fennel": ["fennel", "fenchone"],
    "gardenia": ["gardenia"], "ylang": ["ylang", "ylang-ylang"],
    "magnolia": ["magnolia"],
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
    "earthy": ["earth", "earthy", "mushroom", "humus", "soil", "loam", "geosmin"],
    # musty split out of earthy — a stale/damp/mildew percept distinct from clean soil-earthy
    "musty": ["musty", "moldy", "mouldy", "mildew", "stale", "cellar", "damp", "fusty"],
    "camphor": ["camphor", "camphoraceous"], "pine": ["pine", "piney", "turpentine", "resinous"],
    "balsamic": ["balsam", "balsamic"],
    "medicinal": ["medicinal", "antiseptic", "iodine", "hospital", "disinfectant"],
    # phenolic split out of medicinal — the phenol/carbolic character on its own
    "phenolic": ["phenolic", "phenol", "carbolic", "cresylic", "cresol"],
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
    # new descriptors (each clears the training MIN_POS on this corpus)
    "musky": ["musk", "musky"], "alcoholic": ["alcohol", "alcoholic", "ethanol"],
    "putrid": ["putrid", "rotten", "rotting", "foul", "decayed", "decaying", "stench"],
    "soapy": ["soap", "soapy"], "vegetable": ["vegetable", "cabbage", "celery", "potato"],
    "bready": ["bread", "bready", "yeasty", "yeast", "doughy"],
    "petroleum": ["petroleum", "kerosene", "gasoline", "petrol", "naphtha"],
    # the aroma parallel to the taste model's "tasteless" — molecules documented to have no smell
    # (water, salts, most sugars, heavy/involatile solids). ~800 positives in the HSDB odor text.
    "odorless": ["odorless", "odourless", "odor-free", "odour-free", "no odor", "no odour",
                 "without odor", "without odour", "practically odorless", "practically odourless"],
    # aroma-only (fragrance) descriptors — positives come mainly from the curated non-food
    # fragrance supplement (flagged non-food); keyword cues harvest bonus HSDB-text matches.
    "aldehydic": ["aldehydic", "aldehyde-like"],
    "animalic": ["animalic", "animal-like", "fecal", "faecal", "civet", "castoreum", "indolic"],
    "amber": ["ambergris", "ambery", "amber-like", "amber odor", "amber odour"],
    "leathery": ["leather", "leathery", "suede"],
    "powdery": ["powdery", "powder-like"],
    "resinous": ["resinous", "resiny", "resin-like", "resin odor", "resin odour"],
    "terpenic": ["terpenic", "terpene-like", "terpeny", "turpentine"],
    "ozonic": ["ozone", "ozonic", "aquatic", "watery", "sea breeze", "sea-breeze"],
    "hyacinth": ["hyacinth", "narcissus"],
    "cooling": ["cooling", "cool-fresh", "coolant", "refreshing cool"],
    "metallic": ["metallic", "tinny", "blood-like", "metal-like"],
    "lactonic": ["lactonic", "lactone-like"],
    "celery": ["celery", "lovage", "phthalide"],
    "maple": ["maple", "fenugreek", "curry-like"],
    "popcorn": ["popcorn", "roasted cereal", "cracker", "roasty-nutty"],
    "lilac": ["lilac", "syringa"],
    "osmanthus": ["osmanthus"],
    "geranium": ["geranium", "pelargonium", "rose-geranium"],
    "eucalyptus": ["eucalyptus", "eucalyptol", "cineole"],
    "saffron": ["saffron", "safranal"],
    "chamomile": ["chamomile", "camomile"],
    "tobacco": ["tobacco", "cigarette", "cigar"],
    "carnation": ["carnation", "clove-pink", "dianthus"],
    "mimosa": ["mimosa", "acacia"],
    "narcissus": ["narcissus", "daffodil", "jonquil"],
    "corky": ["corky", "cork taint", "chloroanisole", "musty cork", "mouldy cork"],
    "violet_leaf": ["violet leaf", "violet-leaf", "leafy violet"],
    "fir": ["fir", "pine needle", "pine-needle", "conifer", "spruce"],
    "oakmoss": ["oakmoss", "oak moss", "mossy"],
    "vetiver": ["vetiver", "vetivert"],
    "patchouli": ["patchouli", "patchouly"],
    "sandalwood": ["sandalwood", "sandal"],
    "cedarwood": ["cedarwood", "cedar", "cedrus"],
    "galbanum": ["galbanum"],
    "cognac": ["cognac", "brandy"],
    "immortelle": ["immortelle", "helichrysum", "everlasting"],
    "grapefruit": ["grapefruit", "pomelo"],
    "bergamot": ["bergamot"],
    "mandarin": ["mandarin", "tangerine"],
    "lime": ["lime"],
    "yuzu": ["yuzu"],
    "honeysuckle": ["honeysuckle"],
    "freesia": ["freesia"],
    "cyclamen": ["cyclamen"],
    "linden": ["linden", "lime blossom", "tilia"],
    "fig": ["fig"],
    "quince": ["quince"],
    "lychee": ["lychee", "litchi"],
    "coriander": ["coriander", "cilantro"],
    "cumin": ["cumin", "cuminic"],
    "passionfruit": ["passion fruit", "passionfruit", "passion-fruit"],
    "rosemary": ["rosemary"],
    "black_pepper": ["black pepper", "peppercorn", "peppery"],
    "nutmeg": ["nutmeg", "mace"],
    "sage": ["sage", "salvia"],
    "thyme": ["thyme"],
    "oregano": ["oregano", "marjoram"],
    "mustard": ["mustard", "horseradish", "wasabi", "isothiocyanate"],
    "pineapple": ["pineapple", "ananas"],
    "strawberry": ["strawberry"],
    "raspberry": ["raspberry"],
    "cucumber": ["cucumber"],
    "myrrh": ["myrrh"],
    "frankincense": ["frankincense", "olibanum"],
    "blackberry": ["blackberry", "bramble"],
    "blueberry": ["blueberry", "bilberry"],
    "mango": ["mango"],
    "guava": ["guava"],
    "watermelon": ["watermelon"],
    "tomato": ["tomato"],
    "juniper": ["juniper", "gin-like"],
    "hazelnut": ["hazelnut", "filbert"],
    "allspice": ["allspice", "pimento"],
    "dill": ["dill"],
    "turmeric": ["turmeric", "curcuma"],
    "davana": ["davana"],
    "costus": ["costus"],
    "tagetes": ["tagetes", "marigold"],
    "truffle": ["truffle"],
    "clary_sage": ["clary sage", "clary"],
    "elemi": ["elemi"],
    "labdanum": ["labdanum", "cistus", "rockrose"],
    "wormwood": ["wormwood", "absinthe", "artemisia"],
    "styrax": ["styrax", "storax"],
    "opoponax": ["opoponax", "opopanax"],
    "champaca": ["champaca", "champak"],
    "boronia": ["boronia"],
    "angelica": ["angelica"],
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
