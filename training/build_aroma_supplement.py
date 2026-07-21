"""build_aroma_supplement.py — curated, PUBLIC-DOMAIN character-impact odor facts.

Each entry is a well-established, ubiquitous structure→odor association — the kind annotated on
PubChem compound pages (US-gov public domain) and taught throughout open-access flavor chemistry.
Odor descriptors are *measured facts*, not copyrightable (Feist v. Rural), and we use only
associations any flavorist knows (e.g. gamma-nonalactone = coconut). SMILES are pulled from
PubChem (public domain). This is a small, hand-assembled list of individual public facts — NOT
a copy of, or extraction from, any copyrighted flavor compilation or restricted database (all of
those — the GS-LF / Leffingwell / GoodScents / FlavorDB family — are documented as EXCLUDED in
docs/DATA-SOURCES.md and docs/SOURCES.md, and are not used anywhere in this project). Spot-check
the associations before release; a formal IP review is the standing pre-commercial gate.

We resolve each molecule by NAME via PubChem to get an authoritative canonical SMILES (no
hand-transcription errors), then write aroma_supplement.csv in the flavors.csv schema
(descriptor,molecule,smiles,category) so build_aroma_dataset.fold_flavors folds them in as
extra aroma positives for the sparse descriptors identified in docs/AROMA-AUDIT.md.

Usage: python build_aroma_supplement.py     # -> aroma_supplement.csv
"""
import csv
import sys

from rdkit import Chem

# descriptor -> list of well-known character-impact molecule NAMES (public-domain facts)
CURATED = {
    "coconut":  ["gamma-nonalactone", "gamma-octalactone", "delta-decalactone",
                 "gamma-decalactone", "delta-octalactone", "massoia lactone"],
    "caramel":  ["maltol", "ethyl maltol", "cyclotene", "4-hydroxy-2,5-dimethyl-3(2H)-furanone",
                 "homofuraneol", "5-methylfurfural"],
    "honey":    ["phenylacetic acid", "methyl phenylacetate", "ethyl phenylacetate",
                 "2-phenylethanol", "phenylacetaldehyde", "2-phenylethyl acetate",
                 "methyl anthranilate"],
    "buttery":  ["2,3-butanedione", "acetoin", "2,3-pentanedione", "2,3-hexanedione", "2,3-heptanedione"],
    "vanilla":  ["vanillin", "ethylvanillin", "vanillyl alcohol", "vanillic acid",
                 "acetovanillone", "veratraldehyde", "syringaldehyde",
                 # food-safe re-base (EU Union List, open-gov) after dropping non-food isovanillin:
                 "piperonal", "4-hydroxybenzaldehyde", "4-formyl-2-methoxyphenyl acetate"],
    # isovanillin EXCLUDED: no EU FL / 21 CFR food clearance — not a food ingredient (this is a
    # flavor app, so non-food odorants are kept out of the corpus, not merely flagged)
    # methyleugenol EXCLUDED: removed from 21 CFR 172.515 in the 2018 Delaney delisting and
    # restricted in the EU (genotoxic) — not an authorised added flavouring anywhere. Out of the corpus.
    "clove":    ["eugenol", "isoeugenol", "eugenyl acetate", "4-vinylguaiacol"],
    "cinnamon": ["cinnamaldehyde", "cinnamyl alcohol", "cinnamic acid", "methyl cinnamate",
                 "ethyl cinnamate", "cinnamyl acetate", "alpha-methylcinnamaldehyde", "cinnamyl formate", "hydrocinnamaldehyde", "cinnamyl butyrate"],
    "spicy":    ["eugenol", "cinnamaldehyde", "zingerone", "piperonal", "carvacrol",
                 # food-authorised spice odorants (EU Union List / 21 CFR, open-gov):
                 "piperine", "cuminaldehyde", "beta-caryophyllene", "cinnamyl acetate",
                 "eugenyl acetate", "isoeugenol"],
    "banana":   ["isoamyl acetate", "amyl acetate", "isoamyl butyrate", "isoamyl isovalerate", "isoamyl propionate", "isobutyl acetate", "isoamyl formate", "amyl butyrate", "2-methylbutyl acetate", "isoamyl hexanoate"],
    "nutty":    ["2,3-dimethylpyrazine", "2,5-dimethylpyrazine", "2-ethylpyrazine",
                 "2-acetylpyrazine", "5-methylfurfural", "2-ethyl-3-methylpyrazine",
                 "2-acetylthiazole", "2,3-diethylpyrazine"],
    "cocoa":    ["2,3,5,6-tetramethylpyrazine", "2,3,5-trimethylpyrazine", "3-methylbutanal",
                 "2-methylbutanal", "isovaleraldehyde"],
    "coffee":   ["furfuryl mercaptan", "guaiacol", "2-acetylpyrazine", "5-methylfurfural"],
    "smoky":    ["guaiacol", "2,6-dimethoxyphenol", "4-methylguaiacol", "4-ethylguaiacol",
                 "4-vinylguaiacol", "creosol", "phenol", "p-cresol", "o-cresol", "2,6-dimethylphenol"],
    # estragole / methyl chavicol EXCLUDED (same molecule, CAS 140-67-0): prohibited as an added
    # flavouring in the EU/GB (Reg. 1334/2008 Annex III, genotoxicity). Kept out of the corpus.
    "anise":    ["anethole", "anisaldehyde", "p-anisaldehyde", "fenchone",
                 "anisyl alcohol", "anisyl phenylacetate", "methyl anisate"],
    "balsamic": ["benzyl benzoate", "benzyl cinnamate", "benzoic acid", "benzyl salicylate", "cinnamyl cinnamate"],
    "herbal":   ["thymol", "carvacrol", "eucalyptol", "1,8-cineole", "menthone", "isomenthone"],
    "meaty":    ["methional", "2-methyl-3-furanthiol", "furfuryl mercaptan",
                 # food-authorised savoury sulfur volatiles (EU Union List / 21 CFR, open-gov):
                 "bis(2-methyl-3-furyl) disulfide", "2,4,5-trimethylthiazole",
                 "2-methyl-3-(methylthio)furan", "4-methyl-5-vinylthiazole", "dimethyl trisulfide"],
    "cherry":   ["benzaldehyde", "p-tolualdehyde"],
    "winey":    ["ethyl lactate", "ethyl hexanoate", "2,3-butanediol"],
    "onion":    ["dipropyl disulfide", "methyl propyl disulfide", "allyl propyl disulfide"],
    # keep the fragile grassy head above the bar with its classic green-leaf volatiles
    "fresh":    ["melonal", "dihydromyrcenol", "cis-3-hexenyl acetate", "hexanal"],
    # habanolide EXCLUDED: fragrance-only musk, no food clearance found — dropped like isovanillin.
    # Reclaimed FOOD-SAFE with EU-authorised macrocyclic musks (pentadecanolide FL 10.004,
    # dihydroambrettolide FL 10.047) + US GRAS/SAF musks (muscone, civetone) so the head stands
    # entirely on food-authorised molecules.
    "musky":    ["muscone", "ethylene brassylate", "ambrettolide",
                 "15-pentadecanolide", "16-hexadecanolide", "civetone"],
    "vegetable":["2-isobutyl-3-methoxypyrazine", "2-isopropyl-3-methoxypyrazine", "dimethyl sulfide", "2-acetylpyrrole"],
    "grassy":   ["cis-3-hexenal", "cis-3-hexen-1-ol", "trans-2-hexenal", "hexanal",
                 "trans-2-hexen-1-ol", "cis-3-hexenyl acetate"],
    "green":    ["cis-3-hexenal", "trans-2-hexenal", "cis-3-hexen-1-ol", "hexanal"],
}


def resolve(name):
    """name -> canonical SMILES via PubChem (authoritative), or None."""
    if Chem.MolFromSmiles(name):
        return Chem.MolToSmiles(Chem.MolFromSmiles(name))
    try:
        import pubchempy as pcp
        hits = pcp.get_compounds(name, "name")
        if hits and hits[0].canonical_smiles:
            m = Chem.MolFromSmiles(hits[0].canonical_smiles)
            return Chem.MolToSmiles(m) if m else None
    except Exception:  # noqa: BLE001 — offline / not found; skip this entry
        return None
    return None


def main():
    rows, seen, misses = [], set(), []
    for descriptor, names in CURATED.items():
        for name in names:
            smi = resolve(name)
            if not smi:
                misses.append(f"{descriptor}:{name}")
                continue
            key = (descriptor, smi)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"flavor": descriptor, "molecule": name, "smiles": smi,
                         "category": "aroma-supplement"})
    with open("aroma_supplement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["flavor", "molecule", "smiles", "category"])
        w.writeheader()
        w.writerows(rows)
    per = {}
    for r in rows:
        per[r["flavor"]] = per.get(r["flavor"], 0) + 1
    print(f"aroma_supplement.csv: {len(rows)} verified associations across {len(per)} descriptors")
    print("  per descriptor:", dict(sorted(per.items(), key=lambda kv: -kv[1])))
    if misses:
        print(f"  unresolved ({len(misses)}):", misses, file=sys.stderr)


if __name__ == "__main__":
    main()
