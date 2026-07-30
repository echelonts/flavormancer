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
import json
import os
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
    # --- Heads that were MEMORIZING (#256): each fired on exactly its own training molecules and
    # nothing else. The fix is structural DIVERSITY within the class, not more of one scaffold —
    # the same approach that moved `tingling` off zero in #247. All well-established
    # character-impact chemistry for these materials, public-domain flavour/essential-oil facts.
    "pine":     ["alpha-pinene", "beta-pinene", "camphene", "delta-3-carene", "terpinolene",
                 "myrcene", "bornyl acetate", "isobornyl acetate", "longifolene", "borneol",
                 "verbenone", "alpha-terpineol"],
    "eucalyptus": ["1,8-cineole", "alpha-terpineol", "terpinen-4-ol", "p-cymene", "aromadendrene",
                 "globulol", "alpha-phellandrene", "gamma-terpinene", "trans-pinocarveol"],
    "fennel":   ["anethole", "fenchone", "estragole", "alpha-phellandrene", "anisaldehyde",
                 "fenchyl alcohol", "limonene", "beta-phellandrene", "camphene"],
    "celery":   ["3-n-butylphthalide", "sedanolide", "sedanenolide", "neocnidilide",
                 "beta-selinene", "3-n-butyl-4,5-dihydrophthalide", "senkyunolide A", "ligustilide"],
    "rosemary": ["1,8-cineole", "camphor", "borneol", "verbenone", "alpha-pinene", "bornyl acetate",
                 "camphene", "isoborneol", "rosmarinic acid"],
    "turmeric": ["ar-turmerone", "alpha-turmerone", "beta-turmerone", "ar-curcumene", "zingiberene",
                 "beta-sesquiphellandrene", "curlone", "curcumene"],
    "allspice": ["eugenol", "methyl eugenol", "beta-caryophyllene", "1,8-cineole", "chavicol",
                 "eugenyl acetate", "alpha-phellandrene", "terpinen-4-ol"],
    "frankincense": ["incensole", "incensole acetate", "alpha-pinene", "octyl acetate", "verbenone",
                 "serratol", "cembrene", "alpha-thujene", "octanol"],
    "narcissus": ["indole", "p-cresol", "benzyl acetate", "cinnamyl alcohol", "benzyl benzoate",
                 "methyl benzoate", "alpha-terpineol", "eugenol", "benzyl alcohol"],
    "freesia":  ["linalool", "alpha-ionone", "beta-ionone", "geraniol", "nerol", "benzaldehyde",
                 "dihydro-beta-ionone", "linalyl acetate", "citronellol"],
    "elemi":    ["elemol", "elemicin", "limonene", "alpha-phellandrene", "beta-elemene",
                 "dill apiole", "elemene", "sabinene"],
    "costus":   ["costunolide", "dehydrocostus lactone", "costol", "costic acid",
                 "dihydrocostunolide", "alpha-costene", "aplotaxene"],
    "green":    ["cis-3-hexenal", "trans-2-hexenal", "cis-3-hexen-1-ol", "hexanal"],
    # --- INDICATIVE heads (#262): right less than half the time when they fire, because 11-30
    # positives cannot draw a boundary against 2,400 negatives. These are the ones with deep,
    # uncontroversial public chemistry, so the fix is straightforward: more structurally diverse
    # positives. Probability calibration was tested first and does NOT work here — isotonic
    # regression needs data too, so it lifted `fruity` (101 positives) and made `marine` and
    # `gardenia` worse. Data is the lever.
    "citrus":   ["limonene", "citral", "geranial", "neral", "citronellal", "linalool",
                 "gamma-terpinene", "beta-pinene", "sabinene", "myrcene", "terpinolene",
                 "octanal", "decanal", "nonanal", "dodecanal", "nootkatone", "valencene",
                 "citronellol", "neryl acetate", "geranyl acetate"],
    "almond":   ["benzaldehyde", "benzyl alcohol", "benzyl acetate", "furfural", "5-methylfurfural",
                 "heliotropin", "p-tolualdehyde", "phenylacetaldehyde", "benzyl benzoate",
                 "2-phenylethanol", "salicylaldehyde", "cinnamaldehyde"],
    "bready":   ["2-acetyl-1-pyrroline", "6-acetyl-2,3,4,5-tetrahydropyridine", "maltol", "furfural",
                 "methional", "2,3-butanedione", "2-acetylpyrazine", "5-hydroxymethylfurfural",
                 "2-acetylfuran", "3-methylbutanal", "2,5-dimethylpyrazine", "furfuryl alcohol"],
    "earthy":   ["geosmin", "2-methylisoborneol", "2-isopropyl-3-methoxypyrazine",
                 "2-isobutyl-3-methoxypyrazine", "2-ethyl-3-methoxypyrazine", "patchoulol",
                 "alpha-humulene", "1-octen-3-ol", "2-methoxy-3-methylpyrazine", "borneol"],
    "cumin":    ["cuminaldehyde", "p-cymene", "gamma-terpinene", "beta-pinene", "safranal",
                 "perillaldehyde", "cuminyl alcohol", "myrcene", "alpha-terpinene"],
    "oregano":  ["carvacrol", "thymol", "p-cymene", "gamma-terpinene", "linalool", "borneol",
                 "alpha-terpinene", "myrcene", "terpinen-4-ol", "carvacrol methyl ether"],
    "thyme":    ["thymol", "carvacrol", "p-cymene", "gamma-terpinene", "linalool", "borneol",
                 "camphene", "alpha-thujene", "terpinen-4-ol", "geraniol"],
    "dill":     ["carvone", "dillapiole", "alpha-phellandrene", "limonene", "myristicin",
                 "dihydrocarvone", "myrcene", "p-cymene", "anethofuran"],
    "bergamot": ["linalyl acetate", "linalool", "limonene", "gamma-terpinene", "nerol",
                 "beta-pinene", "myrcene", "neryl acetate", "geranial", "sabinene"],
    "violet":   ["alpha-ionone", "beta-ionone", "methyl ionone", "alpha-irone", "beta-irone",
                 "dihydro-beta-ionone", "alpha-isomethylionone", "2,6-nonadienal",
                 "dihydro-alpha-ionone"],
    "soapy":    ["decanal", "undecanal", "dodecanal", "2-methylundecanal", "1-decanol",
                 "1-dodecanol", "lauric aldehyde", "nonanal", "1-undecanol", "methyl laurate"],
    "powdery":  ["heliotropin", "vanillin", "coumarin", "alpha-ionone", "musk ketone",
                 "ethylvanillin", "anisyl alcohol", "benzyl salicylate", "4-methylacetophenone"],
    "melon":    ["melonal", "cis-3-hexenyl acetate", "cis-6-nonenal", "2,6-nonadienal", "cis-3-nonen-1-ol", "benzyl acetate", "nonanal", "cis-6-nonen-1-ol"],
    "mango":    ["3-carene", "gamma-decalactone", "alpha-terpinene", "gamma-octalactone", "terpinolene", "cis-ocimene"],
    "guava":    ["ethyl butyrate", "cis-3-hexenol", "ethyl hexanoate", "3-sulfanylhexyl acetate", "hexanal", "beta-caryophyllene", "limonene", "ethyl acetate", "1-hexanol"],
    "cassis":   ["4-methoxy-2-methyl-2-butanethiol", "beta-caryophyllene", "linalool", "terpinen-4-ol", "delta-3-carene", "gamma-terpinene", "alpha-pinene"],
    "blackberry":["2-heptanone", "methyl anthranilate", "furaneol", "linalool", "alpha-ionone"],
    "blueberry":["linalool", "trans-2-hexenal", "geraniol", "eucalyptol", "alpha-terpineol"],
    "passionfruit":["hexyl butyrate", "3-sulfanylhexyl acetate", "3-sulfanylhexan-1-ol", "beta-ionone", "linalool", "hexyl hexanoate"],
    "lime":     ["limonene", "gamma-terpinene", "citral", "beta-pinene", "terpinolene",
                 "geranial", "neral", "alpha-terpineol", "myrcene", "sabinene"],
    "mandarin": ["limonene", "gamma-terpinene", "methyl N-methylanthranilate", "myrcene",
                 "alpha-pinene", "linalool", "octanal", "decanal", "thymol", "terpinolene"],
    "ginger":   ["zingiberene", "6-gingerol", "6-shogaol", "zingerone", "beta-sesquiphellandrene",
                 "ar-curcumene", "alpha-farnesene", "camphene", "geranial", "beta-bisabolene"],
    "blackpepper": ["piperine", "beta-caryophyllene", "limonene", "sabinene", "alpha-pinene",
                 "beta-pinene", "delta-3-carene", "myrcene", "alpha-phellandrene", "linalool"],
    "nutmeg":   ["myristicin", "sabinene", "alpha-pinene", "beta-pinene", "elemicin",
                 "safrole", "terpinen-4-ol", "limonene", "myrcene", "eugenol"],
    "cardamom": ["1,8-cineole", "alpha-terpinyl acetate", "linalool", "limonene", "sabinene",
                 "alpha-terpineol", "myrcene", "geraniol", "linalyl acetate", "nerolidol"],
    "sage":     ["camphor", "1,8-cineole", "alpha-thujone", "beta-thujone", "borneol", "camphene",
                 "alpha-pinene", "bornyl acetate", "viridiflorol", "humulene"],
    "juniper":  ["alpha-pinene", "myrcene", "sabinene", "limonene", "terpinen-4-ol",
                 "beta-pinene", "gamma-cadinene", "alpha-terpinene", "germacrene D", "borneol"],
    "tea":      ["linalool", "geraniol", "cis-jasmone", "methyl jasmonate", "beta-ionone",
                 "indole", "benzyl alcohol", "2-phenylethanol", "hexanal", "trans-2-hexenal"],
    "tobacco":  ["megastigmatrienone", "solanone", "beta-damascenone", "neophytadiene",
                 "beta-ionone", "furfural", "phenylacetaldehyde", "damascone", "cyclotene"],
    "marine":   ["2,6-nonadienal", "cis-3-hexenol", "dimethyl sulfide", "1-octen-3-ol",
                 "cis-4-heptenal", "geosmin", "trans-2-nonenal", "benzothiazole", "octanal"],
    "elderflower": ["linalool", "cis-rose oxide", "nerol oxide", "hotrienol", "alpha-terpineol",
                 "beta-damascenone", "hexanal", "phenylacetaldehyde", "nonanal"],
    "lilac":    ["lilac aldehyde", "lilac alcohol", "linalool", "alpha-terpineol", "indole",
                 "benzyl alcohol", "cinnamyl alcohol", "hotrienol", "nerol oxide"],
    "mimosa":   ["anisaldehyde", "heliotropin", "benzyl alcohol", "2-phenylethanol",
                 "methyl anisate", "palmitic acid", "linalool", "benzaldehyde", "anisyl alcohol"],
    "violetleaf": ["2,6-nonadienal", "trans-2-hexenal", "cis-3-hexenol", "2-nonenal",
                 "cis-3-hexenyl acetate", "1-octen-3-ol", "hexanal", "trans-2-nonenal"],
    "tomato":   ["2-isobutylthiazole", "cis-3-hexenal", "trans-2-hexenal", "hexanal",
                 "beta-ionone", "6-methyl-5-hepten-2-one", "1-penten-3-one", "geranylacetone",
                 "methional", "cis-3-hexenol"],
    "medicinal": ["phenol", "guaiacol", "thymol", "eugenol", "methyl salicylate",
                 "carvacrol", "1,8-cineole", "camphor", "menthol", "m-cresol"],
    "rancid":   ["butyric acid", "isovaleric acid", "hexanoic acid", "octanoic acid",
                 "trans-2-nonenal", "heptanoic acid", "valeric acid", "propionic acid",
                 "2,4-decadienal", "4-methyloctanoic acid"],
    # fig and magnolia sat just under the 0.70 AUROC bar and fell off the roster when the corpus
    # shifted around them. Rather than let two heads go, give each its documented character
    # chemistry — sesquiterpenes and the creamy lactone for fig, the ocimene/anthranilate floral
    # set for magnolia — and keep the generic fruit esters OUT, which is what cost blackberry.
    "fig":      ["germacrene D", "beta-caryophyllene", "gamma-decalactone", "methyl salicylate",
                 "trans-2-hexenal", "alpha-copaene", "alpha-ylangene", "delta-cadinene",
                 "gamma-dodecalactone", "alpha-humulene"],
    "magnolia": ["methyl anthranilate", "trans-beta-ocimene", "citral", "geraniol",
                 "alpha-terpineol", "methyl benzoate", "isoeugenol", "nerolidol",
                 "benzyl alcohol", "cis-beta-ocimene"],
    # --- #257: the broad heads. sweet/ethereal/pungent now clear the 50% precision floor but sit
    # EXACTLY on it with high thresholds (0.75 / 0.64 / 0.66), which means they only stay confident
    # by being very reluctant to fire. Food-authorised character molecules push them clear on the
    # food side rather than removing the industrial odorants that also (truthfully) smell this way.
    "sweet":    ["vanillin", "ethylvanillin", "maltol", "ethyl maltol", "furaneol", "homofuraneol",
                 "heliotropin", "anisaldehyde", "gamma-decalactone", "delta-decalactone",
                 "gamma-undecalactone", "4-hydroxy-2,5-dimethyl-3(2H)-furanone", "veratraldehyde",
                 "acetovanillone", "anisyl alcohol", "methyl anthranilate"],
    # ethereal = the light, volatile, solvent-like top note. Food-authorised acetals, formates and
    # short esters, NOT the chlorinated solvents that share the descriptor honestly (see #257).
    "ethereal": ["acetaldehyde", "1,1-diethoxyethane", "ethyl formate", "methyl formate",
                 "methyl acetate", "acetaldehyde dimethyl acetal", "2-methyl-1,3-dioxolane",
                 "propanal", "isobutyraldehyde", "diethyl carbonate"],
    # fatty sat at 0.21 precision on 27 positives — one of the worst. Its chemistry is unusually
    # crisp for a broad head: mid-chain aldehydes, free fatty acids and the lactones.
    "fatty":    ["nonanal", "decanal", "2,4-decadienal", "trans-2-nonenal", "trans-2-decenal",
                 "2-undecenal", "octanoic acid", "decanoic acid", "hexanoic acid", "lauric acid",
                 "myristic acid", "gamma-nonalactone", "methyl octanoate", "ethyl decanoate",
                 "1-octanol", "heptanal"],
    "alcoholic": ["ethanol", "1-propanol", "isobutanol", "isoamyl alcohol", "1-butanol",
                 "2-methylbutanol", "1-hexanol", "phenethyl alcohol", "1-pentanol"],
    # pungent (an odor/chemesthesis head, also tagged mouthfeel): sharp biting Piper long/black-pepper
    # amides on top of the corpus's documented pungent molecules (piperine, isothiocyanates, etc.)
    "pungent":  ["piperlongumine", "piperlonguminine", "pipernonaline", "sarmentine",
                 "dehydropipernonaline", "guineensine",
                 # #257: food-authorised sharp-smelling volatiles on the food side — short acids,
                 # isothiocyanates and alliaceous sulfur. Appended to the existing Piper amides
                 # rather than declared as a second "pungent" key, which would silently shadow
                 # them (last key wins in a dict literal) and delete six molecules.
                 "acetic acid", "formic acid", "propionic acid", "butyric acid",
                 "allyl isothiocyanate", "methyl isothiocyanate", "phenethyl isothiocyanate",
                 "diallyl disulfide", "diallyl sulfide", "isovaleric acid", "methyl mercaptan"],
}


# Persistent resolution cache — makes the build DETERMINISTIC and network-independent. Without it,
# every build re-queries PubChem for each un-hinted name, and any rate-limit/timeout silently drops
# that molecule, churning the marginal (n~10) descriptor heads build-to-build. Cache once, reuse
# forever; delete resolve_cache.json to force a re-fetch.
_CACHE_PATH = "resolve_cache.json"
try:
    with open(_CACHE_PATH, encoding="utf-8") as _cf:
        _RESOLVE_CACHE = json.load(_cf)
except (OSError, ValueError):
    _RESOLVE_CACHE = {}


def resolve(name):
    """name -> canonical SMILES via PubChem (authoritative), or None. Cached persistently so the
    build is deterministic and does not depend on live network reachability."""
    if Chem.MolFromSmiles(name):
        return Chem.MolToSmiles(Chem.MolFromSmiles(name))
    if name in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[name]
    smi = None
    try:
        import pubchempy as pcp
        hits = pcp.get_compounds(name, "name")
        if hits and hits[0].canonical_smiles:
            m = Chem.MolFromSmiles(hits[0].canonical_smiles)
            smi = Chem.MolToSmiles(m) if m else None
    except Exception:  # noqa: BLE001 — offline / not found; leave uncached so a later online build can fill it
        return None
    if smi:  # only cache successful resolutions; misses stay retryable
        _RESOLVE_CACHE[name] = smi
        try:
            with open(_CACHE_PATH, "w", encoding="utf-8") as _cf:
                json.dump(_RESOLVE_CACHE, _cf, indent=0, sort_keys=True)
        except OSError:
            pass
    return smi


def _items():
    """(descriptor, name, smiles_hint) for every curated + open-gov-sourced association. The sourced
    additions (aroma_additions.csv) are kept as a data file because the set is large; each carries an
    optional authoritative SMILES so tricky names resolve without a PubChem round-trip. Every sourced
    molecule is open-government food-authorised (see food_safe_supplement.csv for its citation)."""
    out = [(d, n, "") for d, names in CURATED.items() for n in names]
    if os.path.exists("aroma_additions.csv"):
        with open("aroma_additions.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d, m = (r.get("descriptor") or "").strip(), (r.get("molecule") or "").strip()
                if d and m:
                    out.append((d, m, (r.get("smiles") or "").strip()))
    return out


def main():
    rows, seen, misses = [], set(), []
    for descriptor, name, smi_hint in _items():
        smi = smi_hint or resolve(name)
        if smi:                                          # canonicalise so dedup + downstream keying match
            m = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(m) if m else ""
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
