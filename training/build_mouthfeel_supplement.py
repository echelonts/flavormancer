"""build_mouthfeel_supplement.py — curated PUBLIC-DOMAIN mouthfeel / chemesthesis facts.

Mouthfeel (the oral tactile percept: warming, cooling, tingling, pungency, astringency) is a
DIFFERENT sensory modality from odor — it's trigeminal/somatosensory, not olfactory — so it gets
its OWN curated list and its own CSV, kept separate from the aroma supplement (build_aroma_supplement.py).

Same provenance rules as the aroma supplement: only well-established, public-domain
structure -> sensation associations taught throughout open-access flavor/sensory chemistry
(capsaicin = burning heat; menthol = cooling; sanshool = tingling; tannins = astringency). These are
measured facts, not copyrightable (Feist), and NOT copied from any restricted compilation.

IMPORTANT — food safety is a SEPARATE, per-molecule flag, not implied here. A head predicts a
SENSATION; a training molecule being food-present (capsaicin in chili, tannins in tea/wine) is a
sensory fact, not a food-use clearance. The corpus leans on food-present naturals + food-authorised
agents (nonivamide FL/GRAS, gallic/tannic acid 21 CFR), and the per-molecule food-safe flag +
the standing IP/food review gate still govern any commercial use.

Usage: python build_mouthfeel_supplement.py   # -> mouthfeel_supplement.csv
"""
import csv
import sys

from build_aroma_supplement import (
    resolve,  # reuse the cached, authoritative PubChem resolver
)

# descriptor -> well-known character-impact mouthfeel molecule NAMES (public-domain facts).
# cooling and pungent already ship as heads (trained from the aroma/odor corpus, tagged mouthfeel);
# these three add the rest of the trigeminal picture.
CURATED = {
    # cooling: TRPM8 physiological coolants — menthol family + food-authorised WS-agents (some, like
    # WS-23, have essentially NO odour, so this is a genuine SENSATION head, distinct from the aroma
    # 'cooling' odour descriptor — exactly as taste:sweet differs from aroma:sweet).
    "cooling":   ["menthol", "menthone", "isomenthone", "neomenthol", "isopulegol", "menthyl lactate",
                  "menthyl acetate", "WS-3", "WS-23", "WS-5", "WS-12", "cyclohexanecarboxamide, N-ethyl-",
                  "frescolat MGA", "coolact P", "icilin", "eucalyptol", "camphor", "borneol"],
    # pungent (sensation): TRPV1 / mustard-oil irritants — capsaicinoids, isothiocyanates, alliaceous
    # sulfur, peppery amides. Distinct from the aroma 'pungent' odour head (sharp-smelling volatiles).
    "pungent":   ["capsaicin", "dihydrocapsaicin", "nonivamide", "piperine", "allyl isothiocyanate",
                  "phenethyl isothiocyanate", "benzyl isothiocyanate", "diallyl disulfide", "allicin",
                  "6-gingerol", "6-shogaol", "cinnamaldehyde"],
    # warming: capsaicinoid & warm-spice heat agents (chili, black pepper, ginger, cinnamon, clove)
    "warming":   ["capsaicin", "dihydrocapsaicin", "nordihydrocapsaicin", "nonivamide", "piperine",
                  "6-gingerol", "6-shogaol", "zingerone", "cinnamaldehyde", "eugenol",
                  "capsiate", "dihydrocapsiate"],
    # astringent: tannins & polyphenols (tea, wine, fruit skins) — the pucker / dry mouthfeel
    "astringent":["gallic acid", "tannic acid", "catechin", "epicatechin", "epigallocatechin gallate",
                  "epicatechin gallate", "ellagic acid", "quercetin", "chlorogenic acid",
                  "ferulic acid", "caffeic acid", "1,2,3,4,6-pentagalloylglucose"],
    # tingling: STRICT paresthesia agents only — Sichuan-pepper sanshools, jambu/Acmella isobutyl-
    # amides, pellitorine. A genuinely narrow class: only ~7 unique public structures resolve, BELOW
    # the >=10 bar, so this stays a documented near-miss (a vocab descriptor, no shipped head yet) —
    # NOT padded with the Piper pungent amides, which read pungent and now feed that head instead.
    "tingling":  ["hydroxy-alpha-sanshool", "alpha-sanshool", "beta-sanshool", "gamma-sanshool",
                  "hydroxy-beta-sanshool", "hydroxy-gamma-sanshool", "spilanthol", "pellitorine",
                  ("(2E,4E)-dodeca-2,4-dienoic acid N-isobutylamide", "CCCCCCC/C=C/C=C/C(=O)NCC(C)C"),
                  # more genuine tingle amides — tested against PubChem for an authoritative structure
                  "bungeanool", "isobungeanool", "tetrahydrobungeanool", "neoherculin",
                  "ZP-amide A", "ZP-amide C", "hydroxy-epsilon-sanshool", "sanshoamide",
                  "dihydro-alpha-sanshool",
                  # STRUCTURAL DIVERSITY, deliberately beyond Zanthoxylum: a head trained only on
                  # sanshools memorizes that one scaffold instead of learning "long-chain unsaturated
                  # N-alkylamide". These are paresthesia amides from other genera — Echinacea,
                  # Anacyclus (pellitory), Heliopsis, Piper — with different chain lengths and
                  # unsaturation patterns, so the class itself becomes learnable.
                  "dodeca-2E,4E,8Z,10E-tetraenoic acid isobutylamide",
                  "dodeca-2E,4E,8Z,10Z-tetraenoic acid isobutylamide",
                  "undeca-2E,4Z-diene-8,10-diynoic acid isobutylamide",
                  "dodeca-2E,4E-dienoic acid isobutylamide",
                  "deca-2E,4E-dienoic acid isobutylamide",
                  "anacyclin", "scabrin", "achilleamide", "retrofractamide A",
                  "guineensine", "piperlonguminine", "trichostachine",
                  "N-isobutyl-2E,4E-decadienamide", "echinacein", "affinin"],
}
# NOTE: cooling & pungent ALSO exist as AROMA odour-descriptor heads (menthol smells cool; mustard
# smells pungent). Those stay — this file trains the SENSATION versions. A molecule can be both.
# (obsolete note retained below for context)
# `pungent` and `cooling` are AROMA heads (trained on the odor corpus) that we also TAG
# mouthfeel — they're genuinely both, like `sweet` is taste + aroma. Their extra positives (e.g. the
# Piper pungent amides) live in the AROMA supplement, not here. This file holds the mouthfeel-ONLY
# descriptors (warming / astringent / tingling) that get their own train_mouthfeel.py -> mouthfeel_models/.


def main():
    from rdkit import Chem
    rows, seen, misses = [], set(), []
    for descriptor, entries in CURATED.items():
        for entry in entries:
            name, hint = entry if isinstance(entry, tuple) else (entry, "")
            smi = hint if (hint and Chem.MolFromSmiles(hint)) else resolve(name)
            if not smi:
                misses.append(f"{descriptor}:{name}")
                continue
            key = (descriptor, Chem.MolToSmiles(Chem.MolFromSmiles(smi)))  # canonical, so synonyms dedup
            if key in seen:  # same molecule under a synonym — count once
                continue
            seen.add(key)
            rows.append({"flavor": descriptor, "molecule": name, "smiles": smi, "category": "mouthfeel"})
    with open("mouthfeel_supplement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["flavor", "molecule", "smiles", "category"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    per = Counter(r["flavor"] for r in rows)
    print(f"mouthfeel_supplement.csv: {len(rows)} associations across {len(per)} descriptors")
    print("  per descriptor:", dict(per))
    if misses:
        print(f"  unresolved ({len(misses)}):", misses, file=sys.stderr)


if __name__ == "__main__":
    main()
