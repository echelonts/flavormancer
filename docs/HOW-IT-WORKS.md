# How Flavormancer works

A plain-English tour of the methods behind every feature — what the machine is actually doing
when you type a molecule and get a flavor read. The one rule throughout: **every value is tagged
by how it was derived** (computed / trained / rule / estimate / lookup / qualitative), so nothing
ever reads as more certain than its source. Nothing here needs the cloud — it all runs on a box
you own.

---

## 0. The core idea

A molecule's **structure** determines how it tastes and smells. Flavormancer learns that
mapping — *structure → flavor* — from open chemistry data, so it can answer for a **novel or
unmeasured** compound that's in no database, not just for ones someone already tabulated. That's
the difference between **prediction** and **lookup**, and it's the whole point.

---

## 1. How a molecule becomes numbers

Everything starts by turning a structure into something math can chew on:

- **SMILES** — a molecule written as text (`O=Cc1ccc(O)c(OC)c1` is vanillin). You can type a name
  (`vanillin`) and we resolve it to SMILES via PubChem; the models themselves only ever see the
  structure.
- **Morgan fingerprint** (a.k.a. ECFP) — a 2,048-bit on/off vector where each bit marks the
  presence of a small local atom-neighborhood. Two molecules with similar substructures light up
  similar bits. This is the workhorse representation. *(RDKit computes it.)*
- **Physicochemical descriptors** — a dozen global properties: molecular weight, logP
  (oil-vs-water), TPSA (polarity), H-bond donors/acceptors, rotatable bonds, ring counts, fraction
  sp³. Fingerprints capture *local* shape; these capture the *global* properties that also drive
  flavor (a big polar sugar reads sweet; a small greasy ester reads fruity). *(`chemfeatures.py`.)*

The taste and aroma models learn from **fingerprint + descriptors together**. Similarity search
and the map deliberately use the **bare fingerprint** (the standard chemical-similarity measures
need the raw bits).

---

## 2. Taste — six trained heads + two rules

Six `RandomForest` classifiers (`train_taste.py`), each answering one yes/no question with a
probability, scored by honest held-out **CV-AUROC** (how well it separates the class on data it
never trained on):

| head | how | CV-AUROC |
|---|---|---|
| sweet / bitter / umami | trained on merged open taste databases (ChemTastesDB) | ~0.95 / 0.95 / 0.99 |
| sour | trained head **+** an acidic-group chemistry **rule** as a cross-check | 0.91 |
| salty | trained head **+** a cation-aware inorganic-salt **rule** | 0.93 |
| tasteless | a tasted-vs-tasteless head (documented tasteless molecules as positives) | 0.89 |

Why the rules for sour and salty? Sourness is really a **solution/pH** property and saltiness an
**ionic** one — not purely molecular shape — so we keep a transparent structural rule (does it
have an acid group? is it a simple alkali salt?) *alongside* the model, and show both. The salty
rule is careful: it fires on `NaCl`/`KCl` but **not** on MSG or sodium benzoate, where the organic
anion owns the taste.

A separate **regressor** estimates **sweetness intensity** (relative to sucrose) from the same
features (trained on SweetenersDB).

---

## 3. Aroma — 42 descriptor heads from public odor text

Odor is the hard, licensed part of this field — most rich odor datasets are NonCommercial. Our
clean route:

1. **`build_odor_notes.py`** pulls PubChem's **"Odor"** annotations — which are **public-domain**
   (HSDB / Haz-Map, US-government) for ~2,250 compounds — free-text descriptions like *"sweet,
   floral, slightly minty."*
2. **`build_aroma_dataset.py`** normalizes that free text into a **controlled descriptor
   vocabulary** by whole-word keyword matching (so *"minty," "peppermint," "menthol"* all map to
   `minty`), producing a presence/absence label per descriptor. We also fold in the curated
   character-impact facts from `flavors.csv`.
3. **`train_aroma.py`** trains one RandomForest per descriptor (on fingerprint + descriptors) and
   **keeps only the heads that clear CV-AUROC ≥ 0.70** — an honest bar. **42 heads** survive
   (citrus, floral, minty, almond, fatty, petroleum, earthy, medicinal, sulfurous, camphor,
   fruity, fishy, garlic, ethereal, ammoniacal, pungent, pine, rose, rancid, alcoholic,
   woody, green, grassy, putrid), 0.71–0.98. The lower-population heads (≈10–20 documented
   positives) are the noisiest — each ships with its honest CV-AUROC shown in the UI.

This is **presence/absence** ("which notes apply"), not **intensity** ("how strong") — the free
text carries no scored ratings, and no public-domain intensity data exists. Intensity is the one
piece that comes with a customer's panel data or a licensed set (PMP 2001). Where PubChem has a
cited odor + detection threshold for a molecule, we show that too (a **lookup**, not a guess).

---

## 4. Substitution search — "find me a molecule that behaves like this"

Rank every labeled molecule by **Tanimoto similarity** (overlap of fingerprint bits) to the query
and return the closest — for reformulation and cost-down. Self is excluded by InChIKey so the top
hit is a genuine analogue, not the molecule itself. *(`predict.substitute`.)*

---

## 5. Flavor Studio — from a flavor you want to the molecules that make it

A flavor *is* a set of notes, so it's **one picker** over everyday **flavors** (banana, saffron,
pumpkin…) and model **notes** (citrus, floral…). Pick any mix; each candidate molecule scores by
**how many of your picks it carries** (a flavor's character molecule, or a molecule the aroma/taste
models tag with that note), filtered to **food-safe (GRAS)** if you want, with **drop-in
substitutes** for each. The **natural-language box** ("a food-safe cherry flavoring with fruity
notes") is a lightweight offline parser that maps your words onto those same picks.

---

## 6. Flavor-space map — seeing the whole universe

Every molecule is embedded two ways:

- **Similarity layout** — the 2,048-bit fingerprints are projected to 2D/3D with **UMAP** using
  the **Jaccard** metric (the right similarity for binary bits — same family as Tanimoto).
  Structurally similar molecules land near each other; taste/aroma classes cluster. UMAP axes have
  no units — only *local* proximity means anything.
- **Property-axes layout** — the *same* molecules replotted on **real, labelled axes**: molecular
  weight × logP (× TPSA in 3D). *That* view is a true metric space — read a molecule's size and
  greasiness straight off the axes, and watch whether a taste/aroma color tracks the physics.

Color by taste, aroma, or **both** (taste where known, else aroma). *(`build_flavor_map.py`.)*

---

## 7. Chirality explorer — where a mirror image changes the smell

Some molecules come in mirror-image / geometric forms that smell different (R-carvone is
spearmint, S-carvone is caraway). We **enumerate every stereoisomer** (all R/S centers *and* E/Z
double bonds; RDKit `EnumerateStereoisomers`), label each with its CIP descriptor, and — because
the models are *achiral* (they read the flat structure) — surface the **documented** odor/taste
where PubChem records it per isomer. Click any isomer to open its own full read. *(`predict.stereoisomers`.)*

---

## 8. Mixture screen — what happens when ingredients meet

Three layers, clearly separated by confidence:

1. **Documented-hazard screen** (**lookup/rule**) — a curated set of *known* precursor→product
   hazards (benzoate + vitamin C → benzene; nitrite + amine → nitrosamine; sugar + asparagine +
   heat → acrylamide…), gated on the process (heat/refining/fermentation) that actually causes
   them. This is a **safety screen, not a reaction predictor** — it fires only on documented
   chemistry.
2. **Reaction-template augmentation** (**qualitative, indicative**) — RDKit reaction-SMARTS
   templates for the flavor-relevant condensations (esterification → fruity esters, Schiff base →
   the Maillard first step, hemi/thio-acetals). A template firing means *the functional groups to
   form the product are present* — "these could plausibly react to X" — never "it happens." Each
   product gets its own predicted taste + aroma.
3. **Palette match** — the single molecule whose **taste + aroma** label-set best matches the
   blend's combined palette (mean of taste-Jaccard and aroma-Jaccard). An honest label-set
   approximation — a blend's real perceived flavor (suppression/synergy) needs the customer's
   formulation data.

---

## 9. Behavior, safety & formulation

- **Behavior** (**computed/estimate/qualitative**) — logP, MW, TPSA, H-bonding, ring/atom counts
  (exact from structure); water solubility (ESOL estimate); volatility tier and pKa ranges
  (qualitative); measured BP/MP/vapor pressure when the property table is loaded. Every line
  carries a "why it matters" note.
- **Safety** (**defensive, caution-only — never a clearance**) — a disclaimer on every result,
  structural tox-alert screening, an optional **GRAS** cross-reference, a preliminary TTC/Cramer
  tier, EU declarable-allergen labeling, and **Tox21** in-vitro assay flags (12 assays, trained on
  public-domain data). It **flags for review; it never clears a compound for use.**

---

## 10. The data behind it — public, crawled, cited

- **`build_properties.py`** enriches all 8,393 molecules with names + measured MP/BP/vapor pressure
  from **public-domain PubChem**.
- **`build_all_molecules.py` / `build_structures.py`** assemble the full unique-molecule universe
  (and backfill structures for reference-only molecules).
- **`build_enrichment.py`** builds the master table (taste + aroma + chemistry per molecule).
- **`build_spectra.py`** records *which* spectra PubChem has per molecule (availability metadata —
  public domain); actual spectra and retention indices we **link** to (PubChem, NIST WebBook),
  never rehost, since NIST data is individual-use.

All sources are cited in the app footer and [`SOURCES`](DATA-SOURCES.md).

---

## 11. Why it's built this way

- **Prediction, not lookup** — the models read *structure*, so they answer for molecules no
  database has.
- **On-premise** — proprietary candidate structures never leave your network; you screen
  confidential molecules without exposing your direction to any external service.
- **One integrated, honestly-tagged read** — taste, aroma, behavior, safety, substitution, and
  formulation in a single screen, each value labelled by how it was derived.
- **It sharpens on your data** — the open-data models are the floor; trained on a customer's own
  formulation and sensory data, on the same on-prem box, they become specific to that customer's
  products — and that's where **aroma intensity** and the richer models come online.

For the exact per-capability confidence and what data unlocks each further step, see
[`CAPABILITIES.md`](CAPABILITIES.md) and [`DATA-REQUIREMENTS.md`](DATA-REQUIREMENTS.md).
