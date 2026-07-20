# Flavormancer — Capability Catalogue

The full, honest map: what the tool does today, what it *could* do on public data,
what it *cannot* do on public data (and what would unlock it), and the ultimate
shape of the tool. The organizing discipline throughout: **every output is tagged
by how it's derived, and nothing claims more certainty than its source supports.**

> **Edition note.** This catalogues the **commercial** edition (Apache-2.0,
> commercial-clean data). **Aroma ships here** as 24 presence/absence odor-descriptor
> heads trained on public-domain HSDB text — what's still gated is scored **intensity**
> (*how strong* a note is), which needs research/customer panel data and lives in the
> **academic edition** (or trains on a customer's own data, on-prem).

**Confidence tiers used below**
- **computed** — exact from structure (RDKit). Not a prediction; a calculation.
- **trained** — an ML model fit on open data; accuracy varies by data volume.
- **rule** — a validated structural rule (deterministic, transparent).
- **estimate** — a published QSPR with known error, labeled as such.
- **lookup** — a fact from a loaded reference table (only as good as the table).
- **qualitative** — a class/flag, not a number.

---

## A. Live now — built into `predict.py` and the pipeline

**Taste** — six trained heads (RandomForest on Morgan fingerprint + physicochemical features)
- Sweet / bitter / umami — probabilities (**trained**; CV-AUROC ~0.95 / 0.95 / 0.99).
- Sour — **trained** indicative head (0.91) **plus** an acidic-group **rule** as a cross-check.
- Salty — **trained** indicative head (0.93) **plus** a cation-aware inorganic-salt **rule**.
- Tasteless — **trained** tasted-vs-tasteless head (0.89).
- Sweetness intensity — vs-sucrose strength (**trained**, regressor on SweetenersDB).
- Multitaste — fires when 2+ taste heads are high (**trained**-derived).
- Known-taste ground truth — verified labels override predictions (**lookup**).

**Aroma** — **41 odor-descriptor heads ship** (**trained**)
- Presence/absence per descriptor (citrus, floral, minty, almond, fatty, petroleum, earthy,
  medicinal, sulfurous, camphor, fruity, fishy, garlic, ethereal, ammoniacal, pungent) —
  RandomForests on fingerprint + physicochemical features, over public-domain HSDB odor text +
  curated character-impact facts. CV-AUROC **0.73–0.97**; surfaced for **any** molecule.
- Documented odor + detection thresholds shown where cited (**lookup**, public-domain HSDB).
- **Not** yet: scored **intensity** — needs research/customer panel data (the gated upgrade).
  See [`AROMA.md`](AROMA.md).

**Physicochemical & behavior**
- logP, molecular weight, TPSA, H-bond donors/acceptors, rotatable bonds,
  aromatic rings, heavy atoms (**computed**).
- Water solubility, logS (**estimate**, ESOL ~0.7 log RMSE).
- Aroma-volatility tier — top/middle/base note (**qualitative**).
- Ionizable groups + typical pKa ranges (**qualitative**).
- Measured BP / vapor pressure / threshold when a property table is loaded (**lookup**).

**Stability**
- Oxidation / hydrolysis / photodegradation watch-flags (**rule/qualitative**).

**Chemesthesis (trigeminal)**
- Cooling / pungent / astringent class flags (**rule/lookup**, qualitative).

**Safety (all defensive, caution-only — never a clearance)**
- Disclaimer + scope on every result.
- Structural tox-alert screen — nitro/N-nitroso/azo/epoxide (**rule**).
- GRAS / approved-ingredient cross-reference — FDA SAF, public domain, `build_gras_reference.py` (**lookup**).
- **In-vitro tox-assay flags** — 12 Tox21 assays (genotoxic-stress SR-p53/SR-ATAD5, AhR,
  mitochondrial, endocrine), `predict_tox()` (**trained**, Tox21 public domain). INDICATIVE
  activity for review, **never a determination**.
- Preliminary TTC concern tier (**qualitative** heuristic; Toxtree for the real call).
- EU declarable fragrance-allergen labeling flag — `labeling()` (**lookup**, curated subset).

**Formulation-level**
- `check_mixture(ingredients, processes)` — process-aware documented hazard screen:
  benzene, nitrosamine, ethyl carbamate (incl. citrulline route), acrylamide, furan,
  3-MCPD/glycidyl esters, 4-methylimidazole, and biogenic amines (histamine/tyramine).
  Hazards needing a process (high_heat / refining / fermentation) show as **active**
  when it's declared, **conditional** otherwise (**lookup/rule**).
- `analyze_balance()` — OAV ranking (concentration ÷ odor threshold), "overbearing"
  flag, FEMA-max overdose flag; quantitative when thresholds loaded, **qualitative**
  volatility ranking otherwise.

**Analytical, search & workbench**
- **Flavor Studio** — one picker over everyday flavors *and* notes → ranked food-safe
  molecules + drop-in swaps (the "molecule behind strawberry", live).
- **Flavor-space map** — 2D/3D **similarity** (UMAP) + **interpretable property axes**
  (MW × logP × TPSA), colored by taste / aroma / both.
- **Chirality explorer** — every stereoisomer (R/S + E/Z); isomers with distinct
  documented odor/taste surfaced as their own reads.
- **Master enrichment table** — the whole universe, sortable/searchable, each column
  annotated with why it matters.
- **Nearest-neighbor substitution search** (Tanimoto/Morgan; pgvector at the product layer).
- **External references** — per-molecule links to PubChem (public domain) + NIST WebBook
  for spectra (IR/MS/NMR) and GC retention index, with an availability flag (**link, not host**).
- GC-MS Kovats retention index as an on-read *value* — data-gated (licensed/customer table); see B.

---

## B. Buildable on open / public data — "could add if we wanted"

Each needs the noted public data and tops out at the noted confidence. Most are
either *lookups awaiting a dataset* or *quantitative upgrades* of something above.

**High value, clean to build**
- **Bitterness intensity** — the parallel to sweetness intensity. Data: BitterDB. *(trained.)*
- **Quantitative dosing** — load published **odor thresholds** + **FEMA usual/max
  use levels** to make OAV/balance fully quantitative. Data: threshold compilations,
  FEMA lists. *(lookup — already wired, needs the tables.)*
- **GC-MS Kovats RI prediction** — solid QSPR; squarely in the bench workflow. Data:
  public NIST/literature RI sets. *(trained.)*
- **Natural occurrence / food-source mapping** — "the molecule behind strawberry."
  Great demo context. Data: FlavorDB. *(lookup.)*
- **Full allergen annex + natural-vs-nature-identical-vs-artificial status** — the
  complete EU allergen list (we ship a curated subset) + regulatory status. Data:
  regulatory lists. *(lookup.)*

**Feasible, with caveats**
- **Tox QSAR endpoints** — Ames mutagenicity (well-studied), hepatotox/carcinogen
  categories (ProTox/Tox21), indicative LD50. Caution-only flags. Data: Tox21/ToxCast,
  public Ames sets. *(trained — noisier; frame carefully.)*
- **Full Cramer/TTC classification** — integrate **Toxtree** for the validated tree
  (vs our preliminary hint). *(rule, external tool.)*
- **Further contaminant screens** — patulin, methanol (pectin route), and others
  beyond the documented set now built in. *(lookup/rule — precursor signatures.)*
- **Better pKa** — open models (e.g. MolGpKa) for per-molecule pKa vs our group ranges. *(trained.)*
- **Henry's-law / air–water partition** — aroma-release rate. *(estimate/lookup; QSPR or measured.)*
- **HLB / surfactant behavior** — foaming/emulsion hints for surfactant-like molecules. *(estimate.)*
- **Approximate color** — chromophore/conjugation → rough color for colored
  compounds. *(estimate — limited.)*
- **3D / conformer descriptors** — shape/volume for finer aroma modeling. *(computed.)*
- **Synonym / identifier resolution** — name↔SMILES↔CAS via PubChem for UX. *(lookup.)*

**Explicitly evaluated and rejected (accuracy)**
- **Structure-based boiling point (Joback)** — re-validated across 12 flavor molecules:
  **mean abs error 33 °C, max 89 °C** (benzaldehyde −89, vanillin −64, methyl salicylate
  +53) — near-perfect on simple aliphatics (limonene, ethyl acetate) but badly off on the
  aromatic/oxygenated molecules that matter most, and you can't tell which in advance. So
  BP/vapor pressure stays a **measured lookup** (now PubChem-sourced via `build_properties.py`),
  never an estimate.

---

## C. Cannot do on open data — fundamental limits (and what unlocks them)

These are not effort problems; the data does not exist publicly. Each is unlocked
by a *specific* private asset — proprietary data the model would need.

- **Finished-blend perception.** A blend isn't the sum of its molecules —
  suppression, masking, synergy reshape it. Predicting how a *formulation* actually
  tastes is unsolved publicly. → Unlocked by **their formulation→outcome archive.**
- **Consumer liking / preference.** "Will people enjoy this" needs panel/consumer
  data. → Unlocked by **sensory-panel + consumer datasets** (the Gastrograph moat).
- **Mouthfeel / body / carbonation in a real matrix.** Formulation- and
  process-level, not molecular. → Unlocked by **their matrix + process data.**
- **Their proprietary product outcomes.** Accuracy on *their* specific products. →
  Unlocked by **a customer's own formulation archive.**
- **Shelf-life / stability in their matrix.** Real degradation depends on pH, temp,
  light, packaging, matrix. → Unlocked by **their stability-test data.**
- **Exact reaction-byproduct yields** in a given process (beyond known precursor
  flags). → Needs **reaction + kinetics + matrix data**; largely an analytical/lab task.
- **Cost / sourcing optimization.** → Unlocked by **their supplier/cost data.**
- **Novel-molecule safety / GRAS determination.** Not a modeling gap — a
  *regulatory* one. This is never a model's call; it belongs to toxicologists and
  the FEMA GRAS / FDA process. The tool flags and cautions, never clears.

---

## D. The complete tool — everything considered

The maximal *honest* tool, on-prem, one screen:

**Enter a molecule** (name or SMILES) → a complete single-molecule dossier: taste
(sweet/bitter/umami + sweetness intensity + sour/salty rules), aroma descriptor
profile *(academic edition, or trained on your data)*, behavior
(logP/MW/TPSA/solubility/volatility/pKa, measured props where
loaded), stability watch-flags, chemesthetic class (cooling/pungent/astringent),
an analytical RI estimate, and a layered safety panel (alerts + GRAS + TTC hint) —
each value tagged computed/trained/rule/estimate/lookup/qualitative.

**Enter a formulation** (ingredients + concentrations) → an OAV balance analysis
that ranks impact and flags the component about to overpower the blend, dosing
checks against FEMA max use levels, and the documented dangerous-pair hazard
screen — with the explicit caveat that this is single-molecule impact, not
finished-blend perception.

**Explore** → a flavor-space map and nearest-neighbor substitution search
(structural similarity today; learned embeddings once the aroma model is in) for
cost-down/reformulation, all running on hardware the
customer owns, with their data never leaving the building.

**The honest boundary:** everything above is *single-molecule prediction and
curated lookups on public data* — comprehensive and transparent. The three things
it deliberately can't do — predict how a *blend* actually tastes, whether
*consumers* will like it, and how it behaves in a real *matrix* — are precisely
what a customer's own formulation data unlocks. **The public-data ceiling is not a
weakness; it's the boundary of what public data can do, by design.**
