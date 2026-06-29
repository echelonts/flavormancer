# Flavormancer — Capability Catalogue

The full, honest map: what the tool does today, what it *could* do on public data,
what it *cannot* do on public data (and what would unlock it), and the ultimate
shape of the tool. The organizing discipline throughout: **every output is tagged
by how it's derived, and nothing claims more certainty than its source supports.**

> **Edition note.** This catalogues the **commercial** edition (Apache-2.0,
> commercial-clean data). **Aroma** (odor descriptors) is **not** here — it moves to
> the open-source **academic edition** (coming soon), which adds research odor data
> carrying NonCommercial terms. In the commercial product, aroma trains on the
> customer's own licensed / in-house data.

**Confidence tiers used below**
- **computed** — exact from structure (RDKit). Not a prediction; a calculation.
- **trained** — an ML model fit on open data; accuracy varies by data volume.
- **rule** — a validated structural rule (deterministic, transparent).
- **estimate** — a published QSPR with known error, labeled as such.
- **lookup** — a fact from a loaded reference table (only as good as the table).
- **qualitative** — a class/flag, not a number.

---

## A. Live now — built into `predict.py` and the pipeline

**Taste**
- Sweet / bitter / umami — probabilities (**trained**, RandomForest on merged open taste DBs).
- Sweetness intensity — vs-sucrose strength (**trained**, regressor on SweetenersDB).
- Sour — acidic-group flag (**rule**).
- Salty — cation-aware inorganic-salt flag with organic-anion guard (**rule**).
- Multitaste — fires when 2+ taste heads are high (**trained**-derived).
- Known-taste ground truth — verified labels override predictions (**lookup**).

**Aroma** — *not in the commercial edition.*
- Odor-descriptor prediction needs research odor datasets with NonCommercial terms,
  so it ships in the separate **academic edition** (coming soon), not here. In the
  commercial product, aroma trains on **your** licensed / in-house odor data, on-prem.
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
- GRAS / approved-ingredient cross-reference (**lookup**, data-gated).
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

**Analytical & search**
- GC-MS Kovats retention index — honest hook/stub (a trainable QSPR, see B).
- Flavor-space map + nearest-neighbor substitution search (pgvector; demo layer).

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
- **Structure-based boiling point (Joback)** — validated and *declined*: benzaldehyde
  missed by ~90 °C. BP/vapor pressure stays a **measured lookup**, not an estimate.

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
