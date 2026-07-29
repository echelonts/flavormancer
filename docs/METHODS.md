# Methods, Rules & Heuristics

Every non-obvious rule, threshold, and trick Flavormancer uses — with its rationale and
its ceiling. The discipline throughout: each output is tagged by **how** it's derived,
and a rule is used only where it's *more honest* than a model. Companion to
[`CAPABILITIES.md`](CAPABILITIES.md) (what it does) and [`SOURCES.md`](SOURCES.md) (the data).

**Confidence tiers:** `computed` (exact from structure) · `trained` (ML on open data) ·
`rule` (deterministic structural rule) · `estimate` (published QSPR, known error) ·
`lookup` (loaded reference table) · `qualitative` (a class/flag, not a number).

---

## Taste
- **Sweet / bitter / umami** — `trained`. RandomForest on 2048-bit Morgan radius-2 fingerprints.
- **Sweetness intensity** — `trained`, est. RandomForest regressor on **log₁₀(relative-to-sucrose)**
  sweetness (SweetenersDB). Shown as a multiplier (10^value). **Gate:** displayed only when
  sweet ≥ 0.5 **and logP < 2** — sweeteners are hydrophilic, so the logP cutoff suppresses
  lipophilic false-positives (the classifier wrongly calling limonene "sweet") while keeping
  real sweeteners (sugars, aspartame). Tagged "(est.)"; the value is the model's, not literature.
- **Sour** — `rule`. SMARTS for acidic groups (carboxylic / sulfonic / phosphonic) matching
  **both protonated and deprotonated** forms (this lifted recall 0.57 → 0.93). Sourness is a
  solution/pH property, so a structural proxy is the honest move, not a per-molecule ML target.
- **Salty** — `rule`. Fires only for a salt-forming cation (alkali metal Li/Na/K/Rb/Cs or
  ammonium) **+ a simple INORGANIC anion**. **Defers** when the anion carries carbon — so MSG
  (umami), Na-saccharin (sweet), Na-benzoate (preservative) are NOT called salty. Refuses the
  naive "has sodium → salty" mistake.
- **Known-taste override** — `lookup`. A verified dataset label beats the rule/model and is
  marked "verified."
- **taste_profile / multitaste** — trained heads ranked by probability; multitaste flag when ≥2 fire.

## Applicability domain
- **No-carbon → out of domain.** The trained taste/tox heads are fit on *organic* molecules;
  for carbon-free inputs (water, O₂, N₂, NaCl) their output is meaningless, so the UI suppresses
  them with a banner. Rules (sour/salty), structure, and computed properties stay valid.

## Physicochemical
- **logP / MW / TPSA / H-bond donors-acceptors / rings / heavy atoms** — `computed` (RDKit, exact).
- **Water solubility, logS** — `estimate` (ESOL, Delaney 2004; ~0.7 log RMSE).
- **Volatility tier** — `qualitative`. Heuristic from MW + H-bond donors + TPSA → top/middle/base note.
- **Boiling point / vapor pressure** — `lookup`. Measured values from PubChem (public domain).
  **Structure-based BP (Joback) was evaluated and rejected** (33 °C mean / 89 °C max error across
  12 flavor molecules). The BP parser **prefers atmospheric (~760 mmHg) readings**; when only
  reduced/elevated-pressure data exists it reports the value **with its pressure**, not a bare
  misleading number.
- **pKa** — `qualitative`. Typical ranges for detected ionizable groups, not a per-molecule value.

## Stability
- **Oxidation / hydrolysis / photo watch-flags** — `rule`/`qualitative`. SMARTS for reactive
  motifs (e.g. phenol/aldehyde → oxidation). "Watch for," not a shelf-life prediction.

## Chemesthesis (trigeminal — a dimension beyond taste)
- **Cooling / pungent / astringent** — `rule`/`lookup`. SMARTS (isothiocyanate → pungent;
  ≥3 phenols → astringent) + InChIKey lookups (menthol → cooling/TRPM8; capsaicin & piperine →
  pungent/TRPV1).

## Safety (all defensive, caution-only — never a clearance)
- **Structural tox-alerts** — `rule`. A SMALL curated SMARTS set: aromatic nitro, N-nitroso,
  aromatic azo, epoxide. Deliberately limited to motifs **rare in the GRAS flavor palette** to
  avoid alert fatigue — we do NOT flag aldehydes or Michael acceptors (too many GRAS flavors
  carry them). Prompts for review; **not a comprehensive carcinogen detector** (e.g. furan isn't flagged).
- **GRAS / food-use cross-check** — `lookup`. InChIKey-skeleton match against FDA "Substances
  Added to Food" (public domain).
- **TTC / Cramer tier** — `qualitative`. Preliminary concern tier; Toxtree for the validated call.
- **Tox21 assay flags** — `trained`, caution-only. 12 RandomForest heads (genotoxic-stress, AhR,
  mitochondrial, endocrine); flagged at probability ≥ 0.5; indicative in-vitro signals, never a
  determination. Suppressed for out-of-domain molecules.
- **EU declarable-allergen labeling** — `lookup`. InChIKey match against a curated EU allergen subset.

## Mixtures
- **check_mixture** — `lookup`/`rule`. Detect molecular **roles** (benzoate, nitrite, secondary
  amine, urea, ascorbate, ethanol, asparagine, citrulline, reducing sugar, …) via SMARTS +
  InChIKey, then fire **documented** precursor→product hazards (benzene, nitrosamine, ethyl
  carbamate, acrylamide, furan, 3-MCPD, 4-MEI, biogenic amines) when the required roles co-occur,
  **gated on declared process** (high heat / refining / fermentation) → active vs conditional.
  A **curated documented-hazard screen, NOT a reaction predictor** — and precise: it distinguishes
  **nitrite** (forms nitrosamines) from **nitrate** (does not).
- **Palette match** — `rule`. Jaccard similarity over the 5-basic-taste label sets: union the
  mixture's ingredient tastes, find single labeled molecules with the closest set. NOT blend
  perception — a blend ≠ the sum of its parts (suppression/synergy needs formulation data).

## Search & resolution
- **Substitution** — Tanimoto over Morgan fingerprints (self excluded), each neighbor with its known tastes.
- **Names / structure** — PubChem name/CAS→SMILES resolution; common (Title) + IUPAC names (cached);
  RDKit 2D depiction (graceful without libXrender); typeahead over the curated flavor-volatile list.

---

## Does a head LEARN, or just MEMORIZE?
A high cross-validated AUROC is necessary but **not sufficient**, and this is the single easiest
place to fool yourself. A head trained on twelve molecules that all share one scaffold can score
0.99 by recognizing that scaffold and nothing else — it fires on exactly its own training
molecules and stays silent on every other molecule in the corpus. Cross-validation cannot see
this, because it only ever asks about molecules *inside* the labelled set.

- **Generalization test** — `computed`. Fire the head over the **whole corpus**, then count the
  hits that are **not** in its training positives. That count is the head's discovery power.
  Run it with [`training/audit_generalization.py`](../training/audit_generalization.py).
- **Two thresholds, because zero is ambiguous.** A head with 13 positives against 2400 negatives
  is calibrated conservatively even with balanced class weights — it can have learned its class
  and still rarely clear 0.5 outside the molecules it was fit on. So the audit also scores at
  **0.35**, which splits one number into two very different diagnoses:
  - `novel@0.5 > 0` → **generalizes**.
  - `novel@0.5 = 0 < novel@0.35` → **under-confident**. It found real unlabelled molecules just
    below the bar. The class is learnable and the head isn't broken; more positives sharpen it.
  - `novel@0.35 = 0` → **memorizing**. Fires on its training set and nothing else, at any
    threshold. This is the real failure.
- Worth stating plainly because it bit us: `pine` and `rosemary` read as memorizing at 0.5 and
  turned out to be under-confident (8 and 9 novel hits at 0.35). `celery` and `turmeric` were
  memorizing at both. Same table, opposite verdicts, opposite fixes.
- **Reading the result.** A memorizing head is not worthless — it still labels its own positives
  correctly — but it must not be presented as if it can *discover*, and it is not evidence the
  model learned the class.
- **The fix is diversity, not volume.** `tingling` trained on nine *Zanthoxylum* sanshools learns
  "sanshool"; the same head trained on sanshools **plus** *Echinacea*, *Anacyclus* and *Heliopsis*
  amides learns "long-chain unsaturated N-alkylamide" and starts finding molecules nobody
  labelled. More of the same scaffold never moves the count off zero.
- **Some heads can't be fixed with molecules.** A broad, fuzzy, multi-scaffold class like `sweet`
  **odour** (AUROC 0.724 over 208 positives) isn't thin — it's genuinely hard. The honest answer
  there is a better model (a GNN), not a longer list.

## The honest ceiling on "deeper flavor description"
Taste tops out at the **5 basics + intensity + chemesthesis** on public data — that ceiling is
real and hasn't moved. The rich descriptors people mean by "flavor" — *vanilla, fruity, green,
woody, minty, caramel…* — are **aroma**, a separate modality with its own heads (see
[`AROMA.md`](AROMA.md)), now trained from open sources rather than deferred to licensed data.

What remains gated is **depth**, not vocabulary: the aroma heads are weakly labelled from public
odor text plus a hand-curated character-impact supplement, so they are strongest on the classic,
widely-documented associations and thinnest on the rare naturals — which is precisely what the
generalization test above measures and reports honestly. Expert-labelled odor panel data (licensed
or customer-supplied) is still the route to depth, and it remains the "comes with your data" piece.
