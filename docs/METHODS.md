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

## The honest ceiling on "deeper flavor description"
Taste tops out at the **5 basics + intensity + chemesthesis** on public data. The rich
descriptors people mean by "flavor" — *vanilla, fruity, green, woody, minty, caramel…* — are
**aroma**, not taste: a separate odor model that needs expert-labeled odor data (licensed or
customer; deferred — see [`AROMA.md`](AROMA.md)). That model is the real route to deeper flavor
language, and it's exactly the "comes with your data" piece.
