# Tox — caution-only in-vitro assay heads

Flavormancer predicts **flavor** ([`AROMA.md`](AROMA.md), [`MOUTHFEEL.md`](MOUTHFEEL.md)). This
documents the fourth head family, which is **not** a flavor dimension: twelve **Tox21 in-vitro
assay** heads that act as a *defensive screen*.

> **Read this first.** These heads predict whether a molecule would likely be **active in a
> particular in-vitro assay**. Assay activity is **not** toxicity. A flag here means
> *"a human should look at this"* — it is **never** a toxicity finding, a safety determination, a
> hazard classification, or a clearance for use. Flavormancer flags for review; it does not clear
> compounds. Confirm with a toxicologist and the applicable regulatory process.

## Why this is different from the flavor heads

| | flavor heads (taste / aroma / mouthfeel) | tox heads |
|---|---|---|
| label source | **documented + curated** sensory facts (a molecule is *known to be perceived* as X) | **experimental wet-lab screening** (a molecule *measurably was* active in an assay) |
| what a positive means | people report this percept | this well lit up in a dish |
| in the substitute-match vector? | **yes** (175 dims) | **no** — deliberately |
| framing | prediction | **caution-only review flag** |

Tox is excluded from the flavor-profile vector on purpose: you do not want reformulation
substitutes ranked by *shared toxicity signal*. The tox heads run per read and are stored per
molecule for display and filtering, but they never influence what counts as a flavor match.

## Data — Tox21 (public domain)

**Tox21** is a US federal collaboration (**NIH/NCATS · EPA · FDA · NTP**) that quantitatively
screened roughly 8k compounds against 12 in-vitro assays — nuclear-receptor signalling and
stress-response pathways. We use the **MoleculeNet mirror of the public-domain Tox21 Challenge
set** (`tox21.csv`), which carries a measured active/inactive call per compound per assay.

Public domain, US-government-produced (17 U.S.C. §105) — commercial-clean, same discipline as the
rest of the corpus (see [`SOURCES.md`](SOURCES.md)).

**The Tox21 molecules are NOT imported into the flavor universe.** They are mostly industrial and
pharmaceutical compounds — pesticides, drug-likes — and folding them into the molecule universe
would pollute the flavor map, the enrichment table and the substitute search. Tox21 is a
*predictor we apply to our universe*, not new members of it.

## Model

One `RandomForestClassifier` per assay (`n_estimators=200`, `class_weight="balanced"`,
`random_state=42`), trained by `train_tox.py`. An assay ships only if it has **≥30 positives**;
each reports an honest **5-fold CV-AUROC**, written to `tox_models/manifest.json` and surfaced in
the UI next to the bar so nothing reads as more certain than it is.

⚠️ **Featurization differs from the flavor heads.** Tox heads take the **bare 2048-bit Morgan
fingerprint** (`predict._fp`, radius 2), *not* the fingerprint + physicochemical block
(`predict._feat`) that taste/aroma/mouthfeel use. Feeding a tox head `_feat` output raises a
shape error (2060 vs 2048 features) — batch callers must build a separate matrix.

## The twelve heads

| assay | what it probes | CV-AUROC | positives / n |
|---|---|---|---|
| **NR-AhR** | aryl-hydrocarbon receptor (xenobiotic / dioxin-like) | 0.900 | 768 / 6542 |
| **SR-MMP** | mitochondrial membrane potential (mitochondrial toxicity) | 0.878 | 918 / 5804 |
| **NR-AR-LBD** | androgen receptor (ligand-binding domain) | 0.868 | 237 / 6751 |
| **SR-ATAD5** | ATAD5 — genotoxicity / DNA damage | 0.850 | 264 / 7065 |
| **SR-p53** | p53 — DNA-damage response (genotoxic stress) | 0.848 | 423 / 6767 |
| **NR-PPAR-gamma** | PPAR-γ (metabolic) | 0.829 | 186 / 6443 |
| **NR-ER-LBD** | estrogen receptor (ligand-binding domain) | 0.815 | 349 / 6948 |
| **NR-Aromatase** | aromatase (estrogen synthesis) | 0.810 | 300 / 5815 |
| **NR-AR** | androgen receptor | 0.808 | 308 / 7258 |
| **SR-ARE** | oxidative-stress response (ARE) | 0.799 | 942 / 5825 |
| **SR-HSE** | heat-shock response | 0.790 | 372 / 6460 |
| **NR-ER** | estrogen receptor | 0.719 | 791 / 6186 |

All twelve clear the project's 0.70 bar. `NR-ER` sits closest to it — treat its flags with the
least confidence of the set.

## Where it surfaces

- **Read modal** — a *Safety* group in the Heads card: every assay as a ranked %-bar with its
  CV-AUROC, amber-red once an assay crosses 0.5, under a caution-only note.
- **`predict()`** — `safety.tox_screen` (`assays[]` with `probability`, `auroc`, plain-language
  `meaning`; plus the `flags` shortlist at ≥0.5).
- **`master_enrichment.parquet`** — `tox_<assay>` columns plus a comma-separated `tox_flags`, so
  the universe grid is sortable/filterable by safety signal. 1,817 of 8,847 molecules carry ≥1 flag.
- **MCP / skill** — `read_flavor` and the CLI `read` return `tox_flags`.

## Honest limits

- **Assay activity ≠ toxicity.** No dose, no exposure route, no ADME, no in-vivo endpoint.
- **In-vitro only**, and only these twelve pathways — silence here is *not* evidence of safety.
- Trained on a largely industrial/pharmaceutical chemical space; flavor molecules are often
  outside that distribution, so treat out-of-domain reads with extra care.
- Structure-only: no metabolite, impurity or degradation-product screening.
- The separate **structural alert** screen, **TTC/Cramer** tier, **food-use lookup** and **EU
  allergen labeling** are complementary and equally caution-only — see
  [`CAPABILITIES.md`](CAPABILITIES.md).
