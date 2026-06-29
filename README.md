# Flavormancer

On-prem flavor prediction from chemical structure — **taste, physicochemical
behavior, dosing, and safety flags** for any molecule, plus substitution search.
Enter a **common or IUPAC name** (or a SMILES) and get a single, honest flavor read,
running entirely on hardware you own.

> **This is the commercial edition** — Apache-2.0 and **commercial-clean**: every
> model trains only on permissively-licensed open data, so it's free to use, sell,
> and run behind your firewall. A more robust, **open-source academic edition with
> the full aroma model is coming soon** — see [Two editions](#two-editions).

> **Status:** the Python training pipeline and prediction core are built and
> tested; the .NET serving layer, React workbench, and packaging are in progress.
> See the [roadmap](ROADMAP.md) — M0–M1 landed, M2–M6 next.

---

## What it does

Given one molecule, Flavormancer returns a structured profile where **every value
is tagged by how it was derived**, so nothing reads as more certain than its source:

- **Taste** — sweet / bitter / umami probabilities (trained models) and sweetness
  intensity (regressor), plus sour and salty as transparent chemistry rules.
- **Behavior** — logP, molecular weight, TPSA, H-bonding, ring/atom counts
  (computed); water solubility (ESOL estimate); volatility tier and pKa ranges
  (qualitative); measured boiling point / vapor pressure when a property table is
  loaded (lookup — structure-based BP was evaluated and declined as too inaccurate).
- **Stability & chemesthesis** — oxidation / hydrolysis / photo watch-flags;
  cooling / pungent / astringent class flags.
- **Safety (defensive, caution-only)** — a disclaimer + scope on every result,
  structural tox-alert screening, a preliminary TTC/Cramer concern tier, an optional
  GRAS cross-reference, and EU declarable-allergen labeling. It **flags for review;
  it never clears a compound for use.**
- **Formulation** — a documented dangerous-mixture screen (benzene, nitrosamine,
  acrylamide, ethyl carbamate, furan, and more) and an OAV dosing-balance analysis
  that flags the component about to overpower a blend (quantitative when threshold
  tables are loaded).
- **Substitution search** — nearest-neighbor lookup over the labeled set for
  reformulation and cost-down ("find me a molecule that behaves like this one").

**Aroma** (odor-descriptor prediction) is **not** built into this edition and isn't
pre-trained. It *can* be trained for a customer on **their own odor data**, or on a
**commercially-licensed** dataset (e.g. Leffingwell **PMP 2001**), when that data is
provided. The full *open* aroma model — trained on research odor data that carries
NonCommercial terms — lives in the academic edition.

> *What aroma training needs from you:* molecules (SMILES, or **GC-MS** to identify
> the compounds in your products) paired with your panel's **expert odor descriptors**
> (e.g. green / fruity / woody, ideally with intensity). GC-MS identifies the
> molecules; the sensory labels are what the model learns — GC-MS alone isn't enough.

For exactly what data unlocks each *further* capability (aroma, quantitative dosing,
retention index) and the formats we accept from a client, see
[docs/DATA-REQUIREMENTS.md](docs/DATA-REQUIREMENTS.md).

**Scope:** Flavormancer predicts *flavor properties only*. It is not a safety,
toxicity, GRAS, regulatory, or stability determination. A prediction is never a
clearance to consume.

## Confidence tiers

Every output is labeled by how it was produced: **computed** (exact from
structure), **trained** (ML on open data), **rule** (deterministic structural
rule), **estimate** (published QSPR with known error), **lookup** (from a loaded
reference table), and **qualitative** (a class/flag, not a number). Full map in
[docs/CAPABILITIES.md](docs/CAPABILITIES.md).

## Two editions

Flavormancer ships as two editions of one method:

| | **Flavormancer** (this repo) | **Flavormancer Research** |
|---|---|---|
| Edition | Commercial | Academic / open-source *(coming soon)* |
| License | Apache-2.0 | open-source, **research / NonCommercial** |
| Data | commercial-clean open data only | adds research odor datasets with **NonCommercial** terms |
| Aroma | *would* be trained on your own data, or a licensed set (e.g. PMP 2001), when provided | full open model included (research odor data) |
| Use | free to use, sell, run on-prem | research, teaching, advancing the method |

The split is deliberate. The richest aroma data is licensed for research only, so
it can't ship in a product you sell — keeping it out is exactly what makes this
edition clean to use and sell, and the academic edition is where that fuller model
lives.

## Architecture

Python trains the models offline; a .NET application serves them at runtime —
nothing at runtime depends on Python.

```
data sources ─► Python training (build-time) ─► ONNX (taste) ──┐
                RDKit · scikit-learn                            │
                                                                ▼
React workbench ◄─ JSON API ◄─ ASP.NET Core + ONNX Runtime + Postgres/pgvector
                                              │
                                              ▼
                              Docker Compose on a single on-prem box
```

| Layer | Technology |
|-------|-----------|
| Model training (build-time) | Python · RDKit · scikit-learn · skl2onnx |
| Model handoff | ONNX |
| App / API | ASP.NET Core (C#) |
| ML serving | ONNX Runtime, in-process in .NET |
| Frontend | React |
| Database | PostgreSQL + pgvector |
| Deploy | Linux + Docker Compose (single box) |

## Repository layout

```
training/        Python — dataset build + model training (build-time)
api/             ASP.NET Core — app, auth, endpoints, ONNX serving
frontend/        React — the workbench UI
infra/           Dockerfiles, docker-compose.yml, deploy
docs/            architecture, capabilities, and design docs
tests/           pytest suite for the prediction core
```

## Getting started

See [training/SETUP.md](training/SETUP.md) for the clean-machine setup. Datasets and
trained models are **not** committed — the training scripts pull their sources and
`.gitignore` keeps artifacts out of the repo.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching, commit conventions, and the
review workflow. Commits are signed and signed off (DCO + [CLA](CLA.md)); PRs are
small, linked to an issue, and squash-merged after review.

## License

Licensed under the [Apache License 2.0](LICENSE) — code license only; datasets and
any pretrained models carry their own licenses, noted where used. The academic
edition is a separate repository under its own (NonCommercial) terms.
