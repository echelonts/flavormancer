# Flavormancer

On-prem flavor & aroma prediction from chemical structure — taste, odor,
physicochemical behavior, dosing, and safety flags. Enter a molecule by name or
SMILES and get a single, honest flavor read, running entirely on hardware you own.

> **Status:** early development. The training pipeline and prediction core exist;
> the .NET serving layer, React workbench, and packaging are in progress. See the
> [roadmap](ROADMAP.md) for what's landed (M0–M1) and what's next (M2–M6).

---

## What it does

Given one molecule, Flavormancer returns a structured profile where **every value
is tagged by how it was derived**, so nothing reads as more certain than its source:

- **Taste** — sweet / bitter / umami probabilities (trained models), sweetness
  intensity (regressor), plus sour and salty as transparent chemistry rules.
- **Aroma** — odor-descriptor profile from an open principal-odor-map model.
- **Behavior** — logP, molecular weight, TPSA, H-bonding (computed), solubility
  (estimate), volatility tier and pKa ranges (qualitative).
- **Stability & chemesthesis** — oxidation/hydrolysis/photo watch-flags; cooling /
  pungent / astringent class flags.
- **Safety** — defensive, caution-only: a disclaimer and scope statement on every
  result, structural-alert screening, and an optional GRAS cross-reference. It
  **flags for review; it never clears a compound for use.**

**Scope:** Flavormancer predicts *taste and aroma only*. It is not a safety,
toxicity, GRAS, regulatory, or stability determination. A prediction is never a
clearance to consume.

## Confidence tiers

Outputs are labeled with how they were produced: **computed** (exact from
structure), **trained** (ML on open data), **rule** (deterministic structural
rule), **estimate** (published QSPR with known error), **lookup** (from a loaded
reference table), and **qualitative** (a class/flag, not a number).

## Architecture

Python trains the models offline; a .NET application serves them at runtime. The
training language is a build-time detail — nothing at runtime depends on Python
(aside from an optional aroma sidecar if the GNN won't export cleanly).

```
data sources ─► Python training (build-time) ─► ONNX (taste) ──┐
                RDKit · scikit-learn · OpenPOM   aroma model ─┐ │
                                                              ▼ ▼
React workbench ◄─ JSON API ◄─ ASP.NET Core + ONNX Runtime + Postgres/pgvector
                                              │
                                              ▼
                              Docker Compose on a single on-prem box
```

| Layer | Technology |
|-------|-----------|
| Model training (build-time) | Python · RDKit · scikit-learn · OpenPOM/DeepChem |
| Model handoff | ONNX |
| App / API | ASP.NET Core (C#) |
| ML serving | ONNX Runtime, in-process in .NET |
| Frontend | React |
| Database | PostgreSQL + pgvector |
| Deploy | Linux + Docker Compose (single box) |

## Repository layout

```
training/        Python — dataset build + model training (build-time)
aroma-sidecar/   Python — thin aroma inference service (only if needed)
api/             ASP.NET Core — app, auth, endpoints, ONNX serving
frontend/        React — the workbench UI
infra/           Dockerfiles, docker-compose.yml, deploy
docs/            architecture, capabilities, and design docs
```

## Getting started

A full clean-machine setup walkthrough lands with the training pipeline (see
`docs/`). Datasets and trained models are **not** committed — the training
scripts download their sources, and `.gitignore` keeps artifacts out of the repo.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching, commit conventions, and the
review workflow. Commits are signed; PRs are small, linked to an issue, and
squash-merged after review.

## License

Licensed under the [Apache License 2.0](LICENSE). Code license only — datasets and
pretrained models carry their own licenses, noted where they're used.
