# Changelog

All notable changes to Flavormancer. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Flavormancer is pre-1.0. **v1.0.0 is the MVP** — the point at which both tracks (the Python
service and the .NET/React Track B) are running, packaged and documented. Until then, minor
versions ship working increments.

## [Unreleased]

## [0.2.0] — 2026-07-30

The honesty release. Every trained head now publishes how good it actually is, and the packaging
exists to install it somewhere other than the machine it was built on.

### Added
- **Per-head calibrated thresholds with a 50% precision floor.** Each of the 195 heads carries a
  decision threshold fitted on out-of-fold predictions, plus its measured precision and recall,
  published in `/api/heads` and shown on every bar. Heads that cannot be right more than half the
  time are marked `indicative` rather than confident — kept in full, never dressed up.
- **`docs/ACCURACY.md`** — a plain-language explanation of AUROC, precision, thresholds and
  cross-validation, written to be read without a machine-learning background.
- **`training/audit_generalization.py`** — fires every head across the whole corpus and counts
  discoveries outside its training set, to catch heads that memorise rather than learn.
- **Mouthfeel modality** — 5 trigeminal/chemesthesis heads (cooling, warming, pungent, tingling,
  astringent), surfaced across reads, cards, chips and the map.
- **Docker packaging** — `Dockerfile`, `docker-compose.yml` with a pgvector-backed Postgres, and a
  schema for the substitution index. Artifacts mount at runtime rather than baking into the image.
- `FLAVORMANCER_HOME` so the code and the ~1 GB of trained artifacts can live in different places.
- Numbers glossary in `HOW-IT-WORKS.md` and in the app's own How-it-works panel.

### Changed
- Aroma roster **164 → 172 heads**; confident-capable heads **94 → 108**.
- Chip families (flavor / note / taste / mouthfeel) share one visual language instead of four
  accidental ones, and each studio section explains what its dimension *is*.
- Every molecule has a display name: names fall back to molecular formula, with multi-component
  structures labelled as mixtures. **8,861 of 8,861 named**, down from 770 blank.
- Milestones relabelled to say what kind of work they hold (Foundations / Track B / Ship).

### Fixed
- `predict_aroma` had silently lost its `@lru_cache` to an orphaned decorator — the fix behind the
  40s → 0.01s modal read.
- Substitute/neighbor cards overflowed the modal on mobile (`1fr` will not shrink below
  min-content; needed `minmax(0,1fr)`).
- The modal's 14px card gap had never applied, because an inline `display:block` overrode the flex
  column and block boxes ignore `gap`.
- A duplicate `pungent` key in the curated supplement that would have deleted six molecules.

### Known limits
- `sweet`, `ethereal` and `pungent` odour heads sit exactly at the precision floor. They are broad
  *and* chemically incoherent, so curation cannot lift them — tracked for the GNN work (#199).
- 58 aroma heads remain `indicative` (#262).
- Track B (.NET API, React workbench) is scaffolded but not running (M2/M3).

## [0.1.0] — 2026-07-16

First public demo: taste heads, the aroma descriptor model, substitution search, the flavor-space
map, formulation studio, and the on-prem workbench UI.

[Unreleased]: https://github.com/echelonts/flavormancer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/echelonts/flavormancer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/echelonts/flavormancer/releases/tag/v0.1.0
