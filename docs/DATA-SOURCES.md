# Data Sources — acquisition tracker

What feeds the models, what each source unlocks, its license, and how to get it.

> **The data files themselves are never committed** (licensing + size — see
> `.gitignore`). The training scripts download them; this file records provenance,
> status, and the acquisition steps. Loaders **skip cleanly** when a file is absent,
> so adding a source is just dropping its file in and re-running the pipeline.

## Status

| Source | Unlocks | License | Status | How to get |
|--------|---------|---------|--------|-----------|
| **ChemTastesDB v2.0** | sweet/bitter/umami training + sour/salty rule data (primary) | CC-BY-4.0 | ✅ in use | Zenodo record `14963136`, direct download (see `training/SETUP.md`). Column mapping verified. |
| **cosylab/bittersweet** | more sweet/bitter volume | AGPL-3.0 | ✅ columns verified; opt-in use | `git clone github.com/cosylabiiit/bittersweet` → `bittersweet/data/*.tsv`. AGPL: keep a CC-BY-clean build with `INCLUDE_COSYLAB=False`, or opt in deliberately. |
| **FlavorDB** | ~25k molecules taste+odor + natural-source mapping | **CC BY-NC-SA 3.0** | ❌ skip | NonCommercial — incompatible with a commercial product. |
| **UMP442 / BIOPEP-UWM** | more umami examples | none / unclear | ❌ skip | BIOPEP-UWM is web-only; the `Shoombuatong/Dataset-Code` repost has **no LICENSE** (all-rights-reserved) and is umami *peptide* data (different class from our small-molecule head). |
| **SweetenersDB v2.0** | the sweetness-**intensity** regressor | **MIT** | ✅ in use | direct CSV from `github.com/chemosim-lab/SweetenersDB` (`SweetenersDB_v2.0.csv`, 316 cmpds, `logSw` column). R²≈0.82. |
| **Pyrfume / Leffingwell** | the **aroma model** (OpenPOM) — issues #17 / #18 | per-set; confirm | ⬜ to get | `git clone github.com/pyrfume/pyrfume-data`; Leffingwell odor set. Separate, version-fragile effort. |
| **FEMA GRAS / FDA SAF** | GRAS cross-reference + dosing/OAV lookups | gov public / FEMA | ⬜ to get | FDA "Substances Added to Food" (public domain) → `gras_reference.parquet`; FEMA use-level PDFs → `properties.parquet`. **Verify every scraped dosing number.** |

## Column verification

Verified on 2026-06-27 by downloading/opening the files referenced in
`training/SETUP.md` and checking them against `training/build_taste_dataset.py`.

| Source file | Rows | Real columns | Loader mapping |
|-------------|------|--------------|----------------|
| `ChemTastesDB_database.xlsx` (`ChemTastesDB` sheet) | 4078 | `ID`, `Name`, `PubChem CID`, `CAS number`, `canonical SMILES`, `Taste`, `Class taste`, `Reference_(cod)/[pp]` | `canonical SMILES` → SMILES; `Class taste` → coarse class; `Taste` → granular multi-label taste text. |
| `bittersweet/data/bitter-train.tsv` | 2257 | `Name`, `Taste`, `Reference`, `SMILES`, `Canonical SMILES`, `Bitter` | Use `SMILES`; label column is `Bitter` (`True`/`False`). |
| `bittersweet/data/bitter-test.tsv` | 171 | `Name`, `Taste`, `Reference`, `SMILES`, `Canonical SMILES`, `In Bitter Domain`, `Bitter` | Use `SMILES`; label column is `Bitter` (`True`/`False`). `In Bitter Domain` is metadata, not the target. |
| `bittersweet/data/sweet-train.tsv` | 2205 | `Name`, `Taste`, `Reference`, `SMILES`, `Canonical SMILES`, `Sweet` | Use `SMILES`; label column is `Sweet` (`True`/`False`). |
| `bittersweet/data/sweet-test.tsv` | 161 | `Name`, `Taste`, `Reference`, `SMILES`, `Canonical SMILES`, `In Bitter Domain`, `Sweet` | Use `SMILES`; label column is `Sweet` (`True`/`False`). `In Bitter Domain` is metadata, not the target. |
| `SweetenersDB/SweetenersDB_v2.0.csv` | 316 | `ID`, `Name`, `logSw`, `Smiles` | `Smiles` matches the loader's case-insensitive `smiles` candidate; `logSw` matches `logsw` and is exported as `log_sweetness`. |
| `SweetenersDB/SweetnersDB_v1.0.csv` | 316 | `ID`, `Name`, `logS`, `Smiles` | Historical file. The current intensity loader expects v2.0 `logSw`; do not use v1.0 for `sweet_intensity.parquet` without a separate mapping decision. |

`FlavorDB` and `UMP442/BIOPEP-UWM` are intentionally not verified here: the project
currently marks FlavorDB as skipped because of its NonCommercial license, and the umami
peptide sources are marked skipped because licensing/source suitability is unclear. Their
`[OBTAIN]` markers remain data-acquisition decisions rather than missing column checks.

## Notes

- **Licensing.** ChemTastesDB (CC-BY-4.0) is the clean base. cosylab is AGPL — gated
  behind the `INCLUDE_COSYLAB` flag. Get an IP/OSS-license review before any
  commercial release; full attribution lives in [`SOURCES.md`](SOURCES.md).
- **Column verification.** Each new source's column names must be checked against the
  loaders (the `[VERIFY]` markers) — see issue **#25**. ChemTastesDB, cosylab
  bittersweet, and SweetenersDB v2.0 mappings are resolved above.
- **Suggested priority.** (1) `Taste`-column mining — **done** ✅ → (2) SweetenersDB
  intensity head — **done** ✅ → (3) Pyrfume / Leffingwell for the aroma model →
  (4) cosylab for taste volume (AGPL — pending decision). **FlavorDB is
  CC BY-NC-SA (NonCommercial) — incompatible with a commercial product; skip.**
