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
| **cosylab/bittersweet** | more sweet/bitter volume | AGPL-3.0 | ⬜ to get | `git clone github.com/cosylabiiit/bittersweet` → `bittersweet/data/*.tsv`. AGPL: keep a CC-BY-clean build with `INCLUDE_COSYLAB=False`, or opt in deliberately. |
| **FlavorDB** | ~25k molecules taste+odor + natural-source mapping | confirm terms | ⬜ to get | REST/JSON API at `cosylab.iiitd.edu.in/flavordb` → `flavordb_taste.csv`. Map taste fields; **[VERIFY] columns** before trusting. |
| **UMP442 / BIOPEP-UWM** | more umami examples (only 283 now) | confirm terms | ⬜ to get | BIOPEP-UWM umami DB (form-driven) → `umami_list.csv`. |
| **SweetenersDB v2.0** | the sweetness-**intensity** regressor | **MIT** | ✅ in use | direct CSV from `github.com/chemosim-lab/SweetenersDB` (`SweetenersDB_v2.0.csv`, 316 cmpds, `logSw` column). R²≈0.82. |
| **Pyrfume / Leffingwell** | the **aroma model** (OpenPOM) — issues #17 / #18 | per-set; confirm | ⬜ to get | `git clone github.com/pyrfume/pyrfume-data`; Leffingwell odor set. Separate, version-fragile effort. |
| **FEMA GRAS / FDA SAF** | GRAS cross-reference + dosing/OAV lookups | gov public / FEMA | ⬜ to get | FDA "Substances Added to Food" (public domain) → `gras_reference.parquet`; FEMA use-level PDFs → `properties.parquet`. **Verify every scraped dosing number.** |

## Notes

- **Licensing.** ChemTastesDB (CC-BY-4.0) is the clean base. cosylab is AGPL — gated
  behind the `INCLUDE_COSYLAB` flag. Get an IP/OSS-license review before any
  commercial release; full attribution lives in [`SOURCES.md`](SOURCES.md).
- **Column verification.** Each new source's column names must be checked against the
  loaders (the `[VERIFY]` markers) — see issue **#25**. ChemTastesDB's mapping is
  resolved (both the coarse `Class taste` and the granular `Taste` columns are parsed).
- **Suggested priority.** (1) `Taste`-column mining — **done** ✅ → (2) SweetenersDB
  intensity head — **done** ✅ → (3) Pyrfume / Leffingwell for the aroma model →
  (4) cosylab for taste volume (AGPL — pending decision). **FlavorDB is
  CC BY-NC-SA (NonCommercial) — incompatible with a commercial product; skip.**
