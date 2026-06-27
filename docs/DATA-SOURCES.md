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
| **PlantMolecularTasteDB** | natural-product / phytochemical tastants (1,527 cmpds) | **article CC-BY, but DATABASE unlicensed** | ❌ **excluded (on diligence)** | Gradinaru et al., *Front. Pharmacol.* `10.3389/fphar.2021.751712`. The **article** is CC-BY, but the **database** is web-only (plantmoleculartastedb.org), carries **no open data license**, and the authors registered an **i-Depot** (IP/authorship claim, cert. 122867). Article-CC-BY ≠ database-CC-BY — would need the authors' explicit commercial grant. **Not used.** |
| **FartDB** | largest public tastant set (~31k molecules) | **MIT badge — composite NOT clean** | ❌ **excluded** | HuggingFace `FartLabs/FartDB` is labeled MIT, but it composites **FlavorDB (CC-BY-NC-SA)** + **SciFinder (CAS, proprietary)** + Tas2R-DB (ACS). The MIT badge can't relicense data FartLabs doesn't own — the **OpenPOM GS-LF trap again**. NonCommercial/proprietary upstream → not usable commercially. |
| **cosylab/bittersweet** | more sweet/bitter volume | AGPL-3.0 | ⬜ to get | `git clone github.com/cosylabiiit/bittersweet` → `bittersweet/data/*.tsv`. AGPL: keep a CC-BY-clean build with `INCLUDE_COSYLAB=False`, or opt in deliberately. |
| **FlavorDB** | ~25k molecules taste+odor + natural-source mapping | **CC BY-NC-SA 3.0** | ❌ skip | NonCommercial — incompatible with a commercial product. |
| **UMP442 / BIOPEP-UWM** | more umami examples | none / unclear | ❌ skip | BIOPEP-UWM is web-only; the `Shoombuatong/Dataset-Code` repost has **no LICENSE** (all-rights-reserved) and is umami *peptide* data (different class from our small-molecule head). |
| **SweetenersDB v2.0** | the sweetness-**intensity** regressor | **MIT** | ✅ in use | direct CSV from `github.com/chemosim-lab/SweetenersDB` (`SweetenersDB_v2.0.csv`, 316 cmpds, `logSw` column). R²≈0.82. |
| **Pyrfume — Leffingwell/GoodScents (GS-LF)** | the **aroma model** (OpenPOM) — #17/#18 | **RESTRICTED** | ❌ **excluded** | Use restrictions (John Leffingwell & Google; GoodScents/Arctander/Flavornet © Datu Inc.). **Not used at all, not even for the demo** — it may go to a customer / a commercial soda venture, so we avoid the liability entirely. |
| **Pyrfume — open academic sets** | aroma model, **clean but small** | CC-BY / open-access (verify each) | candidates | `keller_2016` (BMC, CC-BY, ~480 cmpds), `snitz_2013` (PLOS, CC-BY), etc. Confirm each data deposit's license; combine + harmonize descriptor vocabularies for volume. |
| **FEMA GRAS / FDA SAF** | GRAS cross-reference + dosing/OAV lookups | gov public / FEMA | ⬜ to get | FDA "Substances Added to Food" (public domain) → `gras_reference.parquet`; FEMA use-level PDFs → `properties.parquet`. **Verify every scraped dosing number.** |

## Notes

- **Licensing.** ChemTastesDB (CC-BY-4.0) is the clean base. cosylab is AGPL — gated
  behind the `INCLUDE_COSYLAB` flag. Get an IP/OSS-license review before any
  commercial release; full attribution lives in [`SOURCES.md`](SOURCES.md).
- **Column verification.** Each new source's column names must be checked against the
  loaders (the `[VERIFY]` markers) — see issue **#25**. ChemTastesDB's mapping is
  resolved (both the coarse `Class taste` and the granular `Taste` columns are parsed).
- **Suggested priority.** (1) `Taste`-column mining — **done** ✅ → (2) SweetenersDB
  intensity head — **done** ✅ → (3) aroma model (see the data caveat below) →
  (4) cosylab — **skip** (AGPL). **No additional commercial-clean taste source is
  currently available** — the three richest candidates all fail the same "downloadable
  ≠ usable" test: FlavorDB is CC-BY-NC-SA (NonCommercial); FartDB (~31k) is an MIT badge
  over an NC/proprietary composite (FlavorDB + SciFinder); PlantMolecularTasteDB is
  article-CC-BY but its database is i-Depot-registered and unlicensed. Taste stays on
  ChemTastesDB + SweetenersDB until a genuinely clean source — or **customer data** —
  appears.
- **Aroma data — COMMERCIAL-CLEAN ONLY (decision).** Every *rich* odor-descriptor set
  (Leffingwell, GoodScents, Arctander, Flavornet © Datu Inc., FlavorDB / FooDB NC) is
  restricted or NonCommercial — **we do NOT use any of them, not even for the demo**,
  because the demo may be handed to a customer or used in a commercial soda venture.
  Build the aroma model **only on truly-open (CC-BY / public-domain) odor sets** —
  candidates from open-access journals: `keller_2016` (BMC, CC-BY, ~480), `snitz_2013`
  (PLOS, CC-BY), etc. (verify each deposit's license). These are *small*, so the public
  aroma model will be modest — combine + harmonize them for volume, or **defer the
  aroma model until enough clean data (or the customer's own) exists.** OpenPOM *code*
  is MIT and stays usable.
- **Pyrfume's repo MIT covers code/curation only** — each dataset keeps its *own*
  license (Pyrfume: *"data provided as-is… licensing per dataset… takedown requests to
  admin@pyrfume.org"*). **Downloadable ≠ usable.**
- **Affordable commercial path for *rich* aroma.** The Zenodo Leffingwell set (3,523
  odor-descriptor molecules, `zenodo.org/records/4085098`) is **CC-BY-NC** (research-only
  → not for us), but it derives from Leffingwell's **PMP 2001** database, which **is
  commercially licensable** (~$2,775 historical, 2 workstations, from Leffingwell &
  Associates). License PMP 2001 → re-curate the descriptors from *your licensed copy* →
  a strong, commercially-clean aroma model. Or have a client license it and load it
  on-prem (the data liability stays with them).
