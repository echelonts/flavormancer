# Data Requirements — what unlocks each capability, and the format we want it in

Flavormancer is built so that **capabilities light up as data arrives**. Some run today
on commercial-clean open data; others are *built but data-gated* — the code is in place
and switches on the moment a properly-licensed table is dropped in; others still are
**only** unlockable by a customer's own data. This document is the precise spec: for
each capability, **what data unlocks it, the format we want, and what we ask a client to
provide.** It doubles as the data-intake sheet for a pilot.

> **On-prem, always.** Anything a customer gives us is processed on hardware they own
> and **never leaves their building**. That's the core of the product, not a footnote.

---

## 0. Conventions (how to hand us any dataset)

- **One row per molecule.** A flat `CSV` (or `.parquet`) is ideal.
- **Identify the molecule** by any of, in order of preference:
  `smiles` (canonical preferred) · `inchikey` · `cas` · `name` (common or IUPAC — we
  resolve names/CAS via PubChem). At least one structural identifier (`smiles`/`inchikey`/`cas`)
  is strongly preferred; bare names are resolvable but lossier.
- **Units explicit, in the column name or a units row** (e.g. `threshold_ppm`, not `threshold`).
- **Licensing stated.** Tell us the source and its license. We only ingest
  commercial-clean data into the commercial edition (see [`SOURCES.md`](SOURCES.md));
  customer-owned data is governed by the pilot agreement.

---

## 1. Built & running on open data (no client data needed)

| Capability | Source | Notes |
|---|---|---|
| Taste (sweet/bitter/umami + sweetness intensity) | ChemTastesDB (CC-BY), SweetenersDB (MIT) | trained models, shipping |
| Physicochemical (logP/MW/TPSA/…) | RDKit | computed |
| **Measured boiling point / vapor pressure** | PubChem (public domain) — `build_properties.py` | run the builder to populate `properties.parquet` |
| **GRAS / food-ingredient cross-check** | FDA SAF (public domain) — `build_gras_reference.py` | run the builder to populate `gras_reference.parquet` |
| Stability / chemesthesis / dangerous-mixture screen | rules + documented food-safety literature | qualitative |
| Substitution search | the labeled set | Tanimoto today |

---

## 2. Built but data-gated — drop in the table and it switches on

These read pre-defined columns in `properties.parquet` (keyed by `inchikey`). The loader
already exists in `predict.py`; supply the data and the feature reports real numbers.

### 2a. Quantitative dosing — odor thresholds
- **Unlocks:** OAV (odor activity value = concentration ÷ detection threshold) goes from
  a qualitative ranking to **quantitative** "this component is N× over threshold."
- **Why gated:** the standard public threshold compilations (Devos/Oxford, ASTM DS48A,
  van Gemert) are **copyrighted books** — no clean bulk set. Customer-supplied or licensed.
- **Want:** `inchikey` (or `smiles`/`cas`) + `odor_threshold_ppm` (detection threshold).
  Optionally `matrix` (water/air/oil), `reference`.

### 2b. Quantitative dosing — FEMA / use levels
- **Unlocks:** overdose flagging against usual/maximum use levels.
- **Why gated:** FEMA usual/max levels live in copyrighted GRAS publications.
- **Want:** `inchikey` + `fema_use_max_ppm` (and `fema_use_usual_ppm` if available),
  optionally per food category.

### 2c. GC-MS Kovats retention index
- **Unlocks:** RI lookup/prediction in the bench/analytical workflow.
- **Why gated:** **not in PubChem** (we checked); the NIST GC/RI library is **paid**
  (~$595 data-only FTP to ~$2,850 single-seat). Customer RI data, a NIST license, or a
  QSPR trained on a vetted open set.
- **Want:** `inchikey` + `kovats_ri` + `column_phase` (e.g. non-polar DB-5 / polar
  wax) + `temp_program` if available (RI depends on the column).

---

## 3. Aroma — odor-descriptor prediction (intensity is the "comes with your data")

Aroma **ships in the commercial edition as presence/absence** — 13 RandomForest descriptor
heads (citrus, floral, minty, almond, …) trained on **public-domain** PubChem/HSDB odor text
(CV-AUROC 0.80–0.96; see [`AROMA.md`](AROMA.md)). What it does **not** have is **scored
intensity** — *how strong* each note is — because **no public-domain intensity-scored odor
data exists** (the one clean scored set, keller_2016, is naive-subject noise; every expert
intensity set — GS-LF, Dravnieks — is NonCommercial/proprietary). So the marquee upgrade is
**intensity**, trained **for a customer on their own panel data**, on-prem, or on a
**commercially-licensed** set (Leffingwell **PMP 2001**). Richer/rarer descriptors come with
that data too.

**What we need to train an aroma model:**
- **Molecules:** `smiles` — or **GC-MS** output identifying the volatile compounds in your
  products (we resolve those to structures). *GC-MS alone is not enough on its own.*
- **Labels — the part only you have:** each molecule's **expert odor descriptors** from
  your sensory panel (e.g. `green; fruity; woody`), ideally with **intensity** (0–100 or
  a scale you define) and a **consistent vocabulary** (a fixed descriptor list beats free
  text).
- **Format:** `smiles, descriptors, intensity` — one row per molecule (descriptors a
  `;`-separated list or one column per descriptor). A few hundred well-labeled molecules
  trains a modest model; thousands (PMP 2001 scale) earns a GNN (OpenPOM).
- **Why both:** GC-MS identifies *which* molecules; the sensory labels are *what the model
  learns to predict*. Structure → smell needs the smell side, and that's your expertise.

---

## 4. Only your data can unlock these (fundamental, not effort)

Public data cannot produce these at all — they're the heart of a paid pilot:

| Capability | What we need from you | Format |
|---|---|---|
| **Finished-blend perception** (how a *formulation* tastes, with masking/synergy) | your **formulation → sensory-outcome** archive | per-formulation: components + concentrations + panel scores |
| **Consumer liking / preference** | sensory-panel + consumer datasets | product → ratings |
| **Mouthfeel / body / carbonation in a real matrix** | matrix + process data | formulation + process + outcome |
| **Accuracy on *your* products** | your formulation archive (e.g. an 85k-formulation library) | as above |
| **Shelf-life / stability in your matrix** | stability-test data (pH/temp/light/packaging over time) | timepoint measurements |

---

## 5. The loader schemas (so client data maps exactly)

| File (drop in the training/run dir) | Columns the code reads |
|---|---|
| `properties.parquet` | `inchikey` + any of `odor_threshold_ppm`, `fema_use_max_ppm`, `boiling_point_c`, `vapor_pressure_pa` |
| `gras_reference.parquet` | `inchikey` (presence = recognized food substance) |
| taste sources | `smiles` + taste label(s) — see `build_taste_dataset.py` |
| aroma training | `smiles` + odor `descriptors` (+ `intensity`) |

Everything keyed on `inchikey`/`smiles` so we can join your data to a structure cleanly.
Hand us a spreadsheet in roughly these shapes and the matching capability comes alive —
on your hardware, with your data staying yours.
