# Data pipeline — sources, exact columns, and a clean-machine first run

What every build script reads, the **real** column names in each table (not the ones the code
hopes for), and the order to run things in from nothing. Companion to
[`training/SETUP.md`](../training/SETUP.md) (install steps) and
[`SOURCES.md`](SOURCES.md) (licensing and attribution).

Written because the loaders carried `[VERIFY]` markers where nobody had opened the file and
confirmed what was actually in it.

---

## 1. First run from a clean machine

Assumes Python 3.12 and ~8 GB free. The whole chain is CPU-only; no GPU anywhere.

```bash
git clone https://github.com/echelonts/flavormancer.git && cd flavormancer
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" pydantic umap-learn openpyxl   # serving + map + xlsx
```

Then, **in this order** — each step consumes the previous step's output:

| # | command | produces | ~time |
|---|---|---|---|
| 1 | `python build_taste_dataset.py` | `taste_master.parquet` | 1 min |
| 2 | `python train_taste.py` | `taste_models/` + manifest | 3 min |
| 3 | `python build_odor_notes.py --pubchem-all` | `odor_notes.parquet` | **hours** (crawl) |
| 4 | `python build_aroma_supplement.py` | `aroma_supplement.csv` | 10 min (crawl) |
| 5 | `python build_aroma_dataset.py` | `aroma_train.parquet` | 2 min |
| 6 | `python train_aroma.py` | `aroma_models/` + manifest | 10 min (32 cores) |
| 7 | `python build_mouthfeel_supplement.py && python build_mouthfeel_dataset.py && python train_mouthfeel.py` | `mouthfeel_models/` | 5 min |
| 8 | `python train_tox.py` | `tox_models/` | 5 min |
| 9 | `python build_enrichment.py` | `master_enrichment.parquet` | 8 min |
| 10 | `python build_profile_index.py` | `profile_index.npz` | 4 min |
| 11 | `python build_flavor_map.py` | `flavor_map.parquet` | 6 min |
| 12 | `uvicorn app:app --host 0.0.0.0 --port 8000` | the workbench | 50 s to warm |

**Steps 3 and 4 are network crawls against PubChem.** They cache, so a re-run is cheap, but a
cold step 3 is genuinely hours — start it and go and do something else. Everything downstream is
local compute.

**Verify it worked** (this is the honest end-to-end check, not just "the server started"):

```bash
curl -s localhost:8000/api/status                      # {"ready":true,"total":190,...}  (190 artifacts = 189 heads + the intensity regressor)
curl -s localhost:8000/api/heads | jq '.aroma | length' # 166
# vanillin — the vanilla head should fire at 1.0, confident, with its calibrated threshold
curl -s -X POST localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"COc1cc(C=O)ccc1O"}' \
  | jq '.aroma.descriptors[] | select(.odor=="vanilla")'
# {"odor":"vanilla","score":1.0,"threshold":0.42,"confident":true,"precision":1.0,"auroc":0.93,...}
```

`.aroma.top` is the *confident* subset ordered by score, so `top[0]` is whichever head scored
highest — not necessarily the one you are looking for. Query `descriptors[]` by name, as above.

If `/api/status` reports fewer than 195 heads, a training step was skipped — the app degrades
gracefully rather than failing, so it will start regardless.

---

## 2. What each table actually contains

Column names below were read off the built artifacts, not copied from the loader's expectations.
Where a loader accepts several spellings it is noted, because the upstream files change.

### Taste

| table | rows | key columns |
|---|---|---|
| `taste_master.parquet` | 3,845 | `inchikey`, `smiles`, `multitaste`, `sweet`, `bitter`, `umami`, `sour`, `salty`, `n_sources` |
| `taste_notes.parquet` | 676 | `inchikey`, `smiles`, `name`, `taste`, `taste_source` |

`taste_master` is one row per molecule with a 0/1 column per basic taste; `multitaste` marks
molecules ChemTastesDB records under more than one class. **There is no `tasteless` column** —
that head trains from documented-tasteless text in `taste_notes`, which is why
`build_flavor_map.py` overlays it separately rather than reading a column. `n_sources` counts the distinct upstream datasets a molecule appeared in. It is **provenance
metadata only** — nothing downstream reads it today.

Worth knowing how conflicts actually resolve, because it is not a vote: `build_taste_dataset`
aggregates with **positive-wins** — if any source labels a molecule sweet, it is sweet; only if
every source says 0 does it become 0. That maximises recall on sparse, partially-annotated
sources, which is the right default here, but it does mean one mislabelled source can assert a
taste the other four deny. `n_sources` is the column that would let a future weighted rule
override that, which is why it is recorded even though it is unused.


### Aroma

| table | rows | key columns |
|---|---|---|
| `odor_notes.parquet` | 2,258 | `inchikey`, `smiles`, `name`, `odor`, `odor_source`, `odor_threshold_ppm`, `odor_threshold_note` |
| `aroma_supplement.csv` | 2,159 | `flavor`, `molecule`, `smiles`, `category` |
| `aroma_train.parquet` | 2,431 | `inchikey`, `smiles`, + one 0/1 column per descriptor |

`odor` is **free text** as recorded ("sweet, floral, slightly minty"), normalised into the
controlled vocabulary by `build_aroma_dataset.tag()`. `odor_threshold_ppm` is populated only where
PubChem cites one — treat it as a lookup, never a prediction.

### Properties

`properties.parquet` — 9,154 rows: `inchikey`, `common_name`, `iupac_name`, `boiling_point_c`,
`boiling_point_pressure_mmhg`, `vapor_pressure_pa`, `melting_point_c`.

**`boiling_point_pressure_mmhg` is the column that makes the BP usable.** A boiling point measured
at reduced pressure is not comparable to one at atmospheric, and averaging them together would
corrupt any volatility ordering. `build_measured_properties.py` therefore *discards* reduced-pressure
values outright rather than storing them without their pressure.

Backfill tables, both keyed by **`inchikey_skel`** rather than `inchikey`:

- `iupac_backfill.parquet` — `common_name`, `iupac_name` from PubChem **Title + IUPACName**
- `measured_properties.parquet` — `boiling_point_c`, `melting_point_c` from PUG-View

> That key difference is a real trap. `build_enrichment._by_skel()` accepts either column for
> exactly this reason — an `inchikey`-only reader returns an empty dict for these two, silently,
> and every backfilled value simply never appears.

### Food-use registers

| table | rows | key columns |
|---|---|---|
| `saf_universe.parquet` | 2,782 | `smiles`, `inchikey`, `name`, `cas` (FDA Substances Added to Food) |
| `gb_union_list.parquet` | 2,164 | `smiles`, `inchikey`, `name`, `fl` (EU/GB flavourings, FL number) |
| `gras_reference.parquet` | 2,781 | `inchikey` |

These carry **listing status only**. A row here means a regulator lists the substance — it is not
a safety clearance, and the UI must never render it as one. See
[`food-safety provenance`](SOURCES.md).

### Master table

`master_enrichment.parquet` — 8,869 rows, the single table the app reads for everything that is
not a live prediction: `inchikey_skel`, `smiles`, `name`, `mw`, `logp`, `tpsa`, `hbd`, `hba`,
`rot_bonds`, `rings`, `melting_point_c`, `boiling_point_c`, `gras`, `taste_documented`, plus a
`tox_*` column per Tox21 assay.

`name` is **never null**: it falls back to the molecular formula, with multi-component structures
labelled as mixtures (`CaI2O6 (mixture: Ca+2 + IO3- + IO3-)`). A raw SMILES in the grid looks
broken; a formula is honest.

---

## 3. Unresolved source markers

`build_taste_dataset.py` carries `[OBTAIN]` markers for three sources. Their status, so nobody
re-investigates:

| source | status |
|---|---|
| `flavordb_taste.csv` | **Excluded, permanently.** FlavorDB is NonCommercial — incompatible with this project's commercial-clean requirement. The loader tolerates its absence. |
| `umami_list.csv` | Optional. UMP442/BIOPEP-UWM umami SMILES. Not currently used; the umami head trains from ChemTastesDB alone and reaches AUROC 0.990, so this is a nice-to-have. |
| `sweeteners_db.csv` | **Present and used.** Cheron SweetenersDB (MIT) drives the sweetness-intensity regressor (R² 0.836, n=316 — small, treat as indicative). |

The `[VERIFY]` marker at `build_taste_dataset.py:72` is a defensive `KeyError` for when an upstream
file renames a column. It is working as intended: the loader tries several known spellings and
fails loudly with the actual column list rather than silently producing an empty dataset.
