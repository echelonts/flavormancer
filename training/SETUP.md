# Flavormancer — Flavor-Prediction Training & Demo Setup (R620 / single box)

> **Scope:** this is the **Python training + demo track** (Track A in
> `ARCHITECTURE.md`) — the model pipeline and the quick FastAPI/HTML workbench,
> run on the R620. The **product** app the team builds is **.NET/React** and
> lives in `/api` + `/frontend`; it consumes the ONNX models this training
> pipeline produces. Train here, ship there.

Target box: Dell R620, Linux, **no GPU** → CPU training. Install the CPU
PyTorch build. Everything below assumes a fresh user/env so it stays
isolated and reproducible.

## 0. Prereqs
- Miniforge/Mambaforge (mamba resolves the finicky deepchem stack far faster
  than conda). https://github.com/conda-forge/miniforge
- git

## 1. Create an isolated user + workspace
```bash
sudo adduser flavordemo          # optional but clean
sudo su - flavordemo
mkdir -p ~/odor-demo && cd ~/odor-demo
```

## 2. Conda env
OpenPOM is built on DeepChem and is **version-fussy** — this is the part most
likely to eat an afternoon. Pin against OpenPOM's own requirements, don't let
it resolve against "latest". Start here, then reconcile with the repo:

```bash
mamba create -n odor python=3.10 -y
mamba activate odor

# CPU PyTorch (NO cuda wheels — this box has no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

mamba install -c conda-forge rdkit pandas scikit-learn numpy openpyxl pyarrow -y
# openpyxl: read the ChemTastesDB .xlsx  |  pyarrow: read/write the .parquet files
pip install deepchem
pip install dgl dgllife            # OpenPOM's GNN backend
pip install pubchempy              # name -> SMILES, for the demo UX
pip install umap-learn             # for the odor-space map later
```

## 3. Get OpenPOM + data
```bash
git clone https://github.com/BioMachineLearning/openpom.git
pip install -e ./openpom
# Reconcile any version conflicts NOW using openpom/requirements — this is the
# expected friction point. If torch/dgl/deepchem fight, match openpom's pins.

# Leffingwell odor dataset (SMILES + multilabel descriptors) — AROMA head
git clone --depth 1 https://github.com/pyrfume/pyrfume-data.git
# dataset lives under pyrfume-data/leffingwell/

# Taste data — TASTE heads (sweet, bitter, umami) + sweetness intensity.
# 1) ChemTastesDB v2.0 — PRIMARY source, CC-BY-4.0, 4075 molecules, 10 classes:
curl -L -o ChemTastesDB_database.xlsx \
  "https://zenodo.org/records/14963136/files/ChemTastesDB_database.xlsx?download=1"
# 2) cosylab/bittersweet — extra sweet/bitter (AGPL; optional, ignore their py2.7 code):
git clone --depth 1 https://github.com/cosylabiiit/bittersweet.git
# 3) SweetenersDB (Cheron 2017) — sweetness INTENSITY regressor. [OBTAIN]
#    Pull the ~316-compound table (SMILES + relative-to-sucrose sweetness) from the
#    paper's supplementary; save as sweeteners_db.csv. Optional; intensity head
#    is skipped cleanly if absent.
# 4) (optional) more sources the build script will auto-merge IF present:
#      flavordb_taste.csv  — FlavorDB export: columns SMILES + taste   [OBTAIN]
#      umami_list.csv      — UMP442 / BIOPEP-UWM umami SMILES           [OBTAIN]
#    Each is optional and skipped cleanly if the file isn't there.
```

## 4. Sanity checks (do these before training anything)
```bash
python - <<'PY'
import torch, deepchem, rdkit
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())  # expect False on R620
from rdkit import Chem
print("rdkit ok:", Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1")))
PY
```
`cuda? False` is correct and expected here — it'll train on CPU.

## 5. Build the merged taste dataset, then train
```bash
python build_taste_dataset.py   # merges all sources -> taste_master.parquet
python train_odor.py            # AROMA head (OpenPOM) — multi-hour / overnight CPU
python train_taste.py           # TASTE heads (sklearn) — minutes, even merged
```
`build_taste_dataset.py` writes `taste_master.parquet` (multi-label: sweet/
bitter/umami/sour/salty) and, if SweetenersDB is present, `sweet_intensity.parquet`.

`train_taste.py` trains one head per taste that clears the data threshold
(sweet/bitter/umami train; sour/salty auto-skip -> rule), plus a sweetness-
intensity regressor. Prints AUROC / R2 you can quote. Saves to `taste_models/`.

`train_odor.py` writes `odor_model/`, `embeddings.parquet`, `metrics.json`.

## 6. One molecule's full flavor read (taste runs today; aroma once wired)
```bash
python predict.py "OC(=O)CC(O)(CC(=O)O)C(=O)O"   # citric acid → sour=True, low sweet/bitter
```
Returns every taste head present + sweetness intensity + the sour flag. This is
exactly what the workbench screen calls per molecule.

## Notes
- The hard part is the install pins, not the training. Budget for it.
- Keep raw data + checkpoints small; whole project should sit well under 40GB.
- When you move to serving: the model file goes into a light CPU FastAPI
  container; CUDA never has to be containerized because inference is CPU work.


## Aroma model (OpenPOM) — the smell half
The aroma GNN is the one version-fussy install. On the R620 (CPU):

1. `pip install deepchem openpom` (pin compatible torch/rdkit/deepchem; see OpenPOM's README).
2. Get the training data: the OpenPOM repo's curated `curated_GS_LF_merged_4983.csv`
   (easiest), or build from `pyrfume-data/leffingwell`.
3. `python train_odor.py`  -> writes `./odor_model/` (model + tasks.json + metrics.json)
   and `odor_embeddings.parquet` (for the odor map + substitution search). Overnight on CPU.
4. `predict.py` auto-loads `./odor_model/` when present; until then `predict_aroma()`
   returns an honest 'not trained yet' instead of fabricating smells.
5. Product (.NET): try ONNX-exporting the GNN; if it won't export cleanly, run it behind
   the thin Python aroma sidecar (the architecture's documented fallback).
