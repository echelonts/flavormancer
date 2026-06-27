# Flavormancer — Flavor-Prediction Training & Demo Setup (R620 / single box)

> **Scope:** this is the **Python training + demo track** (Track A in
> `ARCHITECTURE.md`) — the model pipeline and the quick FastAPI/HTML workbench,
> run on the R620. The **product** app the team builds is **.NET/React** and
> lives in `/api` + `/frontend`; it consumes the ONNX models this training
> pipeline produces. Train here, ship there.

Target box: Dell R620, Linux, **no GPU** → CPU only. The taste + demo stack is
light (RDKit, scikit-learn, skl2onnx) and installs cleanly — no version-fussy
deep-learning dependencies. Everything below assumes a fresh, isolated env.

## 0. Prereqs
- Python 3.12 + venv, **or** Miniforge/Mambaforge — either works for this stack.
  https://github.com/conda-forge/miniforge
- git

## 1. Create an isolated workspace
```bash
mkdir -p ~/flavormancer-train && cd ~/flavormancer-train
python3 -m venv .venv && source .venv/bin/activate
# (or: mamba create -n flavor python=3.12 -y && mamba activate flavor)
```

## 2. Install the stack (light — no GPU, no DeepChem)
```bash
pip install rdkit pandas scikit-learn numpy openpyxl pyarrow
# openpyxl: read the ChemTastesDB .xlsx  |  pyarrow: read/write the .parquet files
pip install skl2onnx onnxruntime    # export + self-validate the taste ONNX models
pip install pubchempy               # name -> SMILES, for the demo UX
pip install umap-learn              # for the flavor-space map
```

> **Aroma is deferred** (see [`docs/AROMA.md`](../docs/AROMA.md)) — no commercially
> usable public odor data is good enough to train it. The heavy OpenPOM/DeepChem GNN
> stack is therefore **not installed here**. When licensed (PMP 2001) or customer
> data exists, add it then; the aroma pipeline (`build_aroma_dataset.py` +
> `train_aroma.py`) is committed and ready to run.

## 3. Get the taste data
```bash
# ChemTastesDB v2.0 — PRIMARY source, CC-BY-4.0, ~4075 molecules, multi-class taste:
curl -L -o ChemTastesDB_database.xlsx \
  "https://zenodo.org/records/14963136/files/ChemTastesDB_database.xlsx?download=1"

# SweetenersDB v2.0 — sweetness INTENSITY regressor. MIT, from the authors' own lab:
git clone --depth 1 https://github.com/chemosim-lab/SweetenersDB.git
# relative-to-sucrose sweetness (SMILES + logS); the build script reads it if present.
```
> Other taste sources are auto-merged **only if present**, but note the licensing
> decisions in [`docs/SOURCES.md`](../docs/SOURCES.md): cosylab/bittersweet is AGPL
> and FlavorDB is NonCommercial — both are **off by default and not used**.

## 4. Sanity check
```bash
python - <<'PY'
import rdkit, sklearn, skl2onnx
from rdkit import Chem
print("rdkit ok:", Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1")))
PY
```

## 5. Build the dataset, then train
```bash
python build_taste_dataset.py   # merges sources -> taste_master.parquet (+ sweet_intensity.parquet)
python train_taste.py           # taste heads (sklearn) — minutes, even merged
python export_onnx.py           # export taste models to ONNX (+ roundtrip self-validation)
```
`train_taste.py` trains one head per taste that clears the data threshold
(sweet/bitter/umami train; sour also gets a small-data *indicative* head; salty
stays a validated rule), plus a sweetness-intensity regressor. Prints AUROC / R²
you can quote. Saves to `taste_models/`.

## 6. One molecule's full flavor read (taste runs today)
```bash
python predict.py "OC(=O)CC(O)(CC(=O)O)C(=O)O"   # citric acid → sour=True, low sweet/bitter
```
Returns every taste head present + sweetness intensity + sour/salty flags + the
physicochemical / stability / chemesthesis / safety packs. `substitute()` provides
the nearest-neighbor substitution search. This is what the workbench screen calls.

## 7. Run the demo workbench
```bash
pip install fastapi "uvicorn[standard]"
uvicorn app:app --host 0.0.0.0 --port 8000   # then open http://<r620-ip>:8000/
```

## Notes
- The taste/demo stack is light and CPU-only — no CUDA, no DeepChem.
- Keep raw data + model artifacts small; the whole project sits well under a few GB.
- Aroma, when it comes, trains on licensed/customer data and exports to ONNX
  (RandomForest) or runs behind a thin Python sidecar (OpenPOM GNN) — built then,
  not now. See `docs/AROMA.md`.
