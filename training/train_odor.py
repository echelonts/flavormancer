"""
train_odor.py — structure -> odor-descriptor model (OpenPOM / principal odor map)

This is the AROMA half of Flavormancer: a graph neural network that predicts odor
descriptors ("floral", "green", "citrus", ...) from molecular structure. Unlike
taste, odor does NOT track simple fingerprints (near-identical molecules can smell
unrelated), so this uses OpenPOM's message-passing GNN — the open reimplementation
of the Lee et al. 2023 "principal odor map" (Science). See SOURCES.md for credit.

HONESTY NOTES
- Runs on the R620, CPU only (no CUDA). Training is an overnight-ish job.
- DeepChem + OpenPOM are Python-only and version-fussy; this is the one piece that
  stays Python at runtime (ONNX export of GNNs is unreliable -> aroma sidecar).
- The OpenPOM constructor exposes many architecture hyperparameters. The block
  below mirrors the OpenPOM README example; if your installed version renames an
  argument, reconcile against that version's example notebook. The overall flow —
  load -> featurize -> multitask GNN -> score -> save -> embeddings — is stable.
- This script was authored against OpenPOM's documented API and is syntax-checked,
  but the actual training run happens on the R620 where DeepChem is installed; it
  has not been executed in the planning sandbox.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

OUT = Path("odor_model")          # trained model + tasks land here; predict.py reads it
OUT.mkdir(exist_ok=True)
EMB_OUT = Path("odor_embeddings.parquet")  # powers the odor-space map + substitution search

# ---------------------------------------------------------------------------
# 1. Data. Easiest path: OpenPOM ships a curated, deduped dataset
#    (curated_GS_LF_merged_4983.csv, ~138 odor descriptors). Point DATA_CSV at it.
#    Alternative: build your own from pyrfume-data/leffingwell (see SOURCES.md).
# ---------------------------------------------------------------------------
DATA_CSV = "curated_GS_LF_merged_4983.csv"   # from the OpenPOM repo's datasets/
SMILES_FIELD = "nonStereoSMILES"             # OpenPOM's curated-dataset SMILES column

_df = pd.read_csv(DATA_CSV)
TASKS = [c for c in _df.columns if c not in (SMILES_FIELD, "descriptors")]
print(f"{len(_df)} molecules, {len(TASKS)} odor descriptors")

# ---------------------------------------------------------------------------
# 2. Featurize with the OpenPOM graph featurizer + DeepChem CSV loader
# ---------------------------------------------------------------------------
featurizer = GraphFeaturizer()
loader = dc.data.CSVLoader(tasks=TASKS, feature_field=SMILES_FIELD, featurizer=featurizer)
dataset = loader.create_dataset(DATA_CSV)

splitter = dc.splits.RandomStratifiedSplitter()
train_ds, test_ds = splitter.train_test_split(dataset, frac_train=0.85, seed=42)

# class imbalance ratio per task (odor labels are sparse) — OpenPOM uses this
train_ratios = []
for j in range(len(TASKS)):
    col = train_ds.y[:, j]
    pos = max(int(col.sum()), 1)
    train_ratios.append(float((len(col) - pos) / pos))

# ---------------------------------------------------------------------------
# 3. Train the message-passing GNN (CPU). Hyperparameters mirror the OpenPOM
#    example; reconcile names with your installed version if needed.
# ---------------------------------------------------------------------------
model = MPNNPOMModel(
    n_tasks=len(TASKS),
    batch_size=128,
    learning_rate=1e-3,
    class_imbalance_ratio=train_ratios,
    loss_aggr_type="sum",
    node_out_feats=100,
    edge_hidden_feats=75,
    edge_out_feats=100,
    num_step_message_passing=5,
    mpnn_residual=True,
    message_aggregator_type="sum",
    mode="classification",
    number_atom_features=GraphConvConstants.ATOM_FDIM,
    number_bond_features=GraphConvConstants.BOND_FDIM,
    n_classes=1,
    nb_layers=2,
    nb_timesteps=2,
    self_loop=False,
    model_dir=str(OUT),
    device="cpu",           # R620 has no GPU
)

NB_EPOCH = 50
model.fit(train_ds, nb_epoch=NB_EPOCH)
model.save_checkpoint(model_dir=str(OUT))
json.dump(TASKS, open(OUT / "tasks.json", "w"))
print(f"saved model + {len(TASKS)} tasks to {OUT}/")

# ---------------------------------------------------------------------------
# 4. Score — per-descriptor AUROC (the numbers you quote in the pitch)
# ---------------------------------------------------------------------------
y_pred = np.array(model.predict(test_ds))
# normalize to [n_samples, n_tasks] positive-class scores
if y_pred.ndim == 3:
    y_pred = y_pred[:, :, -1]
aurocs = {}
for j, label in enumerate(TASKS):
    yt = test_ds.y[:, j]
    if yt.sum() == 0 or yt.sum() == len(yt):
        continue
    try:
        aurocs[label] = float(roc_auc_score(yt, y_pred[:, j]))
    except ValueError:
        continue
json.dump(aurocs, open(OUT / "metrics.json", "w"), indent=2)
if aurocs:
    print(f"mean AUROC over {len(aurocs)} scorable descriptors: {np.mean(list(aurocs.values())):.3f}")

# ---------------------------------------------------------------------------
# 5. Embeddings -> pgvector substitution search + UMAP odor map
#    OpenPOM exposes a learned-embedding method; name may vary by version.
# ---------------------------------------------------------------------------
try:
    emb = model.predict_embedding(dataset)        # [VERIFY] method name in your version
    pd.DataFrame(np.array(emb)).assign(smiles=_df[SMILES_FIELD].values).to_parquet(EMB_OUT)
    print(f"saved odor embeddings -> {EMB_OUT}")
except Exception as e:  # noqa: BLE001
    print(f"(embedding export skipped — confirm the embedding method name: {e})")
