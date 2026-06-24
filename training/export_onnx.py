"""
export_onnx.py — taste models (scikit-learn) -> ONNX, for in-process serving in .NET.

The taste RandomForests + sweetness-intensity regressor train in Python but SHIP in
.NET: this exports each to ONNX so ASP.NET Core can run them via ONNX Runtime with
zero Python at runtime. Verified roundtrip: ONNX reproduces sklearn probabilities to
~1e-7. Run after train_taste.py.

Output:
  onnx_models/<name>.onnx        one per taste head (+ sweet_intensity regressor)
  onnx_models/manifest.json      {input_name, n_features, fp_radius, models{...}}
                                 the .NET side reads this so nothing is guessed.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

SRC = Path("taste_models")     # produced by train_taste.py
OUT = Path("onnx_models")
OUT.mkdir(exist_ok=True)
FP_BITS, FP_RADIUS, INPUT_NAME = 2048, 2, "fp"

if not SRC.exists() or not list(SRC.glob("*_rf.joblib")):
    raise SystemExit("No taste_models/*.joblib found — run train_taste.py first.")

manifest = {"input_name": INPUT_NAME, "n_features": FP_BITS, "fp_radius": FP_RADIUS, "models": {}}

for path in sorted(SRC.glob("*_rf.joblib")):
    name = path.stem.replace("_rf", "")
    model = joblib.load(path)
    is_regressor = (name == "sweet_intensity")

    initial = [(INPUT_NAME, FloatTensorType([None, FP_BITS]))]
    opts = {} if is_regressor else {id(model): {"zipmap": False}}  # zipmap off -> plain arrays
    onx = to_onnx(model, initial_types=initial, options=opts, target_opset=17)
    out_path = OUT / f"{name}.onnx"
    out_path.write_bytes(onx.SerializeToString())

    # ---- self-validate: ONNX must match sklearn on random fingerprints ----
    X = (np.random.rand(25, FP_BITS) > 0.97).astype(np.float32)
    sess = ort.InferenceSession(str(out_path))
    onnx_out = sess.run(None, {INPUT_NAME: X})
    if is_regressor:
        ref = model.predict(X).ravel()
        got = np.array(onnx_out[0]).ravel()
    else:
        ref = model.predict_proba(X)[:, 1]
        got = np.array([row[1] for row in onnx_out[1]])  # output[1] = probabilities
    max_diff = float(np.max(np.abs(ref - got)))
    assert max_diff < 1e-4, f"{name}: ONNX/sklearn mismatch {max_diff}"

    manifest["models"][name] = {
        "type": "regressor" if is_regressor else "classifier",
        "onnx": out_path.name,
        "prob_output_index": None if is_regressor else 1,
        "max_validation_diff": max_diff,
    }
    print(f"{name:16} -> {out_path.name}  (validated, max diff {max_diff:.2e})")

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
print(f"\nwrote {OUT}/manifest.json — input '{INPUT_NAME}', {FP_BITS}-bit Morgan r{FP_RADIUS}")
print("The .NET TasteModelService reads this manifest; keep FP bits/radius in sync with predict.py.")
