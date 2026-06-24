# training/

Python, build-time only. Builds the merged taste dataset, trains the taste and
sweetness-intensity models, and exports them to ONNX for the .NET app to serve.

- Dataset assembly (merge + dedup by InChIKey across open taste sources)
- Taste model training (RDKit features → scikit-learn classifiers)
- Sweetness-intensity regressor
- ONNX export (`skl2onnx`)

Datasets and trained artifacts are **not** committed — scripts download sources;
`.gitignore` excludes the outputs.
