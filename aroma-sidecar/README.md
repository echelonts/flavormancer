# aroma-sidecar/

Python, runtime — **only if needed**. A thin localhost service that runs aroma
(odor-descriptor) inference when the GNN model can't be exported to ONNX cleanly
for in-process serving in .NET.

Best case this directory stays empty: the aroma model exports to ONNX and the
.NET app serves it directly, leaving zero Python at runtime.
