# api/

ASP.NET Core (C#) — the application that ships. Loads the exported ONNX models and
serves predictions in-process via ONNX Runtime, with no Python at runtime.

- `/predict` and related JSON endpoints against a fixed contract
- ONNX Runtime in-process serving (taste models)
- Molecule handling (RDKit) and the rule layer (sour/salty/safety flags)
- Auth and per-seat licensing (pilot stage)
