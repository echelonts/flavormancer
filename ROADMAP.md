# Roadmap

Development runs in milestones — each a coherent, shippable slice. Commits and
PRs reference their milestone (e.g. *"Part of M2"*), and progress is tracked in
[GitHub Milestones](../../milestones) and Issues.

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0 — Repo & scaffolding** | License, CI, `CONTRIBUTING`, `CODEOWNERS`, templates, monorepo layout | ✅ Done |
| **M1 — Training pipeline (Python)** | Dataset build, taste training, prediction core, ONNX export, demo workbench | ✅ Done |
| **M2 — .NET API + ONNX serving** | ASP.NET Core skeleton, load taste ONNX, `/predict`, in-process ONNX Runtime | 🔜 Next |
| **M3 — React workbench** | React UI against the fixed JSON contract; taste meters with confidence tags | Planned |
| **M4 — Aroma model** | Train OpenPOM on Leffingwell; ONNX-export or sidecar; wire `predict_aroma()` | Planned |
| **M5 — Packaging** | Dockerfiles, `docker-compose`, Postgres + pgvector, single-box deploy | Planned |
| **M6 — Pilot-ready** | Auth + per-seat, pgvector substitution search, polish, demo script | Planned |

## The two tracks

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full rationale.

- **Track A — Python training + demo.** Largely complete (M1): the model
  pipeline plus a lightweight FastAPI/HTML workbench.
- **Track B — the .NET/React product.** M2–M6: Python trains → exports ONNX →
  ASP.NET Core serves in-process → React workbench → Postgres/pgvector → Docker
  Compose on a single box.

Both tracks consume the *same* trained model artifacts.
