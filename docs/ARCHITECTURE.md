# Architecture & Decisions

The single source of truth for *what we're building and why*. Every choice here
was made on a concrete reason, recorded so it doesn't get re-argued later.

---

## Two tracks (read this first)

The project runs on two tracks that share the same trained models:

- **Track A — Demo (solo, fast, Python).** The existing Python pipeline plus a
  FastAPI + HTML workbench, run on a single box. Purpose: a lightweight, working
  demonstration of the approach. Mostly built already. Throwaway-grade on the
  serving side — it exists to prove the method, not to ship.
- **Track B — Product (team, .NET/React).** The real, deployable application the
  team builds. Python trains the models; **.NET serves them**; React is the UI.

Both tracks consume the *same* trained model artifacts, so building Track B
costs nothing on Track A — the demo keeps working while the product is built.

---

## The locked stack (Track B)

| Layer | Technology | Why |
|-------|-----------|-----|
| Model training | **Python** (build-time only) | Thin glue over compiled C++/CUDA cheminformatics + tensor libs (RDKit, DeepChem, scikit-learn); every published flavor/odor model lives here. Iterative, exploratory — wants the lowest-friction ecosystem. |
| Model handoff | **ONNX** (+ a checkpoint for the aroma sidecar) | The clean boundary between the Python and .NET worlds. Trained once, exported, shipped. |
| App / API | **ASP.NET Core (C#)** | Enterprise default for a service like this: strong tooling, broad hiring pool, first-class ONNX Runtime support. |
| ML serving | **ONNX Runtime in-process in .NET** | Taste models (sklearn → `skl2onnx`) run inside the .NET app, no Python at runtime. |
| Aroma serving | **Python FastAPI sidecar** *(only if needed)* | The GNN may not export to ONNX cleanly; if not, a thin localhost sidecar does aroma inference only. Best case it exports and there's zero Python at runtime. |
| Frontend | **React 19 + Vite + TypeScript** | Deepest hiring pool and the lightest fit for a simple single-screen workbench. |
| UI kit | **Tailwind + shadcn/ui** *(not MUI)* | shadcn is copy-in, not import: the components live in our source tree where they can be read and owned. MUI would flatten Flavormancer's existing visual identity into Material and we would spend the port fighting it. The trade is real — MUI wins if you need an enterprise data-grid and date pickers tomorrow; this app is cards, chips, charts and a modal, which is shadcn's sweet spot. |
| Database | **PostgreSQL + pgvector** | Mature and battle-tested; pgvector backs the substitution-search index with first-class vector search. |
| Deploy | **Linux + Docker Compose** on the client-owned box | Single-box, small user count → Compose, not Kubernetes. Containers make the OS matrix irrelevant. |

---

## Deliberate exclusions

What we chose **not** to use matters as much as the stack, and both of these come up often enough
to be worth writing down.

**Node.js — build tooling and MCP only, never a third backend.** Two API stacks (the shipping
Python/FastAPI service and the planned .NET one) is breadth; a third is sprawl, and it reads as
indecision rather than range. Node earns its place in exactly two spots: the React toolchain
(Vite, TypeScript, the test runner), and — optionally — a **TypeScript MCP server** alongside the
Python one, which is ~200 lines and demonstrates the official TS SDK against the same contract.

**Laravel — deliberately absent.** It is a genuinely good fit for CRUD-and-content products and is
used heavily elsewhere in this portfolio. It is the wrong tool here: Flavormancer is on-prem
scientific computing, and adding a comfortable framework that proves nothing new would muddy that
story. Choosing against your most familiar stack when it does not fit is the point.

---

## The core principle: Python trains, .NET ships

Training language is an *internal build detail*, not part of the product. Nobody
operating the product runs the training code or cares what it's written in —
they touch the .NET app and the model artifacts it loads. So:

- **Python is confined to training** (offline, build-time) **plus, at most, one
  small aroma inference sidecar.** It never becomes the backbone.
- Python is the right choice *there* precisely because it's the thinnest wrapper
  over the compiled C++ cheminformatics/tensor code that already exists and is
  tested — rebuilding that in C#/C++ buys a worse-tested reimplementation for a
  build-time task no user sees.
- **Everything that ships and gets shown is C#/.NET + React** — which is the
  right place to spend real engineering effort.

ONNX is the bridge: `train (Python) → export ONNX → load in .NET`.

---

## Cross-platform & deployment

- **Windows, macOS, Linux:** fully supported across .NET, Python/PyTorch, ONNX
  Runtime, and Postgres. Linux is the on-prem server target.
- **FreeBSD:** avoid as a target. .NET on FreeBSD is community-port only (not
  official), and PyTorch/ONNX are rougher still. If a client insists, run the
  Linux containers under its compat layer or provision a Linux box — far cheaper
  than chasing FreeBSD-native builds.
- **Containers settle it:** ship as Docker images via Compose; the same stack
  runs anywhere Docker runs. Standardize on Linux.

**Kubernetes: no.** It earns its complexity with many services across many
machines, autoscaling, and unpredictable traffic — the opposite of one box of
containers for a known set of chemists. Compose is the honest-sized tool and
keeps the deployment legible to the operator, which is part of the on-prem
appeal. K8s would only make sense for a multi-tenant cloud SaaS serving many
houses at once — which is deliberately *not* the model here, because on-prem is
the wedge. (If a Kubernetes deployment is ever needed, build it as a separate
artifact, not bolted onto this.)

---

## How the pieces connect

```
 data sources ──► Python training (build-time)
 (ChemTastesDB,        │  RDKit + DeepChem + sklearn
  cosylab, FlavorDB,   ▼
  SweetenersDB)     model artifacts ──► ONNX (taste) ─────────────┐
                                    └─► aroma checkpoint ──┐      │
                                                           ▼      ▼
 React workbench ◄──── JSON API ◄──── ASP.NET Core app  [Py aroma  ONNX
   (browser)                          + ONNX Runtime      sidecar   Runtime
                                      + Postgres/pgvector  if needed] (in-proc)
                                            │
                                            ▼
                              Docker Compose on the client-owned box
```

---

## Status: built vs to-build

**Built (Track A / training, reusable by Track B):**
- `build_taste_dataset.py`, `train_taste.py`, `predict.py` (aroma training added when clean data exists — see [`AROMA.md`](AROMA.md))
- `app.py` + `workbench.html` — the Python/HTML *demo* serving prototype

**Safety (defensive, caution-only — built into predict.py):**
- Disclaimer + scope statement on every prediction (taste/aroma only).
- Structural-alert screen + optional GRAS cross-reference (flags, never clearances).
- `check_mixture()` — curated dangerous-pair lookup (benzoate+ascorbate→benzene; nitrite+amine→nitrosamine; ethanol+urea→ethyl carbamate; asparagine+sugar→acrylamide), not a reaction predictor.
- Physicochemical pack (logP/MW/TPSA/solubility/volatility/pKa), stability watch-flags, and chemesthetic (cooling/pungent/astringent) class flags — each tagged computed/estimate/qualitative.

**To build (Track B / product — the team work):**
- ONNX export for the taste models (`skl2onnx`)
- ASP.NET Core API + ONNX Runtime in-process serving
- Aroma: get the OpenPOM model trained, then ONNX-export or sidecar — **deferred until clean fuel exists; see [`AROMA.md`](AROMA.md)** for the data/licensing situation and the OpenPOM commercial scope
- React workbench against the fixed JSON contract
- Dockerfiles + `docker-compose.yml` (single-box)
- Auth + per-seat (pilot-stage)

See CAPABILITIES.md for the full map of what's built, what's buildable on open
data, what's impossible on open data (and what unlocks it).

This is the path forward for the team.
