# Build Status

A living inventory: what's **actually built**, what's **built-but-needs-validation**,
what's **stubbed/data-gated**, and what's **deliberately deferred**. Updated as
milestones land. For *why* things are the way they are, see `ARCHITECTURE.md`,
`CAPABILITIES.md`, and `AROMA.md`.

## Legend
- ✅ **Built & validated** — trained/run, output sanity-checked
- 🟡 **Built, needs validation** — code exists, not yet under an end-to-end test suite
- 🔩 **Stub / data-gated** — interface exists, returns an honest "not available" until fueled
- 🧱 **Team milestone (Track B)** — scaffold only, on the roadmap
- ⏸️ **Deferred** — intentionally parked, with a documented reason

---

## Track A — Python core (`training/`, runs on the R620)

### Taste engine ✅
- sweet / bitter / umami classifiers (RandomForest on Morgan fingerprints) — AUROC 0.95 / 0.95 / 0.99
- sweetness-**intensity** regressor (SweetenersDB v2.0) — R² 0.82
- sour: trained **indicative** head (AUROC 0.90) **+** acidity SMARTS rule (recall 0.93)
- salty: cation-aware rule **+** verified dataset-label override
- ranked `taste_profile` + `multitaste` flag
- ONNX export with roundtrip self-validation (< 1e-6)

### Behaviour / chemistry packs (`predict.py`) 🟡 *(spot-validated 2026-06-27)*
- `physchem` — MW, logP, TPSA, H-bonds, solubility, volatility, pKa (computed, RDKit)
- `stability` — oxidation / hydrolysis / photo watch-flags (rule)
- `chemesthesis` — cooling / pungent / astringent (rule)
- `safety` — disclaimer, structural alerts, GRAS cross-check, TTC hint, EU allergen labeling
- `check_mixture` — documented dangerous-pair screen
- `analyze_balance` — OAV dosing balance (qualitative)

All return chemically-sane output on spot checks (vanillin correctly flags its
phenol + aldehyde as oxidation-prone; citric acid; glucose). **Validation gap:**
these aren't yet under a regression-test suite — that's the work to move them to ✅.

### Substitution search ✅ *(new — issue #22 core)*
- `substitute()` — Tanimoto / Morgan nearest-neighbor over the labeled molecule set,
  each neighbor returned with its known tastes
- demonstrated: glucose → ribose / sugars (sim 0.94); vanillin → ethyl ferulate /
  creosol / anisaldehyde (vanillin-adjacent aromatics)
- this is the **Track-A** implementation; Track B mirrors it as a pgvector ANN query
  over the same fingerprints (M6)

### Track-A demo ✅
- FastAPI `app.py` (`/api/predict`, `/api/neighbors`) + `workbench.html`

### Data-gated / stubs 🔩
- `retention_index` (Kovats RI) — needs a trained RI QSPR / NIST data
- `analyze_balance` **quantitative** dosing — needs odor-threshold tables
- GRAS cross-check — needs the FEMA/FDA GRAS list loaded (`gras_reference.parquet`)
- aroma — see **Deferred** below

---

## Track B — Product (the team builds)
- 🧱 `api/` (.NET) — buildable skeleton + `/health`; endpoints / rule-port / in-process
  ONNX serving = **M2** (Aaron)
- 🧱 `frontend/` (React workbench) — greenfield = **M3** (Jamie)
- 🧱 `infra/` (Docker Compose, Postgres + pgvector) = **M5** (Ty)
- 🧱 substitution **at scale** (pgvector ANN), auth / per-seat = **M6**

---

## Deferred
- ⏸️ **Aroma model.** No commercially-clean *public* data yields a working model
  (`keller_2016` scored CV-R² ≤ 0 across all 20 descriptors; the rich set, GS-LF, is
  NonCommercial). The **engine** (OpenPOM — MIT *code*) is kept and wired; it's
  unlocked by **licensed PMP 2001** or **customer data**. Full record: `AROMA.md`.

---

## Product vision — staging (near-term → long-horizon)

What we sell, and when:

1. **Now — clean public data.** Taste + behaviour + safety read on any structure,
   plus the **substitution library** (swap an ingredient for a close analogue).
   This is the public demo and the open-door.

2. **Paid pilot — their data.** An **aroma model** trained on the customer's
   licensed/owned odor data (GC-MS to identify the molecules **+** their sensory
   panel's descriptors as labels — both are required; GC-MS alone has no smell
   labels to learn). Plus **blend-ratio suggestions** tuned on their formulation
   archive. The pitch line is literal: *"aroma comes with your data."*

3. **Long-horizon — R&D + a regulatory wall.** **Novel-molecule generation**
   (propose new flavor molecules) and **retrosynthesis** (synthesis routes).
   Technically far harder, and gated by food-safety approval — a newly-generated
   molecule isn't GRAS and can't be sold as a flavor without a regulatory path.
   A research direction, not a near-term feature; documented here so the staging
   is explicit and doesn't get over-promised in a pitch.
