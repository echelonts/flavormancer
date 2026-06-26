# API Contract (v1)

The fixed JSON contract the **.NET API exposes** and the **React workbench consumes**.
Both sides build against *this document* — it is the single source of truth for the
request/response shapes. It is derived from the reference implementation in
[`training/predict.py`](../training/predict.py); the .NET API ports that logic and its
output **must match these shapes** (verified molecule-for-molecule against Python).

## Conventions

- JSON, UTF-8. Field names are stable; the UI must **ignore unknown fields** (so
  additive changes don't break it).
- **Confidence tiers** — every value is derived one of these ways, and the UI shows
  the tier so nothing reads as more certain than its source:
  `computed` (exact from structure) · `trained` (ML on open data) · `rule`
  (deterministic structural rule) · `estimate` (published QSPR, known error) ·
  `lookup` (from a loaded reference table) · `qualitative` (a class/flag, not a number).
- **Errors** return an HTTP 4xx/5xx status with body `{ "error": "<message>" }`.
- **Optional fields** (`sweet_intensity`, `physchem.measured`, `aroma`, `known_tastes`)
  appear only when their model/table/flag is available — consumers must tolerate absence.

> Example values below are **illustrative** (shape, types, and tiers are what's
> normative — not the specific numbers).

---

## `GET /health`
Liveness probe.

**200** → `{ "status": "ok" }`

---

## `POST /predict`
Single-molecule flavor read.

**Request**
```json
{ "input": "vanillin" }
```
- `input` *(string, required)* — a compound **name** or a **SMILES** string. Resolved
  as SMILES if parseable, otherwise looked up by name.
- Unresolvable input → **422** `{ "error": "couldn't resolve '<input>' to a structure" }`.

**Response 200** *(tier in parentheses)*
```json
{
  "smiles": "O=Cc1ccc(O)c(OC)c1",          // canonical SMILES (computed)

  "sweet": 0.12,                            // probability 0–1 (trained)
  "bitter": 0.74,                           // (trained)
  "umami": 0.03,                            // (trained)
  "sweet_intensity": 1.8,                   // vs sucrose — OPTIONAL (trained)
  "sour": false,                            // RULE call — acidic-group structural rule (rule)
  "sour_reason": [],                        // acidic groups matched (rule)
  "sour_predicted": 0.12,                   // trained sour head — small-data INDICATIVE (trained)
  "salty": false,                           // (rule)
  "salty_reason": "no alkali-salt structure", // (rule)
  "known_tastes": ["bitter"],               // OPTIONAL, verified dataset labels (lookup)
  "multitaste": false,                      // 2+ taste heads ≥ 0.5 (trained-derived)

  "taste_profile": [                        // tastes ranked by dominance (trained heads, desc)
    { "taste": "bitter", "probability": 0.74, "basis": "trained" },
    { "taste": "sweet",  "probability": 0.12, "basis": "trained" },
    { "taste": "sour",   "probability": 0.12, "basis": "trained (indicative)" },
    { "taste": "umami",  "probability": 0.03, "basis": "trained" }
  ],

  "physchem": {
    "computed":  { "mol_weight": 152.15, "logP": 1.21, "tpsa": 46.5,
                   "h_bond_donors": 1, "h_bond_acceptors": 3,
                   "rotatable_bonds": 2, "aromatic_rings": 1, "heavy_atoms": 11 },
    "estimate":  { "logS": -1.6, "note": "ESOL estimate (log mol/L), ~0.7 log RMSE" },
    "qualitative": { "aroma_volatility": "middle", "volatility_note": "...",
                     "ionizable_groups": [ { "group": "phenol", "typical_pKa": "9.8–10.3",
                                            "character": "weak acid" } ] },
    "measured":  { "boiling_point_c": 285, "source": "loaded property table" }  // OPTIONAL (lookup)
  },

  "stability": { "oxidation_watch": [], "hydrolysis_watch": [],
                 "photodegradation_watch": [],
                 "note": "qualitative 'watch for' flags — not a shelf-life prediction" },

  "chemesthesis": { "classes": [], "note": "curated structural class flags, qualitative" },

  "analytical": { "retention_index": { "kovats_ri": null, "note": "needs a trained RI QSPR" } },

  "labeling": { "eu_declarable_allergen": false, "allergen_name": null,
                "note": "EU fragrance-allergen labeling list (curated subset)" },

  "safety": {
    "disclaimer": "Taste/aroma prediction only ...",
    "scope": "Taste/aroma only — not a safety/toxicity/GRAS/stability determination.",
    "structural_alerts": [],                // caution prompts, may be empty (rule)
    "gras_status": "unverified for food use", // (lookup, data-gated)
    "review_required": true,
    "ttc_hint": { "preliminary_tier": "low", "drivers": { "alerts": [], "uncommon_elements": [] },
                  "note": "PRELIMINARY heuristic, not validated Cramer/TTC" }  // (qualitative)
  },

  "aroma": { "descriptors": [ { "label": "sweet", "intensity": 0.0 } ],
             "note": "..." }               // OPTIONAL — only when requested AND the model is trained (trained)
}
```

**Field notes for implementers**
- Taste-head keys (`sweet`/`bitter`/`umami`) are present **per trained classifier**; a
  head below the data threshold is absent and the corresponding rule/flag covers it.
- **Sour carries two signals:** `sour` is the deterministic acidity-rule boolean;
  `sour_predicted` is a small-data **indicative** trained probability. They can disagree
  (the rule flags structure, the model reflects perception) — surface both.
- **`taste_profile`** ranks the trained heads (incl. sour-indicative) by probability,
  descending — the "order of dominance" view. The `sour`/`salty` rules stay separate flags.
- `sweet_intensity` and `physchem.measured` appear only when their model/table is loaded.
- `salty` may be overridden to `true` with `salty_reason: "verified (dataset label)"` when
  a ground-truth label exists (lookup beats rule).
- `aroma` is included only when the request opts in **and** the OpenPOM model is trained;
  until then it is omitted (or returns an honest "not trained yet" marker).

---

## `POST /substitutes`
Nearest-neighbor substitution search ("approved ingredients closest to this one").

**Request**
```json
{ "input": "vanillin", "k": 8 }            // k optional, default 8
```

**Response 200**
```json
{ "neighbors": [
  { "smiles": "O=Cc1ccc(O)cc1", "similarity": 0.83, "known_tastes": ["bitter"] }
] }
```
- **Today:** Morgan-fingerprint Tanimoto similarity (runs without the aroma model).
- **Later:** upgrades to the learned aroma-embedding space (pgvector cosine) once the
  aroma model + embeddings exist — **same response shape**, better neighbors.

---

## Versioning
This is **v1**. Additive changes (new optional fields) are non-breaking; consumers
ignore unknown fields. Breaking changes bump the version and are coordinated via PR.
