# End-to-end UI tests

Playwright tests that drive the real workbench in a headless browser.

## Running

Start the app, then point the tests at it:

```sh
# full suite — needs a server WITH trained models (e.g. the demo box)
FLAVORMANCER_URL=http://127.0.0.1:8000 pytest tests/e2e

# just the smoke tests — need only a running server, no models
FLAVORMANCER_URL=http://127.0.0.1:8000 pytest tests/e2e -m smoke
```

Requires `pytest`, `playwright`, and a browser (`python -m playwright install chromium`).

## How skipping works (so it's safe everywhere)

- **No server reachable** → every e2e test skips (the plain unit-test CI job never fails on them).
- **`smoke` tests** need only a running server; they catch JS errors, broken HTML, and dead
  endpoints even with **no models** — this is what CI runs (`.github/workflows/ci.yml`, `e2e` job,
  against a model-less app).
- **Model-dependent tests** use the `needs_models` fixture and skip when the server has no trained
  models. Run them on a box that has `aroma_models/` + the parquet tables.

## Coverage

Smoke: page loads without JS errors, key sections present, Compare/Formulation/How-it-works open
by default, how-it-works doc links. Model-dependent: molecule read, Formulation Studio starter
(profile + data-gate), Compare (loaders clear), Flavor Studio search, chirality note + data-gate,
flavor-space map.
