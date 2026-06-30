# docs/

Architecture, capability catalogue, and design decisions.

- `ARCHITECTURE.md` — the locked stack and the reasoning behind each choice
- `CAPABILITIES.md` — what the tool does, what's buildable on open data, and the
  honest limits of public-data prediction
- `METHODS.md` — every rule, threshold, heuristic, and trick the system uses, with rationale and ceiling
- `SOURCES.md` — data sources, libraries, research, and license attribution
- `API-CONTRACT.md` — the fixed JSON contract the .NET API exposes and the React UI consumes
- `DATA-SOURCES.md` — data acquisition tracker: what each source unlocks and how to get it
- `DATA-REQUIREMENTS.md` — what data unlocks each capability, the formats we accept, and what we ask clients to provide
- `AROMA.md` — why the aroma model is deferred (clean-data audit + the empirical evaluation + OpenPOM's commercial scope)
- `BUILD-STATUS.md` — living inventory: what's built / needs validation / stubbed / deferred, and the product-vision staging

These document *what we're building and why*, so decisions don't get re-argued.
