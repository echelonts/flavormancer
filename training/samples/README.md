# Curated-data samples (schema only)

These are **small representative samples** of Flavormancer's curated training data, kept public so
the framework is demonstrable. The **full curated datasets are private** — they are the assembled,
open-government-cited training data (the moat), and are not committed to this public repository.

The build/train scripts (`build_aroma_supplement.py`, `build_aroma_dataset.py`, `train_aroma.py`,
`predict.py`) all **skip cleanly** when the full files are absent, so the pipeline still runs on the
inline curated set — bring your own data (or license ours) to reproduce the shipped models.

Every food-clearance in the full set is cited to an open-government register (EU/GB Union List via
`data.food.gov.uk` under OGL v3, or US 21 CFR / FDA SAF, public domain). See `docs/SOURCES.md`.
