# Sources & Attribution

Credit where it's due. Flavormancer stands on a lot of open science and open
software; this file records what we use and who to thank. **Licenses are noted
where known but must be confirmed before any commercial release** — this is a
record of provenance, not legal advice. Get an IP/OSS-license review before ship.

---

## Software & libraries

**Cheminformatics / ML (training side — Python)**
- **RDKit** — molecular parsing, fingerprints, descriptors, SMARTS, InChIKeys. (BSD-3-Clause.)
- **DeepChem** — GNN training framework under OpenPOM. (MIT.) Python-only; the one runtime-Python piece.
- **OpenPOM** (BioMachineLearning/openpom) — the message-passing GNN for odor; open reimplementation of the principal-odor-map work. (MIT — confirm.) The aroma model is theirs in spirit.
- **scikit-learn** — RandomForest taste heads + sweetness-intensity regressor. (BSD-3-Clause.)
- **PyTorch** — tensor/Autograd backend under DeepChem. (BSD-style.)
- **NumPy**, **pandas** — arrays + dataframes. (BSD-3-Clause.)
- **UMAP** — 2D odor/flavor-space map. (BSD-3-Clause.)
- **skl2onnx** — exports the sklearn taste models to ONNX. (Apache-2.0.)
- *(thermo — evaluated for Joback boiling point, then REJECTED for accuracy; not shipped. MIT.)*

**Product / serving side**
- **ONNX** + **ONNX Runtime** — run taste models in-process in .NET. (MIT.)
- **ASP.NET Core / .NET** — the app/API backbone. (MIT.)
- **React** — frontend. (MIT.)
- **PostgreSQL** + **pgvector** — DB + embedding/substitution search. (PostgreSQL License.)
- **FastAPI** — the Track-A demo serving layer (+ aroma sidecar if needed). (MIT.)
- **Docker** — single-box deployment. (Apache-2.0.)

---

## Data sources

**Taste**
- **ChemTastesDB** — Rojas et al., curated taste dataset (sweet/bitter/umami/sour/salty/tasteless). Zenodo, DOI 10.5281/zenodo.5747393 (and the extended record). License **CC-BY-4.0** (commercially clean with attribution).
- **cosylab BitterSweet** — Bagler lab (IIIT-Delhi). Code **AGPL-3.0** (bind-aware: train-your-own-on-data is the safe pattern; `INCLUDE_COSYLAB=False` drops it for clean licensing).
- **FlavorDB** — Bagler lab (IIIT-Delhi); taste + odor + natural-source associations. License **CC BY-NC-SA 3.0** (NonCommercial) — incompatible with a commercial product; **not used**.
- **SweetenersDB v2.0** — Bouysset et al. (2020) *Food Chem.*, building on Chéron et al. (2017); relative-to-sucrose sweetness *intensity* for the regressor. Released **MIT** by the authors' own lab (ChemSenSim, `github.com/chemosim-lab/SweetenersDB`) — the paywall is only on the journal article, not the authors' own data. **In use.**
- **BitterDB** *(future)* — bitterness intensity, if/when added.
- **UMP442 / BIOPEP-UWM** — umami references. BIOPEP-UWM is web-only; the GitHub repost `Shoombuatong/Dataset-Code` carries **no license** (all-rights-reserved) and is umami *peptide* data (a different class from our small-molecule head). **Not used.**

**Aroma**
- **Pyrfume** + **Leffingwell / GoodScents (GS-LF)** odor datasets — the usual training data behind OpenPOM, but **RESTRICTED and NOT USED**: Leffingwell's manifest cites use restrictions (*John Leffingwell & Google*); GoodScents/Arctander/Flavornet (© Datu Inc.) are likewise proprietary. We **exclude all of them** (the demo may go to a customer / commercial use). The aroma model will use only commercial-clean **open** odor data (CC-BY sets like `keller_2016`; smaller — see `DATA-SOURCES.md`). The OpenPOM *code* is MIT.
- **keller_2016** — Keller & Vosshall (2016), *BMC Neuroscience*, **CC-BY-4.0**; ~480 molecules with naive-subject odor-descriptor ratings. The only commercially-clean odor-descriptor set — evaluated for the aroma model and found too noisy to learn from (CV-R² ≤ 0 across all 20 descriptors; see `docs/AROMA.md`).

**Safety / regulatory (lookups — data-gated)**
- **FEMA GRAS list** — usual/maximum use levels for the dosing analyzer. (FEMA.)
- **EU declarable fragrance/flavor allergen annex** — the labeling flags. (EU regulation.)
- **PubChem** — name↔SMILES↔CID resolution. (Public domain data; confirm API terms.)

---

## Research & methods

- **Principal odor map** — Lee et al., "A principal odor map unifies diverse tasks
  in olfactory perception," *Science*, 2023. The basis for the aroma GNN (OpenPOM
  reimplements it; the same group founded **Osmo**). The single most important
  scientific credit here.
- **Molecular fingerprints** — Rogers & Hahn, "Extended-Connectivity Fingerprints,"
  *J. Chem. Inf. Model.*, 2010 (Morgan/ECFP — the taste-head features).
- **Aqueous solubility (ESOL)** — Delaney, "ESOL: Estimating Aqueous Solubility
  Directly from Molecular Structure," *J. Chem. Inf. Comput. Sci.*, 2004.
- **Toxicological Threshold of Concern / Cramer classification** — Cramer, Ford &
  Hall, 1978; **Toxtree** (EU JRC) for the validated decision tree.
- **Odor Activity Value (OAV = concentration ÷ detection threshold)** — standard
  flavor-chemistry framework for blend balance.
- **Group contribution (Joback)** — Joback & Reid, 1987. *Evaluated and rejected*
  for boiling point here (≈90 °C error on benzaldehyde); recorded so the decision
  is documented.
- **Documented food-process contaminants** — benzene (benzoate+ascorbate), N-nitrosamines
  (nitrite+amines), ethyl carbamate, acrylamide (Maillard), furan, 3-MCPD/glycidyl
  esters, 4-methylimidazole, biogenic amines — all from established food-safety literature.

---

## Prior art & landscape (context, not used in the build)

We are not first; the field is active and funded. Mapping it honestly:
- **Osmo** — digitized smell; principal-odor-map authors; B2B fragrance/flavor.
- **Gastrograph AI / Analytical Flavor Systems** — predictive sensory analytics.
- **Tastewise**, **Ai Palette** — F&B trend/flavor AI.
- **Aromyx** — taste/smell biosensors.
- **Senomyx** (acquired by Firmenich) — taste/aroma receptor reverse-engineering.
- **Symrise**, **Givaudan**, **IFF**, **dsm-firmenich**, **MANE/ChemoSensoryx** —
  flavor houses with internal AI for formulation.

---

## Crediting the science

The work this builds on, stated plainly: the aroma model is an open
reimplementation (OpenPOM) of the 2023 *Science* principal-odor-map work; the
taste models train on open curated databases (ChemTastesDB and others); everything
runs on hardware you own. Honesty about provenance is part of the credibility.
