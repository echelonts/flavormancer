# Aroma — evaluated and deferred

Flavor = taste **+** aroma. The taste side ships and is clean (sweet/bitter/umami
~0.95 AUROC + intensity + sour/salty). The aroma side is **deliberately deferred** —
not for lack of effort, but because the public, commercially-clean data is
insufficient. This documents the evaluation, so the decision is legible.

## What an aroma model needs

A model mapping a molecule's **structure → odor descriptors** (sweet, floral,
woody…). The state of the art is the **Principal Odor Map** (Lee et al., *Science*
2023; Google/Osmo), reimplemented open-source as **OpenPOM** (`BioMachineLearning/
openpom`, MIT) — a message-passing GNN trained on ~5,000 **expert-labeled** molecules
(the GS-LF dataset).

## The licensing wall (audited)

OpenPOM's *code* is MIT, but its *training data* is restricted. A full audit of the
Pyrfume catalogue (see [`DATA-SOURCES.md`](DATA-SOURCES.md)) found that **every rich
odor-descriptor dataset is proprietary or NonCommercial** — Leffingwell, GoodScents,
Arctander, Flavornet (©Datu), FlavorDB / FooDB (NC), OlfactionBase ("all rights
reserved"), AromaDB (CSIR), Dravnieks (ASTM ©), sharma_2021 (ACS ©), snitz_2019
(CC-BY-NC). The **only** commercially-clean odor-descriptor set is **`keller_2016`**
(Keller & Vosshall 2016, *BMC Neuroscience*, CC-BY-4.0; ~480 molecules, 20 descriptors).

## The empirical result

We aggregated `keller_2016` ([`training/build_aroma_dataset.py`](../training/build_aroma_dataset.py))
and trained one RandomForest regressor per descriptor on 2048-bit Morgan fingerprints
([`training/train_aroma.py`](../training/train_aroma.py)), scored by **honest 5-fold
cross-validation** (400 trees):

| descriptor | CV-R² | descriptor | CV-R² | descriptor | CV-R² | descriptor | CV-R² |
|---|---|---|---|---|---|---|---|
| acid | −0.24 | cold | −0.19 | fruit | −0.05 | sour | −0.05 |
| ammonia | −0.15 | decayed | −0.20 | garlic | −0.12 | spices | −0.12 |
| bakery | −0.22 | edible | −0.19 | grass | −0.45 | sweaty | +0.03 |
| burnt | −0.13 | fish | −0.04 | musky | −0.23 | sweet | −0.05 |
| chemical | −0.10 | flower | −0.12 | warm | −0.28 | wood | −0.08 |

**All 20 descriptors scored CV-R² ≤ 0** (range −0.45 to +0.03) — every model is
*worse than predicting the mean*. **0/20 usable heads.**

## Root cause

`keller_2016` is **naive-subject** data: random volunteers rating *unfamiliar*
molecules on a 0–100 scale. People can't reliably name what a molecule smells like,
so the labels are noise (the per-descriptor means compress to ~22–30 for nearly every
molecule). The *learnable* odor data uses **expert** labels — which is exactly the
data that's restricted. **That is the structural reason the entire field trains on
GS-LF**, and why a clean public aroma model isn't currently possible.

## Decision

We do **not** ship a negative-R² model — it would output confident, wrong smells, and
a flavor chemist would catch it instantly (worse than nothing). `predict_aroma()`
returns an honest "not available" marker, and **we lead with the taste engine**.

The **OpenPOM scaffold (`training/train_odor.py`) is kept** as the aroma engine for
when clean fuel exists:
1. **License PMP 2001** (~$2,775, Leffingwell & Associates) → re-curate → train.
2. **A customer's own odor data** (the paid pilot) → train on-prem.
3. A future open *expert-labeled* dataset, if one emerges.

Until then: **aroma comes with your data** — which is on-thesis (public data proves
the method; the customer's data unlocks the rest).
