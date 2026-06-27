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

### Can we use OpenPOM commercially? (the precise scope)

Yes for the **engine**, no for the **fuel it ships with** — and that distinction is
the whole answer:

| Component | License | Commercial use |
|---|---|---|
| OpenPOM **code** (the GNN / message-passing model) | MIT | ✅ **Yes** — use the architecture freely |
| Bundled **data** (`curated_GS_LF_merged_4983.csv`) | GS-LF = Leffingwell/GoodScents, **NonCommercial** | ❌ **No** — the MIT repo does *not* relicense third-party data it doesn't own (same rule as Pyrfume/SweetenersDB: a repo's license only covers what the uploader authored) |
| Any **pretrained weights** trained on that data | derivative of NonCommercial data | ❌ **Murky/risky** — a model trained on NC data and used commercially is legally unsettled; not something to bet a product on |

So **OpenPOM does *not* "come with commercially-allowed data"** — it comes with
*NonCommercial* data. The code was never the blocker; the data always was.

**The richest target — deferred for *legal* reasons, not technical ones.** To be
explicit: OpenPOM's curated `curated_GS_LF_merged_4983.csv` (~4,983 molecules with
expert odor labels) is the **richest aroma-descriptor dataset we found anywhere**, and
it's exactly what we'd train on — it's what OpenPOM itself trains on, so the model
*would* work. We defer it **solely because it is NonCommercial / proprietary**, not
because of any limitation in our pipeline. A more legally-aggressive actor might train
on it; for a commercial product we will not. The clean route to comparable richness is
to **license** the equivalent data (Leffingwell **PMP 2001**) or to use the customer's
own — same data quality, with the commercial rights attached.

**What this means for us:** we keep only OpenPOM's *code* — the MIT architecture and
training recipe — and we **never use its GS-LF-trained weights**. When clean fuel
exists we **train our own weights from scratch** on it: either a **licensed copy**
(Leffingwell PMP 2001, ~$2,775, ideally licensed by the customer and run on-prem) or
**the customer's own odor data**. There is no shortcut around this — a usable aroma
model *must* be our own, trained on data we can legally use. That's exactly the taste
pipeline over again: structures + labels in, model out. (For a small set, plain
RandomForest suffices — see the taste heads; OpenPOM's GNN only earns its keep once
the labeled set is large, thousands of expert-labeled molecules like PMP 2001 — so
"keep OpenPOM" really means *keep the option of its architecture*, not any pre-baked
model.) The only requirements: the labels are **expert sensory descriptors** (GC-MS
identifies the molecules but carries no smell labels) and the data is ours/licensed.

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

We carry **no dead scaffold** — the architecture decision lives here, not in an
unrunnable stub. **OpenPOM is the chosen architecture** (MIT, re-addable in an
afternoon) for large sets; RandomForest suffices for small ones, exactly as the
taste heads and the `train_aroma.py` evaluation already demonstrate. We build the
training script when clean fuel exists:
1. **License PMP 2001** (~$2,775, Leffingwell & Associates) → re-curate → train.
2. **A customer's own odor data** (the paid pilot) → train on-prem.
3. A future open *expert-labeled* dataset, if one emerges.

Until then: **aroma comes with your data** — which is on-thesis (public data proves
the method; the customer's data unlocks the rest).
