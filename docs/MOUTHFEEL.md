# Mouthfeel — a public-domain chemesthesis model

Flavor = taste **+** aroma **+** mouthfeel. Taste and aroma ship as trained heads
([`AROMA.md`](AROMA.md)); this documents the third modality — **mouthfeel / chemesthesis**,
the *trigeminal* (somatosensory) sensations in the mouth: cooling, pungency/heat, tingling,
astringency. It is a **separate modality**, not a slice of aroma — a molecule can *smell* minty
without *feeling* cool, and WS-23 cools the mouth with almost no odour at all.

## What ships (2026-07)

**Five trained heads**, one RandomForest per sensation on the Morgan fingerprint + physicochemical
block (the same stack as taste/aroma), in their own `mouthfeel_models/` directory, loaded and tagged
`mouthfeel` by `predict.py` and folded into the 175-dim flavor-profile vector used for substitutes.

| head | what it is | held-out CV-AUROC |
|---|---|---|
| **cooling** | TRPM8 coolants — menthol family + food-authorised WS-agents | 0.94 |
| **pungent** | TRPV1 / mustard-oil heat & bite — capsaicinoids, isothiocyanates, allium sulfur | 0.97 |
| **warming** | capsaicinoid & warm-spice heat — chili, pepper, ginger, cinnamon | 0.98 |
| **astringent** | tannins & polyphenols — the puckering, mouth-drying sensation | 0.95 |
| **tingling** | paresthesia alkylamides — sanshools, spilanthol, ZP-amides, Echinacea/Anacyclus amides | 1.00 |

The high AUROCs reflect **narrow, structurally-distinct classes** (a capsaicinoid does not look like
a tannin) trained against a broad random background — honest for what they are, but not a claim of
fine-grained intensity resolution. Like aroma, this is **presence/absence**, not scored intensity.
A high AUROC alone does not prove a head is useful; see *Does a head generalize, or is it
memorizing?* below for the test that does.

## `cooling` / `pungent` exist in **both** aroma and mouthfeel

Deliberately — exactly as `sweet` is both a taste head and an aroma head. The *aroma* `cooling`
head predicts a molecule that **smells** cool/minty (menthol, eucalyptol); the *mouthfeel* `cooling`
head predicts the **TRPM8 sensation** (menthol, but also odourless WS-coolants). Same name, different
modality. The profile index keys them by dimension (`aroma:cooling` vs `mouthfeel:cooling`) so they
never collide, and the read surfaces each under its own group.

## Pipeline

- `build_mouthfeel_supplement.py` — the curated list: descriptor → well-known public-domain
  trigeminal-agent **names** (capsaicin, menthol, gallic acid, hydroxy-α-sanshool, …), resolved to
  authoritative canonical SMILES via PubChem (a SMILES hint is supported where a name isn't a
  PubChem synonym). Same provenance discipline as the aroma supplement: only measured, public-domain
  structure→sensation facts (Feist — descriptors aren't copyrightable), never copied from a
  restricted compilation.
- `build_mouthfeel_dataset.py` — positives from the supplement + a broad random background from the
  molecule universe (a molecule with no documented chemesthesis is a negative) → `mouthfeel_train.parquet`.
- `train_mouthfeel.py` — one head per sensation, honest bar: ≥10 positives **and** 5-fold
  CV-AUROC ≥ 0.70, else dropped. Kept heads + AUROC → `mouthfeel_models/` (+ `manifest.json`).

## Food safety is separate

A head predicts a **sensation**, not a food-use clearance. Its training molecules are food-present
naturals (capsaicin in chili, tannins in tea/wine) plus food-authorised agents (nonivamide FL/GRAS,
tannic acid 21 CFR) — but that a molecule is a good *warming* example does not assert it is a
flavouring. The per-molecule food-safe flag and the standing IP/food review gate still govern any
commercial use, exactly as for the aroma corpus.

## Does a head generalize, or is it memorizing?

A high CV-AUROC on a narrow, structurally distinctive class can mean "this head learned the class"
or "this head memorized its training molecules" — and the two look identical in the AUROC column.
The honest test is different: run the head over the whole universe and count how many molecules it
fires on that were **not** in its training set.

Measured over 8.8k molecules:

| head | fires | trained on | **novel** |
|---|---|---|---|
| cooling | 32 | 14 | **16** |
| pungent | 24 | 12 | **12** |
| warming | 21 | 12 | **9** |
| astringent | 19 | 11 | **8** |
| tingling | 16 | 18 | **2** |

`cooling` is the clearest case of the model actually working: it independently surfaced **AR-15512**,
**menthyl lactate** and **menthone glycerol ketal** — real commercial coolants it was never shown.

**`tingling` remains the weak one, and this is a known limitation.** Trained only on Zanthoxylum
sanshools it generalized to *nothing* — it fired on exactly its own training molecules, i.e. it was
a lookup table wearing a 1.00 AUROC. Broadening the positives with paresthesia amides from other
genera (Echinacea tetraenoic isobutylamides, Anacyclus, Heliopsis, Piper — deliberately different
chain lengths and unsaturation patterns) moved it to **2 novel hits**, one of which is piperine.
That is real generalization but a thin margin: the class of public tingle structures is genuinely
small, and the head still leans on the scaffold it was taught. Treat a tingling prediction on an
unfamiliar scaffold with less confidence than the other four; #247 tracks broadening it further.
