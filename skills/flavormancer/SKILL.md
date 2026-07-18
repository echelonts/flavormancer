---
name: flavormancer
description: >-
  Bench-chemistry flavor & aroma analysis from chemical structure. Use when working on
  food/beverage flavor formulation, ingredient substitution, or molecular taste/aroma
  questions: predict a molecule's taste (6 heads) and aroma (24 heads); analyze a whole
  formulation "before you pour" (blended note-profile, overpowering-component flag, gap vs a
  target, documented hazards); screen mixtures for documented combination hazards; predict
  reaction products; find molecules by note, by named flavor, or by structural similarity
  (substitutes); explore stereoisomers; and query the flavor-space map. All predictions come
  from on-prem RandomForest models — never a safety, GRAS, or regulatory clearance.
---

# Flavormancer

Flavormancer predicts taste and aroma from chemical structure using on-prem RandomForest models
(Morgan fingerprints + physicochemical descriptors). This skill drives it through a small CLI.

## Setup

The Flavormancer API must be running and reachable. Set `FLAVORMANCER_URL` if it isn't on the
default `http://127.0.0.1:8000`:

```sh
export FLAVORMANCER_URL=http://127.0.0.1:8000
```

Run every command with the bundled script (stdlib only — no install):

```sh
python scripts/flavormancer.py <command> ...
```

Each command prints JSON. Molecules can be given as a **common name, IUPAC name, or SMILES**.

## Commands

| Command | Use it to… |
|---------|-----------|
| `read <molecule>` | Predict one molecule's taste (6 heads) + confident aromas + GRAS status. |
| `read-full <molecule>` | The complete read: physicochemical props (logP/MW/TPSA, **boiling point, vapor pressure**, solubility, volatility), stability, chemesthesis, chirality, retention index, EU-allergen labeling, spectra links, full safety block. |
| `formulate <name:ppm> … [--target notes] [--process p]` | **The flagship.** Read a whole recipe: blended note-profile with the driving ingredient per note, the overpowering-component flag, a gap analysis vs `--target` (food-safe add/cut suggestions), and a hazard screen. `ppm` is optional. |
| `mixture <ing> … [--process p]` | Screen a mixture for **documented** combination hazards (e.g. benzoate + ascorbate → benzene), gated on process. |
| `reactions <ing> … [--process p]` | Indicative reaction-template products (each with its own predicted taste + aroma). |
| `notes <note> … [--any-source] [--limit N]` | Find food-safe molecules carrying a set of notes (`citrus fresh sweet`). |
| `flavor <name>` | The molecule(s) that MAKE a named flavor (`banana` → isoamyl acetate). |
| `substitutes <molecule> [-k N]` | Structurally nearest molecules — drop-in swaps / reformulation. |
| `stereoisomers <molecule>` | Every stereoisomer with any documented isomer-specific odor/taste. |
| `categories` / `map [--label L] [--full]` | Browse categories; query the ~8k-molecule flavor map (label distribution, a label's molecules, or `--full` for every point). |

## Examples

```sh
# A single molecule
python scripts/flavormancer.py read "ethyl maltol"
python scripts/flavormancer.py read-full vanillin        # includes vapor pressure, BP, spectra links

# A formulation, aiming for a target profile
python scripts/flavormancer.py formulate "limonene:250" "citral:40" "linalool:25" --target citrus,fresh

# Safety + reactions
python scripts/flavormancer.py mixture "sodium benzoate" "ascorbic acid"
python scripts/flavormancer.py reactions "acetic acid" "ethanol"

# Discovery
python scripts/flavormancer.py flavor banana
python scripts/flavormancer.py notes citrus fresh sweet
python scripts/flavormancer.py substitutes vanillin -k 6
```

## How to interpret results

- **Taste** = six RandomForest heads as probabilities (sweet/bitter/umami/sour/salty/tasteless);
  sour and salty also carry a rule verdict. **Aroma** = 24 odor heads; "confident" means score ≥ 0.5.
- **Formulation profile** is **directional**: contributions are weighted by odor impact (OAV where
  odor thresholds are loaded, else mass × volatility). It sharpens to calibrated intensity with the
  customer's odor-threshold / panel data (the tool says so in `data_gates`).
- **Stereoisomers**: the models are achiral, so any odor/taste *difference* between isomers is
  **documented** (from PubChem), not predicted.

## Guardrails

Everything here is **flavor prediction only** — NOT a safety, toxicity, GRAS, regulatory, or
stability determination. Hazard/mixture screens are **curated and documented**, not a general
reaction predictor. Always defer to qualified toxicology and regulatory review before use.
