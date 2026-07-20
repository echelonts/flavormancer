# Aroma-head audit — where more data buys more heads

> **Update — supplement applied.** `build_aroma_supplement.py` added a curated **public-domain**
> character-impact set (`aroma_supplement.csv`, resolved via PubChem) for the sparse descriptors
> below. Result: **24 → 37 heads** — **13 new**: `coconut` (0.89), `nutty` (0.94), `caramel` (0.91),
> `winey` (0.90), `onion` (0.87), `honey` (0.80), `herbal` (0.80), `vanilla` (0.93), `buttery` (0.87),
> `balsamic` (0.93), `smoky` (0.99), `cinnamon` (1.00), `spicy` (0.71) — and `grassy` kept above the
> bar (0.80) with its classic green-leaf volatiles. `coconut` is now *predictable* (γ-nonalactone →
> coconut 1.0), which closes its earlier data-gate. `spicy` finally crossed (0.71, borderline — it
> shares character molecules with cinnamon/clove). **Only `sweet`-odor (0.637) remains unlearnable**
> even with 185 examples — a genuine representation limit (revisit with a GNN).
>
> **Small-n caveat (honest):** the *highest*-AUROC new heads are small, structurally-homogeneous
> classes — e.g. `cinnamon` reaches CV-AUROC **1.00** only because all 10 positives are one
> cinnamaldehyde-family scaffold the fingerprint separates trivially. That is memorising a scaffold,
> not a superb general model: these heads are **narrow and high-variance** (like `grassy`/`coffee`)
> and will sharpen — or get honestly re-scored — with more diverse examples. Treat AUROC on n≈10
> heads as indicative, not gospel. The numbers
> below describe the **pre-supplement** baseline + the standing plan.


An honest look at the aroma model: which odor descriptors we can predict from structure today,
which are *just* out of reach, and exactly what data would unlock more. Regenerate the numbers
with `python training/train_aroma.py` (deterministic, `random_state=42`).

**Bar:** an odor descriptor ships as a head only if it has **≥ 10 positive examples** and clears
**CV-AUROC ≥ 0.70** (5-fold). Of **57** candidate descriptors in the current public-domain corpus
(981 labelled molecules): **24 ship**, **2 were evaluated and dropped**, **31 were skipped for too
few positives**.

## 1. The 24 shipping heads (for reference)

AUROC 0.711 → 0.975. The weakest are small-sample (`grassy` 0.711 / n=10, `woody` 0.759 / n=12,
`green` 0.773 / n=13) — they ship, but their AUROC estimates are high-variance at n≈10–13, so more
data would **robustify** them, not just add new ones.

## 2. Evaluated but dropped — the true near-misses

| descriptor | n_pos | CV-AUROC | verdict |
|---|---|---|---|
| **spicy** | 16 | 0.674 | **Chase it.** Just under the bar with modest data — a handful more labelled "spicy" molecules will likely push it over 0.70. Best quick win. |
| **sweet** (odor) | 185 | 0.666 | **Not a data problem.** Plenty of examples, still not learnable from fingerprints — "smells sweet" is structurally diffuse. Needs a richer representation (a GNN, see the model task) or stays documented-only. |

**This settles the "sweet" question.** *Taste*-sweet is a trained head (from ChemTastesDB, learnable);
*odor*-sweet is **not** a head — it was tried (185 positives) and only reached 0.666. So the two "sweet"
signals are genuinely different, and the app already reflects that: the card shows `sweet` only under
**TASTE MODEL**; there is no predicted `sweet` aroma. The clarity fix is purely labelling — never imply a
*predicted* sweet aroma; where "sweet" appears as a note it is documented-only.

## 3. Skipped for too few positives — the real opportunity (n_pos < 10)

These have no head *at all* because they can't even be evaluated at the 10-positive bar. This is where
data collection has the highest leverage.

- **Near the threshold (n_pos 5–9)** — a few more examples each and they get evaluated, several likely
  clearing 0.70:
  `winey(9) · buttery(8) · onion(8) · herbal(7) · balsamic(7) · honey(7) · fresh(7) · musky(7) ·
  vegetable(7) · caramel(6) · coconut(6) · anise(6) · burnt(6) · tarry(6) · apple(5)`
- **Very sparse (n_pos 1–4)** — need substantial data:
  `vanilla(4) · banana(4) · meaty(4) · nutty(3) · clove(3) · cheesy(3) · waxy(3) · soapy(3) ·
  creamy(2) · fecal(2) · smoky(1) · cherry(1) · cinnamon(1) · coffee(1) · cocoa(1)`

**This explains the "coconut — no clean carrier, data-gated" message**: `coconut` has only **6** labelled
positives, below the bar, so no head exists to predict or suggest it. The classic carrier
(γ-nonalactone) simply isn't labelled "coconut" enough times in the clean public set.

## 4. The plan (owner: Aaron / `area:training`)

1. **Quick win:** collect a modest set of labelled **`spicy`** molecules → re-train → likely a 25th head.
2. **Highest leverage:** push the **n_pos 5–9** descriptors over 10 (especially `caramel, coconut, honey,
   buttery, herbal, balsamic, vanilla`) — small, targeted labelling unlocks a batch of new heads and,
   downstream, real gap-analysis carriers for those notes (fixes the "data-gated" suggestions).
3. **Not a data fix:** `sweet`-odor (and other diffuse percepts) — revisit with a **GNN / learned
   representation**, not more fingerprint data.
4. **Robustify:** add examples for the fragile shipping heads (`grassy, woody, green, rose, rancid, pine`).

Only truly-open (CC-BY / public-domain) odor data or the customer's own labels — never the restricted
GS-LF / Leffingwell sets in the commercial edition (see `docs/DATA-SOURCES.md`).
