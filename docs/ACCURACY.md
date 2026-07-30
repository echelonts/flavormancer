# How accurate is Flavormancer, really?

A plain-language guide to every number this tool reports about itself — what it means, what it
hides, and how much to trust a given answer. Written to be readable without a machine-learning
background, and specific enough to be checked.

Companion to [`HOW-IT-WORKS.md`](HOW-IT-WORKS.md) (how it's built) and
[`METHODS.md`](METHODS.md) (every rule and threshold, with rationale).

---

## The setup

Flavormancer is **190 separate yes/no experts**, called *heads*. One asks *does this smell like
vanilla?* Another asks *does this taste bitter?* A third asks *does this feel cooling in the
mouth?* Each looks at a molecule's structure and returns a number between 0 and 1 — how strongly
it believes the answer is yes.

| modality | heads | what they answer |
|---|---|---|
| Taste | 6 | the five basics — sweet, bitter, umami, sour, salty — plus `tasteless` |
| Aroma | 167 | vanilla, citrus, smoky, pine, jasmine… |
| Mouthfeel | 5 | cooling, warming, pungent, tingling, astringent |
| Safety | 12 | Tox21 assay screens — caution flags, never a clearance |

A head has to *earn* its place: it only ships if it clears a minimum score on data it was never
trained on. Heads that fail are not shipped at all.

---

## How we test honestly: 5-fold cross-validation

We have odor data for **2,403 molecules**. Split them into 5 piles. Train on 4 piles, test on the
5th. Rotate, so every pile gets a turn as the test set.

The point of the exercise: **every molecule is graded by a model that never saw it.** A model that
simply memorised its training data would ace a test on its own study material — this makes that
impossible.

You will see the phrase **out-of-fold** throughout. It means exactly this: *scored by a model that
was blind to that molecule.* Every accuracy number on this page is out-of-fold. None of them are
the model grading its own homework.

---

## AUROC — the number that fooled us

**AUROC measures ranking.** Hand a head one real vanilla molecule and one non-vanilla molecule:
how often does it score the vanilla one higher? 0.5 means coin flip. 1.0 means never wrong.

It sounds like the whole story. It isn't, and the gap is the most important thing on this page.

> **The screening trap.** Imagine a test for a disease that 1 person in 200 has. The test can rank
> sick people above healthy people almost perfectly — and *still* flag 100 people to find 10 real
> cases, because there are 199 healthy people for every sick one. Being excellent at ranking does
> not stop you drowning in false alarms.

That is precisely Flavormancer's situation. A rare descriptor like `ginger` has **11 known
positive examples out of 2,403 molecules**. So both of these are true of the same head:

| head | AUROC | precision |
|---|---|---|
| `ginger` | **0.979** | **0.10** |
| `coffee` | **0.960** | **0.46** |
| `pine` | 0.905 | **1.00** |

`ginger` at AUROC 0.979 ranks beautifully. At precision 0.10 it is right **one time in ten** when
it actually says "ginger."

**AUROC is blind to rarity. Precision is not.** For a long time this project published only the
blind one. That is now fixed, and both are reported everywhere a head appears.

---

## Precision and recall — the two worth explaining to anyone

- **Precision** — *when it says yes, how often is it right?* This is **trustworthiness**.
- **Recall** — *of all the real ones out there, how many did it catch?* This is **thoroughness**.

They trade against each other. Raise a head's bar and you get fewer false alarms but more misses.
Lower it and you catch more but cry wolf more often.

---

## Per-head thresholds, and the precision floor

**The old way:** every head fired at **0.5**. Above it, "yes"; below it, "no." That is just a
coin-flip line — nobody chose it for any chemical reason.

It was quietly wrong for the thin heads. With 13 positive examples against 2,400 negatives, a
model hedges even when it has genuinely learned the class, so a real `pine` match could score 0.42
and never be shown to you.

**The new way:** every head gets **its own bar**, fitted on out-of-fold data, under one rule —

> a head labelled **confident** must be right **more than half the time**.

Bars landed all over the place, because heads genuinely differ:

| head | its bar | precision at that bar |
|---|---|---|
| `pine` | 0.75 | **1.00** — never wrong on unseen molecules |
| `astringent` | 0.71 | **1.00** |
| `cooling` | 0.64 | **1.00** |
| `vanilla` | 0.42 | **1.00** |
| `coffee` | 0.43 | 0.50 — just clears |
| `rosemary` | 0.16 | 0.06 — cannot clear the bar at *any* setting |

Note that `pine` moved **up**, not down. That is the tell that this is calibration and not
number-massaging: a system tuned to flatter itself lowers bars, it does not raise them. Bars run
from 0.16 to 0.85 and **no head sits at either limit**, so the data chose, not the clamp.

### The mistake we made getting here

The first attempt maximised a metric called **F1** (a precision/recall blend). On a badly
imbalanced head, F1 peaks in a *low-precision* regime, because recall climbs faster than precision
falls. It handed `blackberry` a bar of **0.18**, where **96% of its calls were wrong**, and would
have had `sweet` firing on 1,317 of 8,855 molecules at 18% precision.

That is what a metric quietly flattering the system looks like. The 50% precision floor exists
because of it, and the reasoning is recorded in `train_aroma._calibrate` so nobody repeats it.

---

## "Indicative" heads — marked, never removed

**73 of 167 aroma heads cannot reach 50% precision at any threshold.** They are right less than
half the time when they fire.

They are **not deleted, disabled, or hidden.** They keep their score, their column in the
178-dimension flavor profile, their chips in search, their colour on the map, and every molecule
they find. What changes is one word: they are reported as **indicative** rather than *confident*,
and the UI hatches their bar so you can see it at a glance.

The reasoning: firing well above the base rate is real evidence, and deleting them would remove
73 usable notes from search to paper over a labelling problem. Mark it, don't hide it.

Progress on these is tracked in [#262](https://github.com/echelonts/flavormancer/issues/262) — the
goal is to build every one of them up to confident, not to retire them.

---

## So: how accurate is it?

**Taste — strong.** Six heads with hundreds of examples each.

| head | AUROC |
|---|---|
| umami | 0.990 |
| sweet | 0.952 |
| bitter | 0.946 |
| salty | 0.933 |
| sour | 0.905 |
| tasteless | 0.885 |

**Mouthfeel — strong.** All five clear the precision floor.

| head | precision |
|---|---|
| cooling | **1.00** |
| astringent | **1.00** |
| tingling | 0.88 |
| warming | 0.69 |
| pungent | 0.57 |

**Aroma — a real spread, and this is the honest part:**

| when the head fires, it is right… | how many heads |
|---|---|
| 90–100% of the time | **25** |
| 75–90% | 9 |
| 50–75% | 60 |
| under 50% *(shipped as `indicative`)* | 72 |

**23 heads have never been wrong on unseen molecules** — clove, jasmine, minty, buttery, cooling,
maple, grape, camphor, hay, cognac, geranium, aldehydic, chamomile, champaca and more.

**Safety — 12 Tox21 screens, caution-only.** These flag a molecule for review. They are never a
clearance, and food-use status comes from open-government registers, not from a model.

---

## Does a head *learn*, or just *memorise*?

A separate honesty check, because none of the above catches it. A head trained on twelve molecules
that all share one chemical scaffold can score 0.99 by recognising that scaffold and nothing else
— it will fire on exactly the molecules it was trained on and stay silent on everything else.
Cross-validation cannot see this, because it only ever asks about molecules *inside* the labelled
set.

So we fire every head across **all 8,855 molecules** and count the hits that were **not** in its
training data. That count is the head's discovery power. Run it yourself:

```bash
python training/audit_generalization.py            # aroma heads
python training/audit_generalization.py --mouthfeel
```

Current state of the 167 aroma heads:

| verdict | count | meaning |
|---|---|---|
| generalises | **146** | finds molecules nobody labelled (median 9) |
| precision-limited | 19 | strict on purpose, to hold the precision floor |
| memorising | **2** | `turmeric`, `celery` — fire only on their own training set |

`turmeric` and `celery` are narrow single-scaffold classes. More molecules will not fix them; that
is a model-architecture problem, tracked with the GNN work in
[#199](https://github.com/echelonts/flavormancer/issues/199).

---

## The short version

> It's 190 separate models. Every one is graded on molecules it has never seen, and every one
> publishes both how confident it is *and* how often it is actually right at that confidence.
> About a hundred are trustworthy enough to act on — two dozen have never been wrong. The rest are
> marked as hints rather than answers, because they're right less than half the time, and saying so
> is better than hiding it.

If someone pushes on why so many are only hints: the rare notes have **10–15 known examples each**.
That is not a broken model — it is not enough chemistry yet, and it is exactly the gap that a
customer's own panel data fills.

---

## Check any of this yourself

Every number above is served live by the running app:

```bash
curl localhost:8000/api/heads | jq '.aroma[] | select(.head=="pine")'
# { "head": "pine", "auroc": 0.905, "threshold": 0.75,
#   "precision": 1.0, "recall": ..., "n_pos": 17, "confident_capable": true }
```

Nothing here is a claim you have to take on faith — the catalog publishes each head's bar, its
measured precision, how many examples it learned from, and whether it is allowed to call itself
confident.
