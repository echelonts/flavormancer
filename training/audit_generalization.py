"""audit_generalization.py — does a head LEARN its class, or just memorize its training set?

A high CV-AUROC is necessary but not sufficient. A head trained on twelve molecules that all share
one scaffold can score 0.99 by recognizing that scaffold and nothing else: it will fire on exactly
the molecules it was trained on and stay silent on every other molecule in the corpus. That head
has memorized. It is not useless — it still labels its own positives correctly — but it cannot
discover, so it must not be presented as if it can.

The test is deliberately simple and hard to fool:

    fire the head over EVERY molecule in the corpus, then count the hits that are NOT in its
    training positives.

That count — `novel` below — is the head's discovery power. Zero means memorization. The AUROC
never reveals this, because cross-validation only ever asks about molecules inside the labelled
set; this asks what happens outside it.

The fix for a memorizing head is structural DIVERSITY among its positives, not more of them: a
`tingling` head trained on nine Zanthoxylum sanshools learns "sanshool", whereas the same head
trained on sanshools + Echinacea + Anacyclus + Heliopsis amides learns "long-chain unsaturated
N-alkylamide" and starts finding molecules nobody labelled. See #247 (tingling) and #256 (the
sixteen memorizing aroma heads) for the method applied end to end.

Some heads CANNOT be fixed with molecules. A broad, fuzzy, multi-scaffold class like `sweet` odour
(AUROC 0.724 over 208 positives) is not thin — it is genuinely hard, and the honest answer is a
better model (a GNN, #28), not a longer list.

Usage:
    python audit_generalization.py                 # all aroma heads
    python audit_generalization.py --mouthfeel     # the mouthfeel heads instead
    python audit_generalization.py --threshold 0.6 # stricter definition of "fires"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIRE = 0.5   # a head "fires" on a molecule at or above this probability
RELAXED = 0.35  # ...and this is the "would it fire if it were less shy?" probe — see below

# Why two thresholds. A head with 13 positives against 2400 negatives is calibrated conservatively
# even with balanced class weights: it can have genuinely learned its class and still rarely clear
# 0.5 outside the molecules it was fit on. Reading novel@0.5 alone therefore conflates two very
# different failures — a head that learned nothing, and a head that learned the class but is shy
# about saying so. Scoring both separates them, and the distinction changes what you'd do next:
#   novel@0.5  > 0                      -> generalizes. Nothing to fix.
#   novel@0.5 == 0 < novel@0.35         -> UNDER-CONFIDENT. It found real molecules nobody labelled,
#                                          just below the bar. More positives would firm it up; the
#                                          class itself is learnable and the head is not broken.
#   novel@0.35 == 0                     -> MEMORIZING. It fires on its training set and nothing
#                                          else at any reasonable threshold. This is the real
#                                          failure, and the fix is structural diversity.
# `pine` and `rosemary` looked memorizing at 0.5 and turned out to be under-confident (8 and 9
# novel at 0.35); `celery` and `turmeric` were memorizing at both. Same table, opposite verdicts.


def _positives(train_path, heads):
    """{head -> set of InChIKey skeletons it was trained to call positive}."""
    if not Path(train_path).exists():
        return {}
    df = pd.read_parquet(train_path)
    cols = [h for h in heads if h in df.columns]
    skel = df["inchikey"].astype(str).str.split("-").str[0]
    return {h: set(skel[df[h].astype(int) == 1]) for h in cols}


def audit(modality="aroma", threshold=FIRE, relaxed=RELAXED):
    """Return one row per head: name, auroc, n_pos, hits, novel, novel_lo, verdict."""
    import predict as P
    P.MODELS_READY.wait()  # heads load on a background thread — don't race it

    models, train_path, manifest = {
        "aroma": (P._AROMA_MODELS, "aroma_train.parquet", "aroma_models/manifest.json"),
        "mouthfeel": (P._MOUTHFEEL_MODELS, "mouthfeel_train.parquet", "mouthfeel_models/manifest.json"),
    }[modality]
    if not models:
        print(f"no {modality} heads loaded", file=sys.stderr)
        return []

    meta = {}
    if Path(manifest).exists():
        meta = json.loads(Path(manifest).read_text()).get("descriptors", {})

    heads = sorted(models)
    pos = _positives(train_path, heads)

    # score every head over the whole corpus in one pass — the enrichment table already holds a
    # canonical molecule list, so the audit sees exactly what the app serves
    from rdkit import Chem
    df = pd.read_parquet("master_enrichment.parquet")
    mols = [Chem.MolFromSmiles(str(s)) for s in df["smiles"]]
    ok = [i for i, m in enumerate(mols) if m is not None]
    x = np.vstack([P._feat(mols[i])[0] for i in ok])
    skel = [Chem.MolToInchiKey(mols[i]).split("-")[0] for i in ok]

    rows = []
    for h in heads:
        p = models[h].predict_proba(x)[:, 1]
        trained = pos.get(h, set())
        # score at the head's OWN calibrated threshold when it has one, so the audit measures the
        # product's actual behaviour rather than a hypothetical flat cut-off it no longer uses
        thr = meta.get(h, {}).get("threshold")
        thr = float(thr) if isinstance(thr, (int, float)) else threshold
        fired = {skel[i] for i in range(len(skel)) if p[i] >= thr}
        fired_lo = {skel[i] for i in range(len(skel)) if p[i] >= relaxed}
        novel, novel_lo = len(fired - trained), len(fired_lo - trained)
        rows.append({"head": h,
                     "auroc": meta.get(h, {}).get("auroc"),
                     "n_pos": len(trained),
                     "hits": len(fired),
                     "novel": novel,
                     "novel_lo": novel_lo,
                     "threshold": round(thr, 2),
                     "precision": meta.get(h, {}).get("cv_precision"),
                     "capable": meta.get(h, {}).get("confident_capable", True),
                     # A head whose calibrated threshold sits ABOVE the relaxed probe is not shy —
                     # it was deliberately made strict to reach the precision floor, and loosening
                     # it would trade away the accuracy it was tuned for. That is precision-limited,
                     # a different diagnosis with a different fix (more positives, not a lower bar).
                     # Order matters: a head that finds nothing at EITHER threshold is memorizing,
                     # whatever its cut-off. Only once we know relaxing would actually find
                     # something does a high threshold mean "strict on purpose" rather than "broken".
                     "verdict": ("ok" if novel else
                                 "memorizing" if not novel_lo else
                                 "precision-limited" if thr > relaxed else "shy")})
    return sorted(rows, key=lambda r: (r["novel"], r["novel_lo"], -(r["auroc"] or 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mouthfeel", action="store_true", help="audit mouthfeel heads instead of aroma")
    ap.add_argument("--threshold", type=float, default=FIRE, help=f"fire threshold (default {FIRE})")
    ap.add_argument("--relaxed", type=float, default=RELAXED,
                    help=f"under-confidence probe threshold (default {RELAXED})")
    a = ap.parse_args()

    rows = audit("mouthfeel" if a.mouthfeel else "aroma", a.threshold, a.relaxed)
    if not rows:
        sys.exit(1)

    tag = {"ok": "", "shy": "  <- under-confident",
           "precision-limited": "  <- precision-limited (strict on purpose)",
           "memorizing": "  <- MEMORIZING"}
    print(f"{'head':16s} {'AUROC':>6s} {'thr':>5s} {'prec':>5s} {'n_pos':>6s} {'hits':>6s} "
          f"{'novel':>6s} {f'@{a.relaxed:g}':>6s}")
    for r in rows:
        au = f"{r['auroc']:.3f}" if r["auroc"] is not None else "  -  "
        pc = f"{r['precision']:.2f}" if r["precision"] is not None else "  -  "
        ind = "" if r["capable"] else "  (indicative)"
        print(f"{r['head']:16s} {au:>6s} {r['threshold']:5.2f} {pc:>5s} {r['n_pos']:6d} "
              f"{r['hits']:6d} {r['novel']:6d} {r['novel_lo']:6d}{tag[r['verdict']]}{ind}")

    indicative = [r["head"] for r in rows if not r["capable"]]
    shy = [r["head"] for r in rows if r["verdict"] == "shy"]
    plim = [r["head"] for r in rows if r["verdict"] == "precision-limited"]
    mem = [r["head"] for r in rows if r["verdict"] == "memorizing"]
    novel = sorted(r["novel"] for r in rows)
    print(f"\n{len(rows) - len(shy) - len(plim) - len(mem)}/{len(rows)} heads generalize at their "
          f"calibrated threshold (median {novel[len(novel) // 2]} novel discoveries).")
    if plim:
        print(f"\n{len(plim)} PRECISION-LIMITED — strict by design, to hold the 50% precision "
              f"floor: {', '.join(plim)}")
        print("  Not a defect and not a shy head: lowering the bar would trade away the accuracy "
              "it was tuned for. More positives is the only real fix.")
    if shy:
        print(f"\n{len(shy)} UNDER-CONFIDENT — found unlabelled molecules at {a.relaxed:g} but not "
              f"{a.threshold:g}: {', '.join(shy)}")
        print("  These learned their class; they're shy because their positives are heavily "
              "outnumbered. More positives sharpen them. Not broken.")
    if indicative:
        print(f"\n{len(indicative)}/{len(rows)} are INDICATIVE — they never reach 50% out-of-fold "
              f"precision at any threshold, so they fire as evidence, never as a confident call. "
              f"They are still far better than the base rate; they are not yet trustworthy alone.")
    if mem:
        print(f"\n{len(mem)} MEMORIZING — fire on their own training molecules and nothing else, "
              f"at any threshold: {', '.join(mem)}")
        print("  Fix: add STRUCTURALLY DIVERSE positives to the curated supplement, then rebuild "
              "and re-run. More of the same scaffold will not move these off zero.")


if __name__ == "__main__":
    main()
