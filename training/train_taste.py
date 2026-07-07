"""
train_taste.py — taste heads (multi-taste) + sweetness-intensity regressor.

Reads the merged dataset from build_taste_dataset.py and trains one binary
RandomForest per trainable taste: sweet / bitter / umami, plus sour, salty, and
tasteless as smaller *indicative* heads (folded up from PubChem documented taste).
Sour AND salty ALSO keep their deterministic chemistry rules in predict.py as a
cross-check (surfaced as sour_predicted / salty_predicted beside the rule's
boolean); tasteless is a documented tasted-vs-tasteless head. Add more data and
the heads sharpen on the next run.

Even with everything merged this trains in minutes on the R620 CPU. The
multi-day budget is the aroma model (deferred — see docs/AROMA.md), not this.

Run order:
  python build_taste_dataset.py   # writes taste_master.parquet (+ sweet_intensity.parquet)
  python train_taste.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

BASIC = ["sweet", "bitter", "umami", "sour", "salty", "tasteless"]
# Salty now ALSO trains as an INDICATIVE head (CV-AUROC ~0.96 once the PubChem documented-
# taste labels are merged in — see augment_from_taste_notes) while KEEPING its alkali-salt
# rule in predict.py as a deterministic second check (surfaced as salty_predicted + salty),
# exactly like sour. Both use the rule to forecast, the model to add a probability.
RULE_TASTES = set()
FP_BITS, FP_RADIUS = 2048, 2
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
# Below this, a taste is too thin for an HONEST head, so it's skipped and
# handled by rule/flag instead. It's not a hard exclusion: add more data (more
# sources) and the taste crosses the line and trains itself on the next run.
# You CAN lower this to force sour/salty heads — but a model on a few dozen
# examples memorizes rather than generalizes, and that's exactly the kind of
# claim that gets punctured by an expert. Lower at your own risk.
MIN_POS, MIN_NEG = 80, 80
OUT = Path("taste_models")
OUT.mkdir(exist_ok=True)
# Clear stale heads first, so a taste that no longer trains (e.g. now rule-handled)
# can't leave an orphan .joblib that export_onnx would pick up.
for _stale in OUT.glob("*_rf.joblib"):
    _stale.unlink()

# Acidic-group SMARTS — used both for the sour rule and to VALIDATE it against
# whatever labeled sour compounds exist (so that data isn't wasted either).
# Match BOTH protonated (-OH) and deprotonated (-O-) forms: sour compounds are
# routinely drawn as carboxylate/sulfonate/phosphate anions or zwitterions, which
# the -OH-only patterns missed (the main driver of the low recall).
ACID_SMARTS = ["[CX3](=O)[OX2H1,OX1-]", "[SX4](=O)(=O)[OX2H1,OX1-]", "[PX4](=O)[OX2H1,OX1-]"]
_ACID = [Chem.MolFromSmarts(s) for s in ACID_SMARTS]


def fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = _MORGAN.GetFingerprint(mol)
    arr = np.zeros((FP_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def featurize(smiles_series):
    feats, keep = [], []
    for i, s in enumerate(smiles_series):
        f = fp(s)
        if f is not None:
            feats.append(f)
            keep.append(i)
    return np.array(feats), np.array(keep)


def train_classifiers(master):
    X_all, keep = featurize(master["smiles"])
    df = master.iloc[keep].reset_index(drop=True)
    manifest = {}
    for taste in BASIC:
        y = df[taste]
        mask = y.notna().values
        yv = y[mask].astype(int).values
        Xv = X_all[mask]
        pos, neg = int((yv == 1).sum()), int((yv == 0).sum())
        if taste in RULE_TASTES:
            print(f"  {taste:7s} RULE by design (pos={pos}, neg={neg}) -> handled in predict.py")
            continue
        if pos < MIN_POS or neg < MIN_NEG:
            print(f"  {taste:7s} SKIP (pos={pos}, neg={neg}) -> handled by rule/omitted")
            continue
        Xtr, Xte, ytr, yte = train_test_split(
            Xv, yv, test_size=0.2, stratify=yv, random_state=42)
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr)
        auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
        print(f"  {taste:7s} AUROC={auc:.3f}  (pos={pos}, neg={neg})")
        joblib.dump(clf, OUT / f"{taste}_rf.joblib")
        manifest[taste] = {"auroc": round(float(auc), 3), "n_pos": pos, "n_neg": neg}
    return manifest


def train_intensity(path=Path("sweet_intensity.parquet")):
    if not path.exists():
        print("  intensity: no sweet_intensity.parquet (download SweetenersDB) -> skipped")
        return
    di = pd.read_parquet(path)
    X, keep = featurize(di["smiles"])
    y = di["log_sweetness"].values[keep]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
    reg.fit(Xtr, ytr)
    r2 = r2_score(yte, reg.predict(Xte))
    print(f"  sweet_intensity R2={r2:.3f}  (n={len(y)})  [small data; treat as indicative]")
    joblib.dump(reg, OUT / "sweet_intensity_rf.joblib")
    return round(float(r2), 3)


def validate_sour_rule(master):
    """Use the (few) labeled sour compounds to measure the acidity rule —
    so the sour data is used for validation rather than discarded."""
    if "sour" not in master or master["sour"].notna().sum() == 0:
        print("  sour: no labels to validate against")
        return
    df = master[master["sour"].notna()]
    tp = fp_hits = pos = 0
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        hit = any(p is not None and mol.HasSubstructMatch(p) for p in _ACID)
        label = int(r["sour"])
        pos += label
        fp_hits += hit
        tp += hit and label
    recall = tp / pos if pos else float("nan")
    print(f"  sour RULE check: recall on {pos} labeled-sour = {recall:.2f} "
          f"(rule fires on {fp_hits}/{len(df)} of the validation set)")


def augment_from_taste_notes(master, path="taste_notes.parquet"):
    """Fold PubChem documented-taste labels in SELECTIVELY: only salty & sour (small classes
    that benefit) and the 'tasteless' molecules (rare confirmed negatives). Sweet/bitter are
    left to ChemTastesDB — a blanket merge adds keyword-text noise there (measured, ~-0.02 AUROC)."""
    import re
    p = Path(path)
    if not p.exists():
        print("  (no taste_notes.parquet — skipping documented-taste augmentation)")
        return master
    tn = pd.read_parquet(p)
    neg = re.compile(r"\b(tasteless|no taste|without taste)")
    # 'tasteless' is now its OWN trained head (a documented, confirmed negative-of-all-tastes
    # class), not just a source of negatives for salty/sour. It reads from the same neg pattern.
    pats = {"salty": re.compile(r"\b(salty|saline)"), "sour": re.compile(r"\b(sour|tart)"),
            "tasteless": neg}
    if "tasteless" not in master.columns:
        master["tasteless"] = np.nan
    existing = set(master["smiles"])
    added_pos = {"salty": 0, "sour": 0, "tasteless": 0}
    new_rows = []
    for smi, txt in zip(tn["smiles"], tn["taste"].astype(str).str.lower()):
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
        if mol is None:
            continue
        tasteless = bool(neg.search(txt))
        labels = {t: (1 if pat.search(txt) else (0 if tasteless else None)) for t, pat in pats.items()}
        if all(v is None for v in labels.values()):
            continue
        if smi in existing:
            for t, v in labels.items():
                if v is None:
                    continue
                cur = master.loc[master["smiles"] == smi, t]
                if v == 1 or (len(cur) and pd.isna(cur.iloc[0])):  # positive wins; neg fills blanks
                    master.loc[master["smiles"] == smi, t] = v
                    if v == 1:
                        added_pos[t] += 1
        else:
            row = {c: None for c in master.columns}
            row.update({"smiles": smi, "inchikey": Chem.MolToInchiKey(mol)})
            for t, v in labels.items():
                row[t] = v
                if v == 1:
                    added_pos[t] += 1
            new_rows.append(row)
    if new_rows:
        master = pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True)
    print(f"  documented-taste augmentation: +{added_pos['salty']} salty, +{added_pos['sour']} sour, "
          f"+{added_pos['tasteless']} tasteless positives (+{len(new_rows)} new molecules)")
    return master


def seed_tasteless_negatives(master):
    """A molecule with any documented basic taste is, by definition, NOT tasteless — so it's a
    clean tasteless NEGATIVE. This turns the (few) documented tasteless positives into a real
    tasted-vs-tasteless head instead of a positives-only class."""
    if "tasteless" not in master.columns:
        master["tasteless"] = np.nan
    tasted = master[["sweet", "bitter", "umami", "sour", "salty"]].eq(1).any(axis=1)
    fill = tasted & master["tasteless"].isna()
    master.loc[fill, "tasteless"] = 0
    print(f"  tasteless negatives seeded from tasted molecules: +{int(fill.sum())}")
    return master


if __name__ == "__main__":
    master = pd.read_parquet("taste_master.parquet")
    print("folding in PubChem documented taste (salty/sour + tasteless negatives):")
    master = augment_from_taste_notes(master)
    master = seed_tasteless_negatives(master)
    print("training taste classifiers:")
    manifest = train_classifiers(master)
    print("validating the sour rule against labeled data:")
    validate_sour_rule(master)
    print("training sweetness-intensity regressor:")
    r2 = train_intensity()
    if r2 is not None:
        manifest["sweet_intensity"] = {"r2": r2}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nsaved models + manifest.json -> {OUT}/")
