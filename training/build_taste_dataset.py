"""
build_taste_dataset.py — merge EVERY taste source into one multi-label table.

Principle: nothing is discarded. Every molecule from every source lands in the
master table. What varies is *how* each taste is used downstream (a trained
head, a validated rule, or a flag) — that's train_taste.py's job, decided by
how much data each taste actually has.

Inputs (download what you have; loaders skip cleanly if a file is absent):
  - ChemTastesDB_database.xlsx   Zenodo, CC-BY-4.0   10 classes, 4075 molecules
  - bittersweet/data/*.tsv       cosylab,  AGPL-3.0   sweet/bitter      [optional]
  - flavordb_taste.csv           FlavorDB export      SMILES + taste     [OBTAIN]
  - umami_list.csv               UMP442/BIOPEP-UWM    umami SMILES       [OBTAIN]
  - sweeteners_db.csv            Cheron SweetenersDB  sweetness intensity[OBTAIN]

Outputs:
  - taste_master.parquet     inchikey, smiles, {sweet,bitter,umami,sour,salty}=1/0/NaN,
                             multitaste flag, n_sources
  - sweet_intensity.parquet  smiles, log_sweetness   (if SweetenersDB present)

Multitaste handling: ChemTastesDB marks genuinely multi-taste molecules with a
single 'multitaste' class (it doesn't say WHICH tastes). We do two honest things
instead of guessing: (1) flag them multitaste=1, and (2) let the cross-source
merge RESOLVE their component tastes — if the same molecule is 'sweet' in one DB
and 'bitter' in another, it becomes sweet=1, bitter=1. So the more real sources
you add, the more multitaste rows resolve into concrete labels. (Adding more
verified databases beats having a model invent labels — model-guessed labels
reintroduce exactly the unverified-data problem that weakened SuperSweet.)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

BASIC = ["sweet", "bitter", "umami", "sour", "salty"]
INCLUDE_COSYLAB = True  # AGPL; set False for a strictly CC-BY build

CHEMTASTES_XLSX = Path("ChemTastesDB_database.xlsx")
COSYLAB_DIR = Path("bittersweet/data")
FLAVORDB_CSV = Path("flavordb_taste.csv")
UMAMI_CSV = Path("umami_list.csv")
SWEETENERS_CSV = Path("sweeteners_db.csv")


def canon(smiles):
    if not isinstance(smiles, str):
        return None, None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    try:
        return Chem.MolToInchiKey(mol), Chem.MolToSmiles(mol)
    except Exception:
        return None, None


def _find(df, candidates):
    cols = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    raise KeyError(f"none of {candidates} in {list(df.columns)}  [VERIFY]")


def chemtastes_labels(cls):
    """One ChemTastesDB class -> per-taste 1/0/NaN + multitaste flag."""
    out = {t: np.nan for t in BASIC}
    multitaste = 0.0
    c = str(cls).strip().lower()
    if c in BASIC:
        for t in BASIC:
            out[t] = 1.0 if t == c else 0.0
    elif c == "tasteless":
        for t in BASIC:
            out[t] = 0.0
    elif c.startswith("non-") and c[4:] in BASIC:
        out[c[4:]] = 0.0
    elif c == "multitaste":
        multitaste = 1.0           # resolved (if possible) via the merge below
    # 'miscellaneous' -> all NaN
    return out, multitaste


def _row(ik, cs, source, labels=None, multitaste=0.0):
    rec = {"inchikey": ik, "smiles": cs, "source": source, "multitaste": multitaste}
    rec.update({b: np.nan for b in BASIC})
    if labels:
        rec.update(labels)
    return rec


def load_chemtastes(path):
    if not path.exists():
        print(f"  [skip] {path}")
        return pd.DataFrame()
    df = pd.read_excel(path)
    sc = _find(df, ["canonical smiles", "smiles"])
    cc = _find(df, ["class taste", "taste class", "class", "taste"])
    rows = []
    for _, r in df.iterrows():
        ik, cs = canon(r[sc])
        if ik is None:
            continue
        labels, mt = chemtastes_labels(r[cc])
        rows.append(_row(ik, cs, "chemtastes", labels, mt))
    print(f"  chemtastes: {len(rows)}")
    return pd.DataFrame(rows)


def load_cosylab(folder):
    if not INCLUDE_COSYLAB or not folder.exists():
        print("  [skip] cosylab")
        return pd.DataFrame()
    rows = []
    for taste in ("bitter", "sweet"):
        for split in ("train", "test"):
            f = folder / f"{taste}-{split}.tsv"
            if not f.exists():
                continue
            t = pd.read_csv(f, sep="\t")
            sc = _find(t, ["smiles"])
            lc = _find(t, ["target", "label", "class"])
            for _, r in t.iterrows():
                ik, cs = canon(r[sc])
                if ik:
                    rows.append(_row(ik, cs, f"cosylab-{taste}", {taste: float(int(r[lc]))}))
    print(f"  cosylab: {len(rows)}")
    return pd.DataFrame(rows)


def load_single_taste_csv(path, taste, source):
    """Generic loader: a CSV of SMILES known to be `taste` (positives only)."""
    if not path.exists():
        print(f"  [skip] {source}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    sc = _find(df, ["smiles", "canonical smiles"])
    rows = []
    for _, r in df.iterrows():
        ik, cs = canon(r[sc])
        if ik:
            rows.append(_row(ik, cs, source, {taste: 1.0}))
    print(f"  {source}: {len(rows)}")
    return pd.DataFrame(rows)


def load_flavordb(path):
    """FlavorDB export with a taste column mapped onto basic tastes. [VERIFY columns]"""
    if not path.exists():
        print("  [skip] flavordb")
        return pd.DataFrame()
    df = pd.read_csv(path)
    sc = _find(df, ["smiles", "canonical smiles"])
    tc = _find(df, ["taste", "tastes", "flavor_profile"])
    rows = []
    for _, r in df.iterrows():
        ik, cs = canon(r[sc])
        if ik is None:
            continue
        tastes = str(r[tc]).lower()
        labels = {b: 1.0 for b in BASIC if b in tastes}  # multi-label friendly
        if labels:
            rows.append(_row(ik, cs, "flavordb", labels))
    print(f"  flavordb: {len(rows)}")
    return pd.DataFrame(rows)


def reconcile(df):
    def agg(s):
        v = s.dropna().values
        if (v == 1).any():
            return 1.0
        if (v == 0).any():
            return 0.0
        return np.nan
    g = df.groupby("inchikey")
    out = g.agg({"smiles": "first", "multitaste": "max",
                 **{t: agg for t in BASIC}})
    out["n_sources"] = g["source"].nunique()
    return out.reset_index()


def build_intensity(path):
    if not path.exists():
        print(f"  [skip] {path} (sweetness intensity)")
        return None
    df = pd.read_csv(path)
    sc = _find(df, ["smiles", "canonical smiles"])
    vc = _find(df, ["relative_sweetness", "log_sweetness", "sweetness", "rs"])
    rows = []
    for _, r in df.iterrows():
        _, cs = canon(r[sc])
        if cs:
            rows.append({"smiles": cs, "log_sweetness": float(r[vc])})
    print(f"  sweetness-intensity: {len(rows)}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("loading sources:")
    parts = [
        load_chemtastes(CHEMTASTES_XLSX),
        load_cosylab(COSYLAB_DIR),
        load_flavordb(FLAVORDB_CSV),
        load_single_taste_csv(UMAMI_CSV, "umami", "ump442"),
    ]
    parts = [p for p in parts if not p.empty]
    merged = reconcile(pd.concat(parts, ignore_index=True))
    merged.to_parquet("taste_master.parquet")

    print("\ntaste_master.parquet written")
    for t in BASIC:
        print(f"  {t:7s} pos={int((merged[t]==1).sum()):5d}  neg={int((merged[t]==0).sum()):5d}")
    mt = merged["multitaste"] == 1
    resolved = mt & merged[BASIC].eq(1).any(axis=1)
    print(f"  multitaste rows: {int(mt.sum())}  | resolved to concrete tastes via merge: {int(resolved.sum())}")
    print(f"  unique molecules: {len(merged)}")

    inten = build_intensity(SWEETENERS_CSV)
    if inten is not None:
        inten.to_parquet("sweet_intensity.parquet")
        print("sweet_intensity.parquet written")
