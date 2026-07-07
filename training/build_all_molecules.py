"""
build_all_molecules.py — the master unique-molecule list (every set, deduped).

Unions the InChIKeys of EVERY molecule Flavormancer knows — taste, aroma, odor, GRAS,
sweetness, documented taste, the curated flavor list — into one table so the enrichment
crawl (build_properties.py) covers the WHOLE universe, not just the taste-training subset.
GRAS is keyed by InChIKey only (no SMILES); PubChem still resolves names + properties from
an InChIKey, so those molecules enrich fine.

Output all_molecules.parquet has a single `inchikey` column (build_properties.load_keys reads
it directly). SMILES aren't needed for the crawl — it fetches by InChIKey — and carrying a
SMILES column would make load_keys drop the GRAS-only rows that have none.

Usage: python build_all_molecules.py            # -> all_molecules.parquet
"""
import glob
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

SKIP = {"all_molecules.parquet", "properties.parquet", "flavor_map.parquet"}  # derived/output


def keys_from(path):
    """Every InChIKey in a dataset — from a SMILES column (preferred, canonicalized) and/or an
    inchikey column (GRAS and similar structure-free reference sets)."""
    df = pd.read_parquet(path)
    cols = {c.lower().strip(): c for c in df.columns}
    out = set()
    sc = cols.get("smiles") or cols.get("canonical smiles") or cols.get("isomeric smiles")
    if sc:
        for s in df[sc].dropna().unique():
            m = Chem.MolFromSmiles(str(s))
            if m is not None:
                out.add(Chem.MolToInchiKey(m))
    if "inchikey" in cols:
        for ik in df[cols["inchikey"]].dropna().astype(str).unique():
            if len(ik) >= 14:  # a real InChIKey, not a skeleton fragment
                out.add(ik)
    return out


if __name__ == "__main__":
    # the curated flavor list is a CSV, not parquet — fold it in too
    all_keys, per = set(), {}
    sources = sorted(glob.glob("*.parquet")) + (["flavors.csv"] if Path("flavors.csv").exists() else [])
    for f in sources:
        if f in SKIP:
            continue
        try:
            if f.endswith(".csv"):
                df = pd.read_csv(f)
                ks = {Chem.MolToInchiKey(m) for s in df.get("smiles", [])
                      if (m := Chem.MolFromSmiles(str(s))) is not None}
            else:
                ks = keys_from(f)
        except Exception as e:  # noqa: BLE001 — skip an unreadable set, report it
            print(f"  ! {f}: {e}")
            continue
        per[f] = len(ks)
        all_keys |= ks
    for f, n in sorted(per.items(), key=lambda x: -x[1]):
        print(f"  {f:34s} {n:5d}")
    out = pd.DataFrame({"inchikey": sorted(all_keys)})
    out.to_parquet("all_molecules.parquet")
    print(f"\nall_molecules.parquet: {len(out)} unique molecules (by full InChIKey) across "
          f"{len(per)} datasets")
