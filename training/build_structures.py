"""
build_structures.py — backfill SMILES for molecules we only know by InChIKey.

The flavor map needs a structure (SMILES -> fingerprint) to place a point, but ~2,000 GRAS
reference molecules are keyed by InChIKey only. PubChem resolves a canonical SMILES from an
InChIKey (public-domain identity), so this pulls them and writes structures.parquet
(inchikey, smiles). build_flavor_map._all_structures() globs any parquet with a `smiles`
column, so once this file exists the next map rebuild includes these molecules automatically —
closing the map to the full unique universe.

Reuses the rate-limited PubChem client from build_properties. Resumable: skips InChIKeys that
already have a SMILES (in structures.parquet or any dataset that already carries one).

Usage: python build_structures.py            # all_molecules.parquet -> structures.parquet
"""
import glob
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from build_properties import _cid, _get, _BASE

RDLogger.DisableLog("rdApp.*")
OUT = "structures.parquet"


def _known_keys():
    """Full InChIKeys we already have a structure for (from any dataset's SMILES, or a prior run)."""
    known = set()
    for f in glob.glob("*.parquet"):
        try:
            d = pd.read_parquet(f)
        except Exception:  # noqa: BLE001
            continue
        if "smiles" in d.columns:
            for s in d["smiles"].dropna().unique():
                m = Chem.MolFromSmiles(str(s))
                if m is not None:
                    known.add(Chem.MolToInchiKey(m))
    return known


def _smiles_for_cid(cid):
    d = _get(f"{_BASE}/pug/compound/cid/{cid}/property/SMILES/JSON")
    props = ((d or {}).get("PropertyTable", {}).get("Properties") or [{}])[0]
    return props.get("SMILES") or props.get("ConnectivitySMILES")


if __name__ == "__main__":
    keys = list(pd.read_parquet("all_molecules.parquet")["inchikey"].dropna().unique())
    known = _known_keys()
    if Path(OUT).exists():
        known |= set(pd.read_parquet(OUT)["inchikey"].dropna())
    todo = [k for k in keys if k not in known]
    print(f"{len(keys)} molecules; {len(known)} already have a structure; {len(todo)} to resolve")

    CHUNK, rows = 100, []
    for i, ik in enumerate(todo):
        cid = _cid(ik)
        smi = _smiles_for_cid(cid) if cid else None
        if smi and Chem.MolFromSmiles(smi) is not None:
            rows.append({"inchikey": ik, "smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smi))})
        if (i + 1) % CHUNK == 0 or (i + 1) == len(todo):
            new = pd.DataFrame(rows, columns=["inchikey", "smiles"])
            if Path(OUT).exists():
                new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True).drop_duplicates("inchikey")
            new.to_parquet(OUT)
            rows = []
            print(f"  {i + 1}/{len(todo)} tried; structures.parquet -> {len(new)} resolved", flush=True)
    print(f"done -> {OUT}")
