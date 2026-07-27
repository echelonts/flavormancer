"""
build_spectra.py — record which spectra PubChem has for each molecule (availability flags).

We don't rehost spectra (most are third-party — NIST/SDBS/Wiley — and NIST is individual-use).
But PubChem's *annotation* of WHICH spectra exist is public-domain metadata, and it's genuinely
useful: the read can show "PubChem has MS · IR · NMR" before the user clicks out. This crawls the
compound "Spectral Information" section per molecule and stores boolean flags in spectra.parquet
(inchikey, has_ms, has_ir, has_nmr, has_uv, has_raman).

Reuses the rate-limited public-domain PubChem client from build_properties. Resumable: skips
InChIKeys already in spectra.parquet.

Usage: python build_spectra.py                 # all_molecules.parquet -> spectra.parquet
       python build_spectra.py --molecules X   # any smiles/inchikey table
"""
import argparse
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from build_properties import _BASE, _cid, _get, load_keys

RDLogger.DisableLog("rdApp.*")
OUT = "spectra.parquet"

# map a flag to the PubChem sub-heading substrings that imply it
SPECTRA = {
    "has_ms": ["mass spectrometry", "ms-ms", "gc-ms", "lc-ms", "other ms"],
    "has_ir": ["ir spectra", "ftir", "atr-ir", "vapor phase ir", "infrared"],
    "has_nmr": ["nmr"],
    "has_uv": ["uv spectra", "uv-vis"],
    "has_raman": ["raman"],
}


def _headings(obj, out):
    if isinstance(obj, dict):
        h = obj.get("TOCHeading")
        if isinstance(h, str):
            out.add(h.lower())
        for v in obj.values():
            _headings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _headings(v, out)


def spectra_for(cid):
    d = _get(f"{_BASE}/pug_view/data/compound/{cid}/JSON?heading=Spectral+Information")
    heads = set()
    if d:
        _headings(d, heads)
    flags = {k: any(any(sub in h for h in heads) for sub in subs) for k, subs in SPECTRA.items()}
    flags["any"] = any(flags.values())
    return flags


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PubChem spectral-availability flags (public-domain metadata).")
    ap.add_argument("--molecules", default="all_molecules.parquet")
    a = ap.parse_args()
    keys = load_keys(a.molecules)
    done = set(pd.read_parquet(OUT)["inchikey"]) if Path(OUT).exists() else set()
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} molecules; {len(done)} already checked; {len(todo)} to crawl", flush=True)

    CHUNK, rows = 100, []
    for i, ik in enumerate(todo):
        cid = _cid(ik)
        f = spectra_for(cid) if cid else {k: False for k in (*SPECTRA, "any")}
        rows.append({"inchikey": ik, **{k: bool(f.get(k)) for k in (*SPECTRA, "any")}})
        if (i + 1) % CHUNK == 0 or (i + 1) == len(todo):
            new = pd.DataFrame(rows)
            if Path(OUT).exists():
                new = pd.concat([pd.read_parquet(OUT), new], ignore_index=True).drop_duplicates("inchikey")
            new.to_parquet(OUT)
            rows = []
            print(f"  {i + 1}/{len(todo)}; {int(new['any'].sum())}/{len(new)} have any spectra", flush=True)
    print(f"done -> {OUT}")
