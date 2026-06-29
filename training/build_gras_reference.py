"""
build_gras_reference.py — GRAS / food-ingredient reference from the FDA SAF inventory.

The FDA "Substances Added to Food" (SAF, formerly EAFUS) inventory is a US-government
work in the **public domain** — commercial-clean. This downloads it (or reads a local
copy), extracts the CAS numbers, resolves them to InChIKey via PubChem (also public
domain), and writes gras_reference.parquet — the table predict.py cross-checks for
"is this a recognized food ingredient at all?" (a DEFENSIVE signal, never a clearance).

Usage:
  python build_gras_reference.py            # download SAF + build gras_reference.parquet
  python build_gras_reference.py saf.xls    # use a locally-downloaded SAF file
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

SAF_URL = ("https://www.cfsanappsexternal.fda.gov/scripts/fdcc/cfc/"
           "XMLService.cfm?method=downloadxls&set=FoodSubstances")
_UA = {"User-Agent": "flavormancer-build-gras/1.0"}
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def load_saf(path=None):
    """Return the SAF table. FDA serves it as cp1252 CSV behind a 4-line preamble."""
    if path and Path(path).exists():
        raw = Path(path).read_bytes()
    else:
        print("downloading FDA SAF (public domain)...")
        with urllib.request.urlopen(urllib.request.Request(SAF_URL, headers=_UA), timeout=90) as r:
            raw = r.read()
    lines = raw.decode("cp1252").splitlines()
    hdr = next(i for i, line in enumerate(lines) if line.startswith("CAS Reg No"))
    return pd.read_csv(StringIO("\n".join(lines[hdr:])), dtype=str)


def cas_list(df):
    """Clean, unique, validly-formatted CAS numbers from the first column."""
    out = []
    for v in df[df.columns[0]].dropna():
        c = v.strip().strip('"').strip()
        if _CAS_RE.match(c):
            out.append(c)
    return sorted(set(out))


def cas_to_inchikey(cas):
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{urllib.parse.quote(cas)}/property/InChIKey/JSON")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20) as r:
                d = json.load(r)
            return d["PropertyTable"]["Properties"][0]["InChIKey"]
        except Exception:  # noqa: BLE001 — not-found / network; skip this CAS
            time.sleep(0.4 * (attempt + 1))
    return None


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_saf(src)
    cas = cas_list(df)
    print(f"SAF rows: {len(df)}  valid CAS: {len(cas)}")
    keys, miss = [], 0
    for i, c in enumerate(cas):
        ik = cas_to_inchikey(c)
        if ik:
            keys.append(ik)
        else:
            miss += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(cas)} resolved — {len(keys)} hits, {miss} misses")
        time.sleep(0.25)
    pd.DataFrame({"inchikey": sorted(set(keys))}).to_parquet("gras_reference.parquet")
    print(f"gras_reference.parquet written: {len(set(keys))} unique food substances "
          f"(FDA SAF, public domain)")
