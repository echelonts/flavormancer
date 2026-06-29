"""
build_properties.py — measured boiling point / vapor pressure from PubChem.

PubChem experimental properties are **public domain** (no commercial restriction), so
this is a commercial-clean source. It pulls each molecule's experimental Boiling Point
(and, best-effort, Vapor Pressure) from PUG-View, parses to °C / Pa, and writes
properties.parquet — the lookup table predict.py reads for MEASURED volatility. We use
a measured lookup here on purpose: structure-based BP (Joback) was evaluated and
rejected (~90 °C error), so a number is only ever reported when it's a real measurement.

Usage:
  python build_properties.py                       # build for all of taste_master.parquet
  python build_properties.py "O=Cc1ccc(O)c(OC)c1"  # test mode: print props for SMILES args
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem

_UA = {"User-Agent": "flavormancer-build-properties/1.0"}
_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest"
_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:to\s*(-?\d+(?:\.\d+)?)\s*)?°?\s*([CF])\b")
_VP_RE = re.compile(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(mm\s?hg|torr|kpa|hpa|pa|atm|bar)", re.I)
_VP_TO_PA = {"mmhg": 133.322, "torr": 133.322, "kpa": 1000.0, "hpa": 100.0,
             "pa": 1.0, "atm": 101325.0, "bar": 100000.0}


def _get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 — network/HTTP/parse; retry then give up
            time.sleep(0.6 * (attempt + 1))
    return None


def _strings(obj, out):
    if isinstance(obj, dict):
        if isinstance(obj.get("String"), str):
            out.append(obj["String"])
        for v in obj.values():
            _strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _strings(v, out)


def _cid(inchikey):
    d = _get(f"{_BASE}/pug/compound/inchikey/{urllib.parse.quote(inchikey)}/cids/JSON")
    cids = (d or {}).get("IdentifierList", {}).get("CID", [])
    return cids[0] if cids else None


def _heading_strings(cid, heading):
    d = _get(f"{_BASE}/pug_view/data/compound/{cid}/JSON?heading={urllib.parse.quote(heading)}")
    out = []
    if d:
        _strings(d, out)
    return out


def parse_bp_c(strings):
    """Median of all sane °C readings (converting °F, averaging ranges)."""
    vals = []
    for s in strings:
        for m in _TEMP_RE.finditer(s):
            # skip a temperature that's a measurement CONDITION ("... at 20 °C"),
            # not the boiling point itself
            if s[max(0, m.start() - 4):m.start()].lower().rstrip().endswith("at"):
                continue
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            t = (lo + hi) / 2
            if m.group(3).upper() == "F":
                t = (t - 32) * 5 / 9
            vals.append(t)
    vals = sorted(v for v in vals if -50 <= v <= 600)
    return round(vals[len(vals) // 2], 1) if vals else None


def parse_vp_pa(strings):
    vals = []
    for s in strings:
        m = _VP_RE.search(s)
        if not m:
            continue
        factor = _VP_TO_PA.get(m.group(2).lower().replace(" ", ""))
        if factor:
            vals.append(float(m.group(1)) * factor)
    vals = sorted(v for v in vals if v > 0)
    return round(vals[len(vals) // 2], 2) if vals else None


def fetch_props(inchikey):
    cid = _cid(inchikey)
    if not cid:
        return None, None
    return parse_bp_c(_heading_strings(cid, "Boiling Point")), \
        parse_vp_pa(_heading_strings(cid, "Vapor Pressure"))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:  # test mode
        for smi in args:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"{smi}: unparseable")
                continue
            ik = Chem.MolToInchiKey(mol)
            bp, vp = fetch_props(ik)
            print(f"{smi:32s} ik={ik}  bp_c={bp}  vp_pa={vp}")
            time.sleep(0.3)
        sys.exit(0)

    master = Path("taste_master.parquet")
    if not master.exists():
        print("taste_master.parquet not found — run build_taste_dataset.py first")
        sys.exit(1)
    keys = pd.read_parquet(master)["inchikey"].dropna().unique()
    rows = []
    for i, ik in enumerate(keys):
        bp, vp = fetch_props(ik)
        if bp is not None or vp is not None:
            rows.append({"inchikey": ik, "boiling_point_c": bp, "vapor_pressure_pa": vp})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(keys)} processed, {len(rows)} with measured props")
        time.sleep(0.3)
    pd.DataFrame(rows).to_parquet("properties.parquet")
    print(f"properties.parquet written: {len(rows)} molecules with measured BP/VP "
          f"(public-domain PubChem experimental data)")
