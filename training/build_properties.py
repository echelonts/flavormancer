"""
build_properties.py — measured properties + identity from PubChem.

PubChem experimental properties AND compound identity are **public domain** (no commercial
restriction), so this is a commercial-clean source. For each molecule it pulls the common
name (Title) and IUPAC name from the property endpoint, plus the experimental Boiling Point
(and, best-effort, Vapor Pressure) from PUG-View, parses to °C / Pa, and writes
properties.parquet — the enrichment table predict.py reads for MEASURED volatility and the
demo reads for instant offline names. We use a measured lookup for BP on purpose: structure-
based BP (Joback) was evaluated and rejected (~90 °C error), so a number is only ever
reported when it's a real measurement. Names default to live PubChem in the app; precomputing
them here makes the substitution / palette / neighbor reads instant for the whole labeled set.

Usage:
  python build_properties.py                                  # default: taste_master.parquet
  python build_properties.py --molecules flavor_volatiles.csv # ANY SMILES/InChIKey set
  python build_properties.py "O=Cc1ccc(O)c(OC)c1"             # test mode: print props for SMILES
The output (properties.parquet) is MERGED when it already exists, so you can accumulate
sets — taste molecules, aroma / GS-LF molecules, a customer's list — by pointing
--molecules at each in turn. (PubChem physical properties are public domain, so building
BP/VP for GS-LF molecules is clean even in the academic edition; only the GS-LF *odor
labels* are NonCommercial, not these properties.)
"""
import argparse
import json
import re
import sys
import time
import urllib.error
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
_PRESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm\s?hg|torr|kpa|hpa|mbar|atm|bar)", re.I)
_P_TO_MMHG = {"mmhg": 1.0, "torr": 1.0, "kpa": 7.50062, "hpa": 0.750062,
              "mbar": 0.750062, "atm": 760.0, "bar": 750.062}


_last_req = [0.0]
_MIN_INTERVAL = 0.22  # ~4.5 req/s — under PubChem's 5/s ceiling. Steady pacing avoids the
#                       503 throttling that a 4-calls-per-molecule burst triggers; once
#                       throttled, every retry's backoff costs far more than this spacing.


def _get(url):
    for attempt in range(5):
        wait = _MIN_INTERVAL - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:  # 503 = throttled: brief backoff, then retry
            # keep the 503 backoff GENTLE (capped): a transient throttle should cost a second
            # or two, not stall the crawl — an over-aggressive backoff turns a blip into a hang
            time.sleep(min((attempt + 1) * (1.0 if e.code == 503 else 0.6), 4.0))
        except Exception:  # noqa: BLE001 — network/parse; retry then give up
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


def _pressure_mmhg(s):
    """Measurement pressure stated in the string; assume atmospheric (760) if none."""
    m = _PRESS_RE.search(s)
    if not m:
        return 760.0
    return float(m.group(1)) * _P_TO_MMHG.get(m.group(2).lower().replace(" ", ""), 1.0)


def parse_bp(strings):
    """Boiling point in °C, PREFERRING an atmospheric (~760 mmHg) reading. Returns
    (bp_c, pressure_mmhg): pressure is None when atmospheric; otherwise the reading is at
    a reduced/elevated pressure and that pressure is reported, so the value stays honest."""
    readings = []  # (temp_c, pressure_mmhg)
    for s in strings:
        press = _pressure_mmhg(s)
        for m in _TEMP_RE.finditer(s):
            # skip a temperature that's a measurement CONDITION ("... at 20 °C")
            if s[max(0, m.start() - 4):m.start()].lower().rstrip().endswith("at"):
                continue
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            t = (lo + hi) / 2
            if m.group(3).upper() == "F":
                t = (t - 32) * 5 / 9
            if -50 <= t <= 600:
                readings.append((round(t, 1), press))
    if not readings:
        return None, None
    atm = sorted(t for t, p in readings if 700 <= p <= 820)
    if atm:
        return atm[len(atm) // 2], None
    # only non-atmospheric data: report the highest-pressure reading (closest to 1 atm)
    t, p = max(readings, key=lambda r: r[1])
    return t, round(p, 1)


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


def _names_for_cid(cid):
    """(common Title, IUPAC) from PubChem's property endpoint — public-domain identity."""
    d = _get(f"{_BASE}/pug/compound/cid/{cid}/property/Title,IUPACName/JSON")
    p = ((d or {}).get("PropertyTable", {}).get("Properties") or [{}])[0]
    return p.get("Title"), p.get("IUPACName")


def fetch_props(inchikey):
    cid = _cid(inchikey)
    if not cid:
        return None, None, None, None, None
    bp, bp_press = parse_bp(_heading_strings(cid, "Boiling Point"))
    vp = parse_vp_pa(_heading_strings(cid, "Vapor Pressure"))
    common, iupac = _names_for_cid(cid)
    return bp, bp_press, vp, common, iupac


def load_keys(path):
    """Unique InChIKeys from a CSV/parquet with a 'smiles' (preferred) or 'inchikey' column."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    sc = cols.get("smiles") or cols.get("canonical smiles") or cols.get("isomeric smiles")
    if sc:
        keys = []
        for s in df[sc].dropna():
            m = Chem.MolFromSmiles(str(s))
            if m is not None:
                keys.append(Chem.MolToInchiKey(m))
        return sorted(set(keys))
    if "inchikey" in cols:
        return sorted(set(df[cols["inchikey"]].dropna().astype(str)))
    raise SystemExit(f"{path}: need a 'smiles' or 'inchikey' column")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measured BP/VP from PubChem (public domain).")
    ap.add_argument("--molecules", help="CSV/parquet with a 'smiles' or 'inchikey' column "
                                        "(default: taste_master.parquet)")
    ap.add_argument("--out", default="properties.parquet", help="output table (merged if it exists)")
    ap.add_argument("smiles", nargs="*", help="SMILES to test-print (no build)")
    a = ap.parse_args()

    if a.smiles:  # test mode
        for smi in a.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"{smi}: unparseable")
                continue
            ik = Chem.MolToInchiKey(mol)
            bp, bp_press, vp, common, iupac = fetch_props(ik)
            cond = f" @{bp_press}mmHg" if bp_press else ""
            print(f"{smi:32s} ik={ik}  name={common!r}  iupac={iupac!r}  "
                  f"bp_c={bp}{cond}  vp_pa={vp}")
            time.sleep(0.3)
        sys.exit(0)

    src = a.molecules or "taste_master.parquet"
    if not Path(src).exists():
        print(f"{src} not found")
        sys.exit(1)
    keys = load_keys(src)
    cols = ["inchikey", "common_name", "iupac_name", "boiling_point_c",
            "boiling_point_pressure_mmhg", "vapor_pressure_pa"]

    def checkpoint(rows):
        """Merge new rows into the output table and persist — called every CHUNK so a
        multi-hour crawl survives interruption and can resume where it left off."""
        new = pd.DataFrame(rows, columns=cols)
        if Path(a.out).exists():  # accumulate across molecule sets / prior checkpoints
            old = pd.read_parquet(a.out)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates("inchikey", keep="last")
        return new.reindex(columns=cols)  # stable schema as older tables gain name columns

    done = set()  # resume: skip molecules already carrying a name in the existing table
    if Path(a.out).exists():
        ex = pd.read_parquet(a.out)
        if "common_name" in ex.columns:
            done = set(ex.loc[ex["common_name"].notna(), "inchikey"])
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} molecules in {src}; {len(done)} already named, {len(todo)} to fetch")

    CHUNK, rows = 100, []
    for i, ik in enumerate(todo):
        bp, bp_press, vp, common, iupac = fetch_props(ik)
        if bp is not None or vp is not None or common or iupac:
            rows.append({"inchikey": ik, "common_name": common, "iupac_name": iupac,
                         "boiling_point_c": bp, "boiling_point_pressure_mmhg": bp_press,
                         "vapor_pressure_pa": vp})
        if (i + 1) % CHUNK == 0 or (i + 1) == len(todo):
            merged = checkpoint(rows)
            merged.to_parquet(a.out)
            rows = []
            print(f"  {i + 1}/{len(todo)} fetched; checkpoint -> {len(merged)} rows, "
                  f"{int(merged['common_name'].notna().sum())} named, "
                  f"{int(merged['boiling_point_c'].notna().sum())} with BP", flush=True)
    final = pd.read_parquet(a.out)
    print(f"{a.out}: {len(final)} molecules total "
          f"(names: {int(final['common_name'].notna().sum())}, "
          f"BP: {int(final['boiling_point_c'].notna().sum())}); public-domain PubChem data")
