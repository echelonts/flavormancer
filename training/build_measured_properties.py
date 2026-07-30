"""build_measured_properties.py — crawl EXPERIMENTAL boiling/melting points from PubChem PUG-View.

`build_properties.py` uses PubChem's property table, which only carries COMPUTED values (MW, logP,
TPSA). Measured boiling and melting points live somewhere else entirely: in PUG-View, under the
Experimental Properties section, sourced from HSDB and other public-domain records. That is why
73% of the enrichment table had no measured BP despite every one of those molecules having a
PubChem record — we had never asked the endpoint that holds them.

Values arrive as free text written by whoever recorded the measurement, so parsing is the real
work here: "246 °C", "115-116 °C at 12 mm Hg", "-11.5 °C", "410 °F". We take the first plausible
Celsius figure, convert Fahrenheit, take the midpoint of a range, and DROP anything reported at
reduced pressure — a boiling point at 12 mmHg is not comparable to one at atmospheric and
silently mixing them would corrupt the volatility ordering the formulation studio relies on.

Public-domain PubChem/HSDB data, cached locally.

    python build_measured_properties.py            # -> measured_properties.parquet
    python build_measured_properties.py --limit 50 # smoke test
"""
import argparse
import contextlib
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem

ENRICH = "master_enrichment.parquet"
OUT = "measured_properties.parquet"
PAUSE = 0.25          # PUG-View is heavier than the property table; stay well under the guidance
TIMEOUT = 10

# "115-116 °C at 12 mm Hg" — reduced-pressure boiling points are not comparable to atmospheric
# ones and must not be mixed into the same column.
_REDUCED = re.compile(r"\bat\s+[\d.]+\s*(mm\s*hg|torr|kpa|mbar|hpa)\b", re.IGNORECASE)
_TEMP = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-\s*(-?\d+(?:\.\d+)?)\s*)?°?\s*([CF])\b")


def _parse_temp_c(text):
    """First plausible temperature in the string, as Celsius. None if unusable."""
    if not text or _REDUCED.search(text):
        return None
    m = _TEMP.search(text)
    if not m:
        return None
    lo = float(m.group(1))
    val = (lo + float(m.group(2))) / 2 if m.group(2) else lo   # midpoint of a reported range
    if m.group(3).upper() == "F":
        val = (val - 32) * 5 / 9
    return round(val, 1) if -200 <= val <= 800 else None       # outside this, it isn't a BP or MP


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001 — not found / throttled / timeout
        return None


def _cid(smiles):
    d = _get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
             f"{urllib.parse.quote(smiles)}/cids/JSON")
    with contextlib.suppress(Exception):
        return d["IdentifierList"]["CID"][0]
    return None


def _experimental(cid, heading):
    """All free-text values PubChem lists under one Experimental Properties heading."""
    d = _get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
             f"?heading={urllib.parse.quote(heading)}")
    if not d:
        return []
    out, stack = [], [d]
    while stack:                      # the response nests sections arbitrarily deep
        node = stack.pop()
        if isinstance(node, dict):
            for info in node.get("Information", []) or []:
                for sw in (info.get("Value", {}) or {}).get("StringWithMarkup", []) or []:
                    if sw.get("String"):
                        out.append(sw["String"])
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(node)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="only crawl N molecules (smoke test)")
    a = ap.parse_args()

    enrich = pd.read_parquet(ENRICH)
    done = set()
    with contextlib.suppress(Exception):
        done = set(pd.read_parquet(OUT)["inchikey_skel"])

    todo = {}
    for _, r in enrich.iterrows():
        skel, smi = r.get("inchikey_skel"), r.get("smiles")
        if not isinstance(skel, str) or not isinstance(smi, str) or skel in done:
            continue
        # only chase molecules that are actually missing a measured value
        if pd.notna(r.get("boiling_point_c")) and pd.notna(r.get("melting_point_c")):
            continue
        todo.setdefault(skel, smi)
    if a.limit:
        todo = dict(list(todo.items())[:a.limit])

    print(f"{len(done)} already crawled; {len(todo)} to go.", flush=True)
    rows, got = [], 0
    for i, (skel, smi) in enumerate(todo.items(), 1):
        mol = Chem.MolFromSmiles(smi)
        cid = _cid(Chem.MolToSmiles(mol)) if mol else None
        if cid:
            bp = next((v for v in (_parse_temp_c(t) for t in _experimental(cid, "Boiling Point"))
                       if v is not None), None)
            mp = next((v for v in (_parse_temp_c(t) for t in _experimental(cid, "Melting Point"))
                       if v is not None), None)
            if bp is not None or mp is not None:
                rows.append({"inchikey_skel": skel, "boiling_point_c": bp, "melting_point_c": mp})
                got += 1
        if i % 100 == 0:
            print(f"  {i}/{len(todo)} crawled, {got} with a measured value", flush=True)
            _save(rows)
        time.sleep(PAUSE)
    _save(rows)
    print(f"Done. {got}/{len(todo)} gained a measured property -> {OUT}")


def _save(rows):
    if not rows:
        return
    new = pd.DataFrame(rows).drop_duplicates("inchikey_skel")
    with contextlib.suppress(Exception):
        new = pd.concat([pd.read_parquet(OUT), new]).drop_duplicates("inchikey_skel", keep="last")
    new.to_parquet(OUT)


if __name__ == "__main__":
    main()
