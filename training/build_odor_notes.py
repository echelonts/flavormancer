"""
build_odor_notes.py — documented odor + odor threshold from PubChem (public-domain sources ONLY).

PubChem carries an "Odor" annotation (free-text descriptions) and an "Odor Threshold"
annotation (detection concentration in ppm/ppb) for many molecules. We keep ONLY notes whose
source is public domain — HSDB and Haz-Map (both NIH/NLM) and CAMEO Chemicals (NOAA/EPA) — and
EXPLICITLY DROP any proprietary flavor source (The Good Scents Company, Leffingwell, Flavornet,
FEMA) so the result stays commercial-clean, consistent with the rest of the project.

What this IS: a cited LOOKUP of real, documented odor + a parsed detection threshold (ppm) —
honest, attributable, commercial-clean. The odor text is also a future training corpus for a
presence/absence descriptor model. What it is NOT: a trained model, and NOT a substitute for
licensed (PMP 2001) or customer panel data — the descriptions are free text with no controlled
vocabulary or intensity, and published thresholds vary widely between sources (we keep the raw
note alongside the parsed number so the range stays visible).

Usage:
  python build_odor_notes.py                                  # default: flavor_volatiles.csv
  python build_odor_notes.py --molecules taste_master.parquet  # any SMILES/InChIKey set
  python build_odor_notes.py "O=Cc1ccc(O)c(OC)c1"             # test mode: print for a SMILES
Output odor_notes.parquet (inchikey, odor, odor_source, odor_threshold_ppm,
odor_threshold_note, odor_threshold_source) is MERGED when it already exists.
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# reuse the rate-limited fetcher + InChIKey->CID + molecule-set loader
from build_properties import _BASE, _cid, _get, load_keys

# keep only public-domain sources; drop proprietary flavor databases outright
PUBLIC_DOMAIN = ("hazardous substances data bank", "hsdb", "haz-map", "cameo chemicals")
BLOCK = ("good scents", "goodscents", "tgsc", "leffingwell", "flavornet", "flavordb", "fema")
_PPM_RE = re.compile(r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(ppb|ppm)", re.I)
_LOW_RE = re.compile(r"threshold low:\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\[?\s*(ppb|ppm)", re.I)


def _pd_source(src):
    """True if a source name is public-domain and not a blocked proprietary flavor DB."""
    sl = (src or "").lower()
    return any(p in sl for p in PUBLIC_DOMAIN) and not any(b in sl for b in BLOCK)


def _source_names(view):
    refs = {}

    def walk(o):
        if isinstance(o, dict):
            if "ReferenceNumber" in o and "SourceName" in o:
                refs[o["ReferenceNumber"]] = o["SourceName"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(view)
    return refs


def _strings_for(view, toc):
    """[(source, string)] for every Information item under a given TOCHeading, PD-filtered."""
    refs = _source_names(view)
    out = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("TOCHeading") == toc:
                for info in o.get("Information", []):
                    src = (refs.get(info.get("ReferenceNumber")) or "").strip()
                    if not _pd_source(src):
                        continue
                    for s in info.get("Value", {}).get("StringWithMarkup", []) or []:
                        t = " ".join((s.get("String") or "").split()).strip()
                        if t:
                            out.append((src, t))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(view)
    return out


def _parse_threshold_ppm(pairs):
    """A representative DETECTION threshold in ppm. The 'Odor Threshold' section mixes true
    detection values with irritation/occupational numbers, so PREFER the structured Haz-Map
    'Odor Threshold Low: N [ppm]' field (the reliable detection value); only if none exists
    fall back to a ppm/ppb value near 'detection', then to any. Median across sources (they
    vary by orders of magnitude); convert ppb->ppm. None when no numeric value is present."""
    lows = []
    for _, s in pairs:
        for m in _LOW_RE.finditer(s):
            ppm = float(m.group(1)) * (0.001 if m.group(2).lower() == "ppb" else 1.0)
            if ppm > 0:
                lows.append(ppm)
    if lows:
        return round(statistics.median(lows), 6)
    det, other = [], []
    for _, s in pairs:
        near = "detection" in s.lower()
        for m in _PPM_RE.finditer(s):
            ppm = float(m.group(1)) * (0.001 if m.group(2).lower() == "ppb" else 1.0)
            if ppm > 0:
                (det if near else other).append(ppm)
    pool = det or other
    return round(statistics.median(pool), 6) if pool else None


def fetch_record(inchikey):
    """dict of odor + threshold for a molecule, or None when nothing public-domain is found."""
    cid = _cid(inchikey)
    if not cid:
        return None
    odor_pairs = _strings_for(_get(f"{_BASE}/pug_view/data/compound/{cid}/JSON?heading=Odor"),
                              "Odor")
    thr_pairs = _strings_for(
        _get(f"{_BASE}/pug_view/data/compound/{cid}/JSON?heading=Odor+Threshold"),
        "Odor Threshold")
    # odor: keep every note (training corpus), shortest first so punchy descriptors lead
    odor_notes = sorted({s for _, s in odor_pairs}, key=len)
    odor = "\n".join(odor_notes) or None
    odor_src = "; ".join(sorted({src for src, _ in odor_pairs})) or None
    thr_ppm = _parse_threshold_ppm(thr_pairs)
    thr_note = "\n".join(dict.fromkeys(s for _, s in thr_pairs)) or None  # ordered-unique
    thr_src = "; ".join(sorted({src for src, _ in thr_pairs})) or None
    if not (odor or thr_ppm or thr_note):
        return None
    return {"inchikey": inchikey, "odor": odor, "odor_source": odor_src,
            "odor_threshold_ppm": thr_ppm, "odor_threshold_note": thr_note,
            "odor_threshold_source": thr_src}


COLS = ["inchikey", "odor", "odor_source", "odor_threshold_ppm",
        "odor_threshold_note", "odor_threshold_source"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Public-domain documented odor + threshold.")
    ap.add_argument("--molecules", help="CSV/parquet with 'smiles' or 'inchikey' "
                                        "(default: flavor_volatiles.csv)")
    ap.add_argument("--out", default="odor_notes.parquet", help="output (merged if it exists)")
    ap.add_argument("smiles", nargs="*", help="SMILES to test-print (no build)")
    a = ap.parse_args()

    if a.smiles:  # test mode
        for smi in a.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"{smi}: unparseable")
                continue
            r = fetch_record(Chem.MolToInchiKey(mol)) or {}
            print(f"{smi:30s} thr_ppm={r.get('odor_threshold_ppm')}  "
                  f"odor={(r.get('odor') or '').splitlines()[:2]}  "
                  f"thr_note={(r.get('odor_threshold_note') or '')[:80]!r}")
        sys.exit(0)

    src = a.molecules or "flavor_volatiles.csv"
    if not Path(src).exists():
        print(f"{src} not found")
        sys.exit(1)
    keys = load_keys(src)
    rows = []
    for i, ik in enumerate(keys):
        rec = fetch_record(ik)
        if rec:
            rows.append(rec)
        if (i + 1) % 50 == 0 or (i + 1) == len(keys):
            new = pd.DataFrame(rows, columns=COLS)
            if Path(a.out).exists():  # checkpoint + accumulate across sets
                old = pd.read_parquet(a.out)
                new = pd.concat([old, new], ignore_index=True).drop_duplicates("inchikey", keep="last")
                new = new.reindex(columns=COLS)
            new.to_parquet(a.out)
            odor_n = int(new["odor"].notna().sum())
            thr_n = int(new["odor_threshold_ppm"].notna().sum())
            print(f"  {i + 1}/{len(keys)} processed; checkpoint -> {len(new)} rows "
                  f"({odor_n} odor, {thr_n} threshold)", flush=True)
    final = pd.read_parquet(a.out)
    print(f"{a.out}: {len(final)} molecules "
          f"({int(final['odor'].notna().sum())} documented odor, "
          f"{int(final['odor_threshold_ppm'].notna().sum())} odor threshold); "
          f"public-domain (HSDB/Haz-Map/CAMEO)")
