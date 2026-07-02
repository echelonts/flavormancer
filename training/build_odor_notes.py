"""
build_odor_notes.py — documented odor descriptions from PubChem (public-domain sources ONLY).

PubChem carries an "Odor" annotation for many molecules. We keep ONLY notes whose source is
public domain — HSDB (NIH/NLM Hazardous Substances Data Bank) and CAMEO Chemicals (NOAA/EPA) —
and EXPLICITLY DROP any proprietary flavor source (The Good Scents Company, Leffingwell,
Flavornet, FEMA) so the result stays commercial-clean, consistent with the rest of the project.

What this IS: a cited LOOKUP of real, documented odor descriptions — honest and attributable,
a real replacement for hand-written illustrative descriptors. What it is NOT: a trained aroma
model, and NOT a substitute for licensed (PMP 2001) or customer panel data — the text is free-
form, coverage is sparse and skewed to industrially-notable chemicals, and there's no
controlled descriptor vocabulary or intensity. It's the honest public-data floor for aroma;
the real model still "comes with your data."

Usage:
  python build_odor_notes.py                                 # default: flavor_volatiles.csv
  python build_odor_notes.py --molecules taste_master.parquet # any SMILES/InChIKey set
  python build_odor_notes.py "O=Cc1ccc(O)c(OC)c1"            # test mode: print odor for SMILES
Output odor_notes.parquet (inchikey, odor, odor_source) is MERGED when it already exists.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem

# reuse the rate-limited fetcher + InChIKey->CID + molecule-set loader
from build_properties import _BASE, _cid, _get, load_keys

# keep only public-domain government sources; drop proprietary flavor databases outright
PUBLIC_DOMAIN = ("hazardous substances data bank", "hsdb", "cameo chemicals")
BLOCK = ("good scents", "goodscents", "tgsc", "leffingwell", "flavornet", "flavordb", "fema")


def _source_names(view):
    """ReferenceNumber -> SourceName map for a PUG-View record."""
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


def fetch_odor(inchikey):
    """(joined public-domain odor notes, joined source names) or (None, None)."""
    cid = _cid(inchikey)
    if not cid:
        return None, None
    view = _get(f"{_BASE}/pug_view/data/compound/{cid}/JSON?heading=Odor")
    if not view:
        return None, None
    refs = _source_names(view)
    notes, srcs = [], set()

    def walk(o):
        if isinstance(o, dict):
            if o.get("TOCHeading") == "Odor":
                for info in o.get("Information", []):
                    src = (refs.get(info.get("ReferenceNumber")) or "").strip()
                    sl = src.lower()
                    if not any(p in sl for p in PUBLIC_DOMAIN) or any(b in sl for b in BLOCK):
                        continue  # source not clean-public-domain -> skip this annotation
                    for s in info.get("Value", {}).get("StringWithMarkup", []) or []:
                        t = " ".join((s.get("String") or "").split()).strip()
                        if t and t not in notes:
                            notes.append(t)
                            srcs.add(src)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(view)
    if not notes:
        return None, None
    # keep EVERY note (this is also a future training corpus) — shortest first so the punchy
    # descriptors lead; newline-delimit so notes (which can contain ';') stay intact downstream.
    return "\n".join(sorted(notes, key=len)), "; ".join(sorted(srcs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Public-domain documented odor from PubChem.")
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
            odor, osrc = fetch_odor(Chem.MolToInchiKey(mol))
            print(f"{smi:32s} odor={odor!r}  src={osrc!r}")
        sys.exit(0)

    src = a.molecules or "flavor_volatiles.csv"
    if not Path(src).exists():
        print(f"{src} not found")
        sys.exit(1)
    keys = load_keys(src)
    cols = ["inchikey", "odor", "odor_source"]
    rows = []
    for i, ik in enumerate(keys):
        odor, osrc = fetch_odor(ik)
        if odor:
            rows.append({"inchikey": ik, "odor": odor, "odor_source": osrc})
        if (i + 1) % 50 == 0 or (i + 1) == len(keys):
            new = pd.DataFrame(rows, columns=cols)
            if Path(a.out).exists():  # checkpoint + accumulate across sets
                old = pd.read_parquet(a.out)
                new = pd.concat([old, new], ignore_index=True).drop_duplicates("inchikey", keep="last")
            new.to_parquet(a.out)
            print(f"  {i + 1}/{len(keys)} processed; checkpoint -> {len(new)} with documented odor",
                  flush=True)
    final = pd.read_parquet(a.out)
    print(f"{a.out}: {len(final)} molecules with public-domain documented odor (HSDB/CAMEO)")
