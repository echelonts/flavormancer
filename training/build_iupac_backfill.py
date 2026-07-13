"""build_iupac_backfill.py — fill in IUPAC names PubChem's property crawl missed.

The master enrichment table has a SMILES for every molecule but the offline name
table (properties.parquet) only carries an IUPAC name for the ones build_properties.py
happened to resolve. This backfills the gap: for every enrichment skeleton with no
IUPAC name yet, ask PubChem (public domain) for its IUPACName by SMILES and write the
results to iupac_backfill.parquet (inchikey_skel -> iupac_name), which app.py loads as a
fallback layer under properties.parquet.

Public-domain PubChem data; we cache locally, never rehost anything licensed.

    python build_iupac_backfill.py
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem

ENRICH = "master_enrichment.parquet"
PROPS = "properties.parquet"
OUT = "iupac_backfill.parquet"
PAUSE = 0.22          # ~4-5 req/s, under PubChem's guidance
TIMEOUT = 6


def _have_iupac_skeletons():
    """Skeletons that already have an IUPAC name (offline table + any prior backfill)."""
    have = set()
    for path, key in ((PROPS, "inchikey"), (OUT, "inchikey_skel")):
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 — file may not exist yet
            continue
        if "iupac_name" not in df.columns:
            continue
        for ik, u in zip(df[key], df["iupac_name"]):
            if isinstance(ik, str) and isinstance(u, str) and u:
                have.add(ik.split("-")[0])
    return have


def _pubchem_iupac(smiles):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
           f"{urllib.parse.quote(smiles)}/property/IUPACName/JSON")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            p = json.load(r)["PropertyTable"]["Properties"][0]
        return p.get("IUPACName")
    except Exception:  # noqa: BLE001 — not found / throttled / timeout
        return None


def main():
    enrich = pd.read_parquet(ENRICH)
    have = _have_iupac_skeletons()

    # unique (skeleton, smiles) still missing an IUPAC name
    todo = {}
    for smi, skel in zip(enrich["smiles"], enrich.get("inchikey_skel", [])):
        if not isinstance(smi, str) or not isinstance(skel, str) or skel in have:
            continue
        todo.setdefault(skel, smi)

    print(f"{len(have)} skeletons already have IUPAC; {len(todo)} to crawl.")
    rows, got = [], 0
    for i, (skel, smi) in enumerate(todo.items(), 1):
        name = _pubchem_iupac(smi)
        if name:
            rows.append({"inchikey_skel": skel, "iupac_name": name})
            got += 1
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} crawled, {got} resolved")
            # checkpoint so a crash doesn't lose progress
            _merge_and_save(rows)
        time.sleep(PAUSE)
    _merge_and_save(rows)
    print(f"Done. Resolved {got}/{len(todo)} missing IUPAC names -> {OUT}")


def _merge_and_save(rows):
    if not rows:
        return
    new = pd.DataFrame(rows).drop_duplicates("inchikey_skel")
    try:
        old = pd.read_parquet(OUT)
        merged = pd.concat([old, new]).drop_duplicates("inchikey_skel", keep="last")
    except Exception:  # noqa: BLE001 — first write
        merged = new
    merged.to_parquet(OUT, index=False)


if __name__ == "__main__":
    main()
