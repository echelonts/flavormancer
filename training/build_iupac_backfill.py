"""build_iupac_backfill.py — fill in the NAMES PubChem's property crawl missed.

Fetches both PubChem **Title** (the preferred display name — "Calcium iodate") and IUPACName in
one request. An earlier version asked only for IUPACName, which is why 770 molecules shipped
nameless: inorganic salts, glycosides and peptides routinely have a Title but no computed IUPAC
name, so a name-or-nothing crawl kept dropping them.

The master enrichment table has a SMILES for every molecule but the offline name
table (properties.parquet) only carries an IUPAC name for the ones build_properties.py
happened to resolve. This backfills the gap: for every enrichment skeleton with no
IUPAC name yet, ask PubChem (public domain) for its IUPACName by SMILES and write the
results to iupac_backfill.parquet (inchikey_skel -> iupac_name), which app.py loads as a
fallback layer under properties.parquet.

Public-domain PubChem data; we cache locally, never rehost anything licensed.

    python build_iupac_backfill.py
"""
import contextlib
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

ENRICH = "master_enrichment.parquet"
PROPS = "properties.parquet"
OUT = "iupac_backfill.parquet"
PAUSE = 0.22          # ~4-5 req/s, under PubChem's guidance
TIMEOUT = 6


def _have_iupac_skeletons():
    """Skeletons that already have an IUPAC name (offline table + any prior backfill)."""
    have = set()
    for path, key in ((PROPS, "inchikey"), (OUT, "inchikey_skel")):
        df = None
        with contextlib.suppress(Exception):  # file may not exist yet
            df = pd.read_parquet(path)
        if df is None:
            continue
        cols = [c for c in ("common_name", "iupac_name") if c in df.columns]
        if not cols:
            continue
        for row in df[[key, *cols]].itertuples(index=False):
            ik, names = row[0], row[1:]
            if isinstance(ik, str) and any(isinstance(n, str) and n.strip() for n in names):
                have.add(ik.split("-")[0])
    return have


def _pubchem_names(smiles):
    """(common_name, iupac_name) from PubChem for one SMILES, in a single request.

    Fetches **Title** as well as IUPACName. Title is PubChem's preferred display name — for
    `[Ca+2].[O-][I+2]([O-])[O-]...` that is "Calcium iodate", which is what a person actually
    wants to read; the IUPAC name for the same salt is unreadable. Asking for IUPACName alone
    (as this crawler previously did) is why 770 molecules shipped with no name at all: many
    inorganics and complex glycosides have a Title but no computed IUPAC name.
    """
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
           f"{urllib.parse.quote(smiles)}/property/Title,IUPACName/JSON")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
            p = json.load(r)["PropertyTable"]["Properties"][0]
        return p.get("Title"), p.get("IUPACName")
    except Exception:  # noqa: BLE001 — not found / throttled / timeout
        return None, None


def main():
    enrich = pd.read_parquet(ENRICH)
    have = _have_iupac_skeletons()

    # unique (skeleton, smiles) still missing an IUPAC name
    todo = {}
    for smi, skel in zip(enrich["smiles"], enrich.get("inchikey_skel", [])):
        if not isinstance(smi, str) or not isinstance(skel, str) or skel in have:
            continue
        todo.setdefault(skel, smi)

    print(f"{len(have)} skeletons already named; {len(todo)} to crawl.")
    rows, got = [], 0
    for i, (skel, smi) in enumerate(todo.items(), 1):
        common, iupac = _pubchem_names(smi)
        if common or iupac:
            rows.append({"inchikey_skel": skel, "common_name": common, "iupac_name": iupac})
            got += 1
        if i % 50 == 0:
            print(f"  {i}/{len(todo)} crawled, {got} resolved")
            # checkpoint so a crash doesn't lose progress
            _merge_and_save(rows)
        time.sleep(PAUSE)
    _merge_and_save(rows)
    print(f"Done. Resolved {got}/{len(todo)} missing names -> {OUT}")


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
