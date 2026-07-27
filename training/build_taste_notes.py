"""
build_taste_notes.py — documented TASTE from PubChem (public-domain sources ONLY).

PubChem carries a "Taste" annotation (free-text descriptions: "sweet taste", "bitter",
"saline/salty", "sour", "tasteless"…) for ~750 compounds, HSDB/NLM-sourced — public domain,
same shape as the "Odor" set we already mine. This pulls it via the annotations API (inline
text + source + CID), keeps only public-domain sources, resolves CIDs to InChIKey/SMILES/name,
and writes taste_notes.parquet (inchikey, smiles, name, taste, taste_source).

Uses: (1) show DOCUMENTED taste beside the model read; (2) parse into extra labels
(sweet/bitter/sour/salty/tasteless) to AUGMENT ChemTastesDB and retrain — potentially enough
salty examples for a first salty head. Commercial-clean (public domain).

Usage: python build_taste_notes.py            # -> taste_notes.parquet
"""
import sys

import pandas as pd

from build_odor_notes import annotation_records, cids_to_structs

COLS = ["inchikey", "smiles", "name", "taste", "taste_source"]

if __name__ == "__main__":
    by_cid = {}
    for cid, strs, src in annotation_records("Taste"):  # public-domain-filtered, all pages
        t, s = by_cid.get(cid, ([], set()))
        by_cid[cid] = (t + strs, s | {src})
    cids = sorted(by_cid)
    print(f"annotations: {len(cids)} public-domain Taste CIDs; resolving structures...", flush=True)
    cid2s = cids_to_structs(cids)
    rows = []
    for cid in cids:
        st = cid2s.get(cid)
        if not st:
            continue
        ik, smi, title = st
        strs, srcs = by_cid[cid]
        rows.append({"inchikey": ik, "smiles": smi, "name": title,
                     "taste": "\n".join(sorted(set(strs), key=len)) or None,
                     "taste_source": "; ".join(sorted(srcs)) or None})
    if not rows:
        print("no taste annotations resolved")
        sys.exit(1)
    df = pd.DataFrame(rows, columns=COLS)
    df.to_parquet("taste_notes.parquet")
    print(f"taste_notes.parquet: {len(df)} molecules with documented taste (public-domain)")
