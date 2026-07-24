"""build_saf_universe.py — resolve the FDA "Substances Added to Food" (SAF) inventory to structures.

The SAF inventory is a US-government work in the PUBLIC DOMAIN (commercial-clean). This reads the
local SAF export, resolves each CAS to a canonical SMILES + InChIKey via PubChem, and writes
saf_universe.parquet — the US counterpart to gb_union_list.parquet, folded into the browse universe
so master_enrichment covers the US food-substance list too. Every row is a recognized US food
substance cited to FDA SAF.

Usage: python build_saf_universe.py [saf_raw.xls]
"""
import csv
import re
import sys
import time
from io import StringIO

import pandas as pd
import pubchempy as pcp
from rdkit import Chem

SRC = sys.argv[1] if len(sys.argv) > 1 else "saf_raw.xls"
OUT = "saf_universe.parquet"
_CAS = re.compile(r"^\d{2,7}-\d{2}-\d$")

raw = open(SRC, "rb").read().decode("cp1252", errors="replace").splitlines()
hdr = next((i for i, line in enumerate(raw) if line.startswith("CAS Reg No")), 0)
rows = list(csv.DictReader(StringIO("\n".join(raw[hdr:]))))
cas_col, name_col = list(rows[0].keys())[0], "Substance"

seen, cas_list = set(), []
for r in rows:
    c = str(r.get(cas_col, "")).strip().strip('"')
    if _CAS.match(c) and c not in seen:
        seen.add(c)
        cas_list.append((c, str(r.get(name_col, "")).strip()))
print(f"SAF unique valid CAS: {len(cas_list)} — resolving via PubChem...", flush=True)


def resolve(cas):
    for attempt in range(2):
        try:
            hits = pcp.get_compounds(cas, "name")
            if hits and hits[0].smiles:
                return hits[0].smiles
        except Exception:  # noqa: BLE001
            time.sleep(0.5 * (attempt + 1))
    return None


out, hit = [], 0
for i, (cas, name) in enumerate(cas_list):
    smi = resolve(cas)
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is not None:
        out.append({"smiles": Chem.MolToSmiles(m), "inchikey": Chem.MolToInchiKey(m),
                    "name": name, "cas": cas})
        hit += 1
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{len(cas_list)} — {hit} resolved", flush=True)
    time.sleep(0.1)

pd.DataFrame(out).drop_duplicates("inchikey").to_parquet(OUT)
print(f"{OUT}: {len(out)} US food substances resolved (FDA SAF, public domain)")
