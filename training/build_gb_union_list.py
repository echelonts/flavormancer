"""build_gb_union_list.py — resolve the EU/GB flavourings Union List to structures.

Reads the full authorisation register (data.food.gov.uk, Open Government Licence v3 — commercial
reuse permitted), keeps the currently-AUTHORISED unique flavourings, resolves each CAS to a
canonical SMILES + InChIKey via PubChem, and writes gb_union_list.csv. This is the bulk food-safe
reference: every entry is an authorised EU/GB food flavouring, cited by FL number.

Usage: python build_gb_union_list.py <gb_auth.csv>
"""
import csv
import sys
import time

import pubchempy as pcp
from rdkit import Chem

SRC = sys.argv[1] if len(sys.argv) > 1 else "gb_auth.csv"
OUT = "gb_union_list.csv"
DFG = "https://data.food.gov.uk/regulated-products/flavouring_authorisations/"

# unique authorised flavourings by FL number
auth = {}
for r in csv.DictReader(open(SRC)):
    if "authorised" in r["Status"].lower():
        fl = r["FLno"].strip()
        if fl and fl not in auth:
            auth[fl] = {"name": r["FlavouringName"].strip(), "cas": r["CASno"].strip()}

print(f"authorised unique FL: {len(auth)} — resolving CAS -> structure via PubChem...")


def resolve(cas):
    for attempt in range(2):
        try:
            hits = pcp.get_compounds(cas, "name")
            if hits and hits[0].isomeric_smiles:
                return hits[0].isomeric_smiles
        except Exception:  # noqa: BLE001 — network/not-found; skip
            time.sleep(0.5 * (attempt + 1))
    return None


rows, hit, miss = [], 0, 0
for i, (fl, v) in enumerate(sorted(auth.items())):
    cas = v["cas"]
    smi = resolve(cas) if cas else None
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is not None:
        rows.append([fl, v["name"], cas, Chem.MolToSmiles(m), Chem.MolToInchiKey(m),
                     "authorised", DFG + fl])
        hit += 1
    else:
        rows.append([fl, v["name"], cas, "", "", "authorised", DFG + fl])
        miss += 1
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(auth)} — {hit} resolved, {miss} unresolved", flush=True)
    time.sleep(0.12)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fl", "name", "cas", "smiles", "inchikey", "status", "source_url"])
    w.writerows(rows)
print(f"{OUT}: {len(rows)} rows, {hit} with a resolved structure, {miss} unresolved")
