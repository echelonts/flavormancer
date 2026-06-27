"""
build_aroma_dataset.py — assemble the clean aroma (odor-descriptor) training table.

Source: keller_2016 (Keller & Vosshall 2016, BMC Neuroscience, CC-BY-4.0) — the only
commercially-clean odor-descriptor dataset available (~480 molecules, 20 descriptors).
Each molecule gets a mean 0-100 panel rating per descriptor.

Output: aroma_master.parquet  (smiles + 20 descriptor columns, 0-100)
"""
from pathlib import Path

import pandas as pd
from rdkit import Chem

KELLER = Path("aroma/keller_2016")
DESCRIPTORS = ["ACID", "AMMONIA/URINOUS", "BAKERY", "BURNT", "CHEMICAL", "COLD",
               "DECAYED", "EDIBLE", "FISH", "FLOWER", "FRUIT", "GARLIC", "GRASS",
               "MUSKY", "SOUR", "SPICES", "SWEATY", "SWEET", "WARM", "WOOD"]


def canon(smiles):
    if not isinstance(smiles, str):
        return None
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m else None


def build():
    mol = pd.read_csv(KELLER / "molecules.csv")
    sti = pd.read_csv(KELLER / "stimuli.csv")
    beh = pd.read_csv(KELLER / "behavior.csv")

    beh = beh[beh["MeasurementValue"].isin(DESCRIPTORS)].copy()
    beh["Value"] = pd.to_numeric(beh["Value"], errors="coerce")
    beh = beh.dropna(subset=["Value"])

    # mean rating per (Stimulus, descriptor) across the panel
    sm = beh.groupby(["Stimulus", "MeasurementValue"])["Value"].mean().unstack()

    # Stimulus -> CID (single-molecule) -> canonical SMILES
    sti = sti[["Stimulus", "CIDs"]].copy()
    sti["CID"] = pd.to_numeric(sti["CIDs"], errors="coerce")
    sm = sm.join(sti.set_index("Stimulus")[["CID"]]).dropna(subset=["CID"])
    sm["CID"] = sm["CID"].astype(int)
    mol_map = mol.dropna(subset=["CID"]).drop_duplicates("CID").set_index("CID")["CanonicalSMILES"]
    sm["smiles"] = sm["CID"].map(mol_map).map(canon)
    sm = sm.dropna(subset=["smiles"])

    # molecule level: mean across the molecule's stimuli (concentrations)
    agg = sm.groupby("smiles")[DESCRIPTORS].mean().reset_index()
    agg.to_parquet("aroma_master.parquet")

    print(f"aroma_master.parquet: {len(agg)} molecules x {len(DESCRIPTORS)} descriptors")
    for d in DESCRIPTORS:
        print(f"  {d:16s} mean={agg[d].mean():5.1f}  max={agg[d].max():5.1f}  "
              f">=15: {int((agg[d] >= 15).sum())}")


if __name__ == "__main__":
    build()
