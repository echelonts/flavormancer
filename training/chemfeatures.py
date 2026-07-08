"""
chemfeatures.py — shared featurization for the taste & aroma models.

The taste/aroma RandomForests learn from a molecule's 2048-bit Morgan fingerprint PLUS a small
block of physicochemical descriptors (size, lipophilicity, polarity, H-bonding, flexibility,
aromaticity). Fingerprints capture local substructure; the descriptors add the global properties
that correlate with taste and odor (a big polar sugar reads sweet; a small greasy ester reads
fruity) and that pure bits miss. Trees need no scaling, so raw values go straight in.

CRITICAL: training (train_taste.py / train_aroma.py) and inference (predict.py) MUST build the
feature vector identically — same descriptors, same order. Both import from here so they can't
drift. Similarity search and the UMAP map deliberately keep the PURE fingerprint (Tanimoto /
Jaccard need the binary bits), so they don't use this block.
"""
import numpy as np
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

# (name, fn) — fixed order; append only, never reorder (would invalidate saved models).
DESC_FNS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("NumRings", rdMolDescriptors.CalcNumRings),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("HeavyAtomCount", lambda m: m.GetNumHeavyAtoms()),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
]
N_DESC = len(DESC_FNS)


def descriptors(mol):
    """Fixed-length physicochemical descriptor vector for a molecule (NaN/errors -> 0)."""
    out = np.zeros(N_DESC, dtype=np.float32)
    for i, (_, fn) in enumerate(DESC_FNS):
        try:
            v = float(fn(mol))
            out[i] = v if v == v else 0.0  # NaN guard
        except Exception:  # noqa: BLE001 — a descriptor that won't compute -> 0, keep the row usable
            out[i] = 0.0
    return out
