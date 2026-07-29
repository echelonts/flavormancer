"""Safety calls must be IDENTICAL across stereoisomers.

Food-use status, the structural-alert screen and the Tox21 heads all key on the molecule's
CONNECTIVITY, not its stereochemistry — food-use listings key on the InChIKey skeleton (first
block), and the Morgan fingerprint the tox heads read is generated without chirality. So a food-use
listing or a tox flag on carvone necessarily covers BOTH (R)- and (S)-carvone.

That is the correct and intended behaviour — a regulator lists the substance, not one enantiomer —
but it rests on implementation details that nothing else pins down. Chirality-aware reads (#218)
would touch exactly this machinery, so these tests exist to make any future change that silently
splits safety by stereochemistry fail loudly instead of quietly under-reporting a hazard.

Note the deliberate asymmetry: *sensory* reads MAY legitimately differ between enantiomers
((R)-carvone is spearmint, (S)-carvone is caraway) — that difference is documented, not predicted.
Safety must not.
"""
import predict
from rdkit import Chem

# (name, R-enantiomer SMILES, S-enantiomer SMILES) — classic flavor enantiomer pairs
PAIRS = [
    ("carvone", "CC(=C)[C@@H]1CC=C(C)C(=O)C1", "CC(=C)[C@H]1CC=C(C)C(=O)C1"),
    ("limonene", "CC(=C)[C@@H]1CCC(C)=CC1", "CC(=C)[C@H]1CCC(C)=CC1"),
    ("menthol", "C[C@@H]1CC[C@H](C(C)C)[C@@H](O)C1", "C[C@H]1CC[C@@H](C(C)C)[C@H](O)C1"),
]


def _both(pair):
    _, a, b = pair
    return Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)


def test_enantiomers_share_an_inchikey_skeleton():
    """The whole guarantee rests on this: stereoisomers differ only past the first InChIKey block."""
    for pair in PAIRS:
        ma, mb = _both(pair)
        assert ma is not None and mb is not None, f"{pair[0]}: unparseable test SMILES"
        # genuinely different molecules...
        assert Chem.MolToSmiles(ma) != Chem.MolToSmiles(mb), f"{pair[0]}: SMILES are not distinct"
        # ...that nonetheless share a connectivity skeleton
        ska = Chem.MolToInchiKey(ma).split("-")[0]
        skb = Chem.MolToInchiKey(mb).split("-")[0]
        assert ska == skb, f"{pair[0]}: skeletons differ ({ska} vs {skb})"


def test_food_use_status_is_identical_across_stereoisomers():
    for pair in PAIRS:
        ma, mb = _both(pair)
        assert predict._gras_status(ma) == predict._gras_status(mb), (
            f"{pair[0]}: food-use status differs between enantiomers — a listing must cover both")


def test_structural_alerts_are_identical_across_stereoisomers():
    for pair in PAIRS:
        ma, mb = _both(pair)
        assert predict._tox_alerts(ma) == predict._tox_alerts(mb), (
            f"{pair[0]}: structural alerts differ between enantiomers")


def test_tox_screen_is_identical_across_stereoisomers():
    """Skipped when the Tox21 heads aren't present (CI has no model artifacts)."""
    if not predict._TOX_MODELS:
        return
    for pair in PAIRS:
        ma, mb = _both(pair)
        pa = {a["assay"]: a["probability"] for a in predict.predict_tox(ma)["assays"]}
        pb = {a["assay"]: a["probability"] for a in predict.predict_tox(mb)["assays"]}
        assert pa == pb, f"{pair[0]}: tox-assay probabilities differ between enantiomers"


def test_safety_block_is_identical_across_stereoisomers():
    """The assembled safety payload, not just its parts — catches a future field that splits."""
    for pair in PAIRS:
        ma, mb = _both(pair)
        assert predict._safety(ma) == predict._safety(mb), (
            f"{pair[0]}: assembled safety block differs between enantiomers")
