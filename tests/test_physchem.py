"""Physicochemical / stability / chemesthesis packs — exact-from-structure, model-independent."""
from rdkit import Chem

import predict


def test_physchem_benzene_mw():
    out = predict.physchem(Chem.MolFromSmiles("c1ccccc1"))
    assert abs(out["computed"]["mol_weight"] - 78.11) < 0.5
    assert "logP" in out["computed"]
    assert "water_solubility_logS" in out["estimate"]


def test_stability_flags_vanillin_motifs():
    out = predict.stability(Chem.MolFromSmiles("O=Cc1ccc(O)c(OC)c1"))  # vanillin
    assert "aldehyde" in out["oxidation_watch"]
    assert "phenol/catechol" in out["oxidation_watch"]


def test_chemesthesis_astringent_polyphenol():
    out = predict.chemesthesis(Chem.MolFromSmiles("Oc1cccc(O)c1O"))  # pyrogallol, 3 phenols
    assert any("astringent" in c for c in out["classes"])


def test_chemesthesis_pungent_isothiocyanate():
    out = predict.chemesthesis(Chem.MolFromSmiles("C=CCN=C=S"))  # allyl isothiocyanate
    assert any("pungent" in c for c in out["classes"])


def test_chemesthesis_silent_on_plain_molecule():
    out = predict.chemesthesis(Chem.MolFromSmiles("CCO"))  # ethanol
    assert out["classes"] == []
