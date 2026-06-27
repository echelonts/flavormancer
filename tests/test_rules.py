"""Deterministic rule / lookup logic — no trained models or data files required.

These cover the rule layer the .NET product (M2) will port: sour/salty calls,
allergen labeling, and the documented dangerous-mixture screen.
"""
from rdkit import Chem

import predict


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_sour_fires_on_acid():
    out = predict._sour(_mol("CC(=O)O"))  # acetic acid
    assert out["sour"] is True
    assert out["sour_reason"]


def test_sour_silent_on_nonacid():
    out = predict._sour(_mol("c1ccccc1"))  # benzene
    assert out["sour"] is False
    assert out["sour_reason"] == []


def test_salty_inorganic_salt():
    out = predict._salty(_mol("[Na+].[Cl-]"))  # NaCl
    assert out["salty"] is True
    assert "inorganic" in out["salty_reason"]


def test_salty_defers_to_organic_anion():
    # monosodium glutamate — cation present, but the organic anion owns the taste
    out = predict._salty(_mol("C(CC(=O)[O-])C(C(=O)O)N.[Na+]"))
    assert out["salty"] is False
    assert "organic anion" in out["salty_reason"]


def test_salty_silent_on_single_fragment():
    out = predict._salty(_mol("OC1OC(CO)C(O)C(O)C1O"))  # glucose
    assert out["salty"] is False


def test_labeling_shape_and_negative():
    out = predict.labeling(_mol("O=Cc1ccc(O)c(OC)c1"))  # vanillin — not declarable
    assert out["eu_declarable_allergen"] is False
    assert out["allergen_name"] is None


def test_check_mixture_empty_is_wellformed():
    out = predict.check_mixture([])
    assert out["active_hazards"] == []
    assert out["conditional_hazards"] == []
    assert "disclaimer" in out


def test_check_mixture_flags_nitrosamine():
    # sodium nitrite + a secondary amine -> documented nitrosamine hazard
    out = predict.check_mixture(["[Na+].[O-]N=O", "CNC"])
    hazards = out["active_hazards"] + out["conditional_hazards"]
    assert any("nitrosamine" in h["possible_product"].lower() for h in hazards)
