"""Substitution search — graceful without a reference set, correct ranking with one.

The ranking logic is tested against a hand-built in-memory index so it needs no data
files; the .NET product (M6) mirrors this as a pgvector ANN query.
"""
from rdkit import Chem

import predict


def _canon(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def test_substitute_rejects_bad_smiles():
    assert "error" in predict.substitute("nope")


def test_substitute_graceful_without_data(monkeypatch):
    monkeypatch.setattr(predict, "_SUB_INDEX", ([], [], []))
    out = predict.substitute("CCO")
    assert out["neighbors"] == []
    assert "note" in out


def test_substitute_ranks_by_similarity(monkeypatch):
    mols = ["CCO", "CCCO", "c1ccccc1"]  # ethanol, propanol, benzene
    fps = [predict._MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in mols]
    canon = [_canon(s) for s in mols]
    monkeypatch.setattr(predict, "_SUB_INDEX", (fps, canon, [[], [], []]))
    out = predict.substitute("CCO", k=2)
    neighbors = out["neighbors"]
    # the query itself is excluded
    assert all(n["smiles"] != _canon("CCO") for n in neighbors)
    # propanol (closer to ethanol) ranks above benzene
    assert neighbors[0]["smiles"] == _canon("CCCO")
    # similarities come back sorted descending
    sims = [n["similarity"] for n in neighbors]
    assert sims == sorted(sims, reverse=True)
