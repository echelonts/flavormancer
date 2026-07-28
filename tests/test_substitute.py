"""Substitution search — graceful without a reference set, correct ranking with one.

The ranking logic is tested against a hand-built in-memory index so it needs no data
files; the .NET product (M6) mirrors this as a pgvector ANN query.
"""
import predict
from rdkit import Chem


def _canon(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def test_substitute_rejects_bad_smiles():
    assert "error" in predict.substitute("nope")


def test_substitute_graceful_without_data(monkeypatch):
    monkeypatch.setattr(predict, "_SUB_INDEX", ([], [], [], [], None, []))
    out = predict.substitute("CCO")
    assert out["neighbors"] == []
    assert "note" in out


def test_substitute_ranks_by_similarity(monkeypatch):
    mols = ["CCO", "CCCO", "c1ccccc1"]  # ethanol, propanol, benzene
    fps = [predict._MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in mols]
    canon = [_canon(s) for s in mols]
    monkeypatch.setattr(predict, "_SUB_INDEX", (fps, canon, [[], [], []], [[], [], []], None, []))
    out = predict.substitute("CCO", k=2)
    neighbors = out["neighbors"]
    # the query itself is excluded
    assert all(n["smiles"] != _canon("CCO") for n in neighbors)
    # propanol (closer to ethanol) ranks above benzene
    assert neighbors[0]["smiles"] == _canon("CCCO")
    # similarities come back sorted descending
    sims = [n["similarity"] for n in neighbors]
    assert sims == sorted(sims, reverse=True)


def test_mixture_to_molecule_graceful_without_profiles(monkeypatch):
    # no profile matrix -> a clean error, never a crash / real (slow) index build
    monkeypatch.setattr(predict, "_SUB_INDEX", ([], [], [], [], None, []))
    out = predict.mixture_to_molecule(["CCO", "CCCO"])
    assert "error" in out


def test_mixture_to_molecule_rejects_all_bad(monkeypatch):
    monkeypatch.setattr(predict, "_SUB_INDEX", ([], [], [], [], None, []))
    out = predict.mixture_to_molecule(["nope", "xyz"])
    assert "error" in out


def test_substitutes_threshold_and_index_lookup(monkeypatch):
    """substitutes() returns every match above min_match (self excluded), ranked; and an in-corpus
    query reads its profile straight off the index row (no forests) via _query_profile/_index_row."""
    import numpy as np
    mols = ["CCO", "CCCO", "c1ccccc1"]  # ethanol, propanol, benzene
    canon = [_canon(s) for s in mols]
    fps = [predict._MORGAN.GetFingerprint(Chem.MolFromSmiles(s)) for s in mols]
    profiles = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    dims = ["taste:sweet", "aroma:x", "aroma:y"]
    monkeypatch.setattr(predict, "_CLASSIFIERS", {"sweet": None})  # 1 taste col, matches the mock width
    monkeypatch.setattr(predict, "_SUB_INDEX", (fps, canon, [[], [], []], [[], [], []], profiles, dims))
    predict._PN_CACHE.clear(); predict._SKEL2ROW.clear()
    # ethanol is in the index -> query profile is its row (no model inference needed)
    assert predict._index_row(Chem.MolFromSmiles("CCO")) == 0
    out = predict.substitutes("CCO", k=10, min_match=0.5)
    subs = out["substitutes"]
    assert [n["smiles"] for n in subs] == [_canon("CCCO")]  # propanol passes 0.5; benzene (orthogonal) filtered; self excluded
    assert subs[0]["profile_match"] >= 0.5
