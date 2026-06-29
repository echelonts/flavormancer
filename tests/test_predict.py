"""End-to-end predict() — must run and stay well-formed even with NO trained models
(as in CI). The trained taste heads sharpen the output; their absence must never crash it.
"""
import predict


def test_predict_wellformed_without_models():
    out = predict.predict("OC(=O)CC(O)(CC(=O)O)C(=O)O")  # citric acid
    for key in ("smiles", "physchem", "stability", "chemesthesis",
                "analytical", "labeling", "safety", "taste_profile"):
        assert key in out
    assert out["safety"]["review_required"] is True
    assert out["sour"] is True  # the rule fires regardless of trained heads


def test_predict_rejects_bad_smiles():
    out = predict.predict("not-a-molecule")
    assert "error" in out


def test_predict_aroma_is_honest_placeholder():
    out = predict.predict_aroma("CCO")
    assert out["available"] is False
    assert "AROMA.md" in out["note"]


def test_predict_includes_tox_screen():
    # tox_screen is always present and well-formed (honest available:False with no models, as in CI)
    out = predict.predict("CCO")
    assert "tox_screen" in out["safety"]
    assert isinstance(out["safety"]["tox_screen"]["available"], bool)
