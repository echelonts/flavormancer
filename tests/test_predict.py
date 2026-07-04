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


def test_predict_aroma_honest_about_model_presence():
    # With no trained heads present (as in CI), aroma is honestly unavailable rather than
    # fabricated; when aroma_models/ IS built, it returns a descriptor list.
    out = predict.predict_aroma("CCO")
    if predict._AROMA_MODELS:
        assert out["available"] is True
        assert isinstance(out.get("descriptors"), list)
    else:
        assert out["available"] is False
        assert "note" in out


def test_predict_includes_tox_screen():
    # tox_screen is always present and well-formed (honest available:False with no models, as in CI)
    out = predict.predict("CCO")
    assert "tox_screen" in out["safety"]
    assert isinstance(out["safety"]["tox_screen"]["available"], bool)


def test_applicability_domain_flag():
    # inorganic / carbon-free molecules are flagged out-of-domain; organics are in-domain
    assert predict.predict("O")["applicability"]["in_domain"] is False  # water
    assert predict.predict("[Na+].[Cl-]")["applicability"]["in_domain"] is False  # NaCl
    assert predict.predict("O=Cc1ccc(O)c(OC)c1")["applicability"]["in_domain"] is True  # vanillin


def test_safety_structural_alert_fires():
    # regression: a nitro aromatic must trip the structural tox-alert screen. (The tox-models
    # dict was accidentally named _TOX, clobbering the structural-alert SMARTS dict; this
    # silently emptied structural_alerts with no models and crashed with them.)
    out = predict.predict("O=[N+]([O-])c1ccccc1")  # nitrobenzene
    assert out["safety"]["structural_alerts"]  # non-empty
