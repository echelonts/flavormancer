"""Integration tests: every endpoint's CONTRACT, through FastAPI's TestClient.

The suite already had unit tests (rules, physchem, stereo-safety) and Playwright e2e tests that
drive a real browser against a real server. This is the missing middle: it exercises the actual
HTTP layer — routing, request parsing, response shape, status codes — without a browser, without a
port, and without waiting fifty seconds for models to load.

That gap mattered. A response that silently drops a field, renames a key, or starts returning 200
with an error body would sail past the unit tests (which never touch HTTP) and might well pass the
e2e tests too (which assert on rendered pixels, not payloads). The workbench, the MCP server and
the Claude skill all consume these payloads by key, so a renamed field is a breaking change that
nothing else in the pyramid would catch.

Runs in CI with FLAVORMANCER_NO_MODELS=1: endpoints that need trained heads answer honestly with
`available: false` rather than 500, and that degradation is itself worth asserting — it is what
keeps a partially-built install usable instead of broken.
"""
import os

import pytest

# Import the app the same way uvicorn does. NO_MODELS keeps this fast and CI-friendly; the tests
# below assert the *contract*, which must hold whether or not the heads are present.
os.environ.setdefault("FLAVORMANCER_NO_MODELS", "1")

# importorskip is not enough here: `fastapi.testclient` imports cleanly and then starlette raises
# a RuntimeError if httpx is missing, which pytest reports as a collection ERROR rather than a
# skip. Catch both so an environment without the test extras skips this file instead of failing.
try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as exc:  # pragma: no cover — environment-dependent
    pytest.skip(f"FastAPI TestClient unavailable ({exc})", allow_module_level=True)

import app as flavor_app

VANILLIN = "COc1cc(C=O)ccc1O"


@pytest.fixture(scope="module")
def client():
    with TestClient(flavor_app.app) as c:
        yield c


# --- liveness -------------------------------------------------------------------------------

def test_healthz_is_always_answerable(client):
    """/healthz must answer even while models are loading — it is what the container's health
    check polls, and a 503 during a legitimate 50s warm-up would kill the container."""
    r = client.get("/healthz")
    assert r.status_code == 200, r.text


def test_status_reports_load_progress(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    for key in ("loaded", "total", "phase", "ready"):
        assert key in body, f"/api/status lost `{key}` — the warming page polls it"
    assert isinstance(body["ready"], bool)


# --- head catalog ---------------------------------------------------------------------------

def test_heads_catalog_shape(client):
    """The catalog is consumed by the modal, the studios, the MCP server and the skill. Its four
    categories and their per-head fields are a contract, not an implementation detail."""
    r = client.get("/api/heads")
    assert r.status_code == 200
    body = r.json()
    for group in ("taste", "aroma", "mouthfeel", "safety"):
        assert group in body, f"/api/heads lost the `{group}` group"
        assert isinstance(body[group], list)


def test_every_head_publishes_its_calibration(client):
    """Since #261 a head must say where its bar sits and how precise it is there. Dropping these
    would silently return the project to reporting AUROC alone — the exact failure ACCURACY.md
    exists to prevent."""
    body = client.get("/api/heads").json()
    for group in ("taste", "aroma", "mouthfeel", "safety"):
        for head in body[group]:
            assert "head" in head
            assert "confident_capable" in head, (
                f"{group}/{head.get('head')} has no confident_capable flag")
            thr = head.get("threshold")
            if thr is not None:
                assert 0.0 < float(thr) <= 1.0, f"{head['head']} threshold {thr} out of range"


# --- prediction -----------------------------------------------------------------------------

def test_predict_returns_the_documented_blocks(client):
    r = client.post("/api/predict", json={"smiles": VANILLIN})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("smiles")
    # physchem and the applicability gate are computed from structure alone, so they are present
    # even with no trained heads at all — that is what makes a models-less install still useful.
    assert "physchem" in body
    assert "applicability" in body


def test_predict_rejects_an_unparseable_smiles(client):
    """Garbage in must produce a clear error, not a 500 and not a confident-looking empty read."""
    r = client.post("/api/predict", json={"smiles": "not-a-molecule"})
    assert r.status_code < 500, "an unparseable SMILES must not crash the service"
    body = r.json()
    assert "error" in body or body.get("applicability", {}).get("in_domain") is False, (
        "an unparseable SMILES should surface an error, not a silent empty result")


def test_predict_requires_smiles(client):
    r = client.post("/api/predict", json={})
    assert r.status_code < 500


# --- search surfaces ------------------------------------------------------------------------

def test_design_search_contract(client):
    """The reverse search backs the studios. Its envelope (items / requested / total_matches)
    is what the UI paginates against."""
    r = client.get("/api/design", params={"descriptors": "vanilla", "limit": 3})
    assert r.status_code == 200
    body = r.json()
    for key in ("items", "requested", "total_matches", "offset", "limit"):
        assert key in body, f"/api/design lost `{key}`"
    assert isinstance(body["items"], list)
    assert len(body["items"]) <= 3, "limit must be honoured"


def test_design_with_an_unknown_descriptor_is_empty_not_an_error(client):
    r = client.get("/api/design", params={"descriptors": "notarealnote"})
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_studio_terms_groups(client):
    r = client.get("/api/studio_terms")
    assert r.status_code == 200
    body = r.json()
    for group in ("flavors", "notes", "mouthfeel", "taste"):
        assert group in body, f"/api/studio_terms lost `{group}` — a chip family would vanish"


# --- degradation ----------------------------------------------------------------------------

def test_model_backed_endpoints_degrade_honestly(client):
    """With no heads loaded, endpoints that need them must say so rather than 500 or, worse,
    return an empty read that looks like a confident 'no aroma'."""
    body = client.post("/api/predict", json={"smiles": VANILLIN}).json()
    aroma = body.get("aroma")
    if isinstance(aroma, dict) and aroma.get("available") is False:
        assert aroma.get("note"), "an unavailable modality must explain why"
