"""Playwright end-to-end fixtures for the Flavormancer workbench.

These tests drive the real UI in a headless browser against a running server. They are
designed to be safe everywhere:

  * if no server is reachable at FLAVORMANCER_URL (e.g. the plain unit-test CI job), every
    e2e test SKIPS — it never fails the build;
  * `smoke`-marked tests need only a running server (they pass even with no trained models —
    they catch JS errors, broken HTML, and dead endpoints);
  * the rest need trained models and skip via the `models_present` fixture when absent.

Run the full suite on a box with models:  FLAVORMANCER_URL=http://127.0.0.1:8000 pytest tests/e2e
Run just the model-free smoke tests:       pytest tests/e2e -m smoke
"""
import os
import urllib.request

import pytest

BASE = os.environ.get("FLAVORMANCER_URL", "http://127.0.0.1:8000").rstrip("/")


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: e2e test that needs only a running server (no models)")


def _get_json(path, timeout=4):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        import json
        return json.load(r)


def _reachable():
    try:
        urllib.request.urlopen(BASE + "/api/studio_terms", timeout=3)
        return True
    except Exception:  # noqa: BLE001 — any failure means "not reachable", skip
        return False


@pytest.fixture(scope="session")
def base_url():
    if not _reachable():
        pytest.skip(f"Flavormancer server not reachable at {BASE} — start it to run e2e tests")
    return BASE


@pytest.fixture(scope="session")
def models_present(base_url):
    """True when the server has trained taste/aroma models (a real read comes back numeric)."""
    try:
        import json
        req = urllib.request.Request(base_url + "/api/predict", method="POST",
                                     data=json.dumps({"smiles": "vanillin"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.load(r)
        return isinstance(d.get("sweet"), (int, float))
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def _browser():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(_browser, base_url):
    ctx = _browser.new_context(viewport={"width": 1280, "height": 1000})
    pg = ctx.new_page()
    pg._flavor_errors = []
    pg.on("pageerror", lambda e: pg._flavor_errors.append(str(e)))
    yield pg
    ctx.close()


@pytest.fixture
def needs_models(models_present):
    if not models_present:
        pytest.skip("trained models not present on the server — skipping model-dependent e2e test")
