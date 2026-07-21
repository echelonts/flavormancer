"""End-to-end UI tests for the Flavormancer workbench.

`smoke` tests need only a running server; the rest use `needs_models` and skip without
trained models. See conftest.py for how skipping works.
"""
import pytest


def _open(page, base_url):
    page.goto(base_url + "/", wait_until="domcontentloaded")
    page.wait_for_timeout(700)


# ── smoke: structure & no-JS-error checks (pass even with no models) ──────────
@pytest.mark.smoke
def test_page_loads_without_js_errors(page, base_url):
    _open(page, base_url)
    assert "Flavormancer" in page.title() or page.locator("header").count() >= 1
    assert page._flavor_errors == [], f"page errors: {page._flavor_errors}"


@pytest.mark.smoke
def test_key_sections_present(page, base_url):
    _open(page, base_url)
    for sel in ("#q", "#studio", "#formStudio", "#cmpPanel", "#mixPanel", "#browse",
                ".map-section", "#howStudio"):
        assert page.locator(sel).count() >= 1, f"missing section: {sel}"


@pytest.mark.smoke
def test_panels_open_by_default(page, base_url):
    _open(page, base_url)
    assert page.locator("#formWrap").is_visible(), "Formulation Studio should be open by default"
    assert page.locator("#cmpPanel").is_visible(), "Compare should be open by default"
    assert page.locator("#howWrap").is_visible(), "How-it-works should be open by default"


@pytest.mark.smoke
def test_how_it_works_toggle_and_doc_links(page, base_url):
    _open(page, base_url)
    links = page.locator("#howWrap .how-foot a")
    assert links.count() == 3
    for i in range(3):
        assert (links.nth(i).get_attribute("href") or "").startswith("https://github.com/")
    page.locator("#howToggle").click()
    page.wait_for_timeout(250)
    assert page.locator("#howWrap").is_hidden()


# ── model-dependent flows ────────────────────────────────────────────────────
def test_search_reads_a_molecule(page, base_url, needs_models):
    _open(page, base_url)
    page.fill("#q", "vanillin")
    page.click("#go")
    page.wait_for_selector(".meter, .cname, #behaviorCard", timeout=30000)
    body = page.locator("body").inner_text().lower()
    assert "vanillin" in body
    assert page._flavor_errors == [], f"page errors: {page._flavor_errors}"


def test_formulation_studio_starter(page, base_url, needs_models):
    _open(page, base_url)
    page.locator(".form-ex", has_text="citrus soda").click()
    page.wait_for_selector("#formResults .fprof-row", timeout=30000)
    page.wait_for_timeout(600)
    assert page.locator("#formResults .fprof-row").count() >= 3
    assert page.locator("#formResults .form-gate").count() == 1  # honest data-gate note
    assert page.locator("#formTargetChips .chip").count() >= 20  # target picker populated


def test_compare_loads_and_loaders_clear(page, base_url, needs_models):
    _open(page, base_url)
    page.locator(".cmp-ex", has_text="vanillin vs").click()
    page.wait_for_selector("#cmpResults .cmp-card:not(.loading) img", timeout=30000)
    page.wait_for_timeout(500)
    assert page.locator("#cmpResults .cmp-card img:visible").count() == 2
    assert page.locator("#cmpResults .cmp-load:visible").count() == 0, "loaders must clear after load"


def test_flavor_studio_finds_molecules(page, base_url, needs_models):
    _open(page, base_url)
    chip = page.locator("#studioChips .schip").first
    chip.wait_for(timeout=15000)
    chip.click()
    page.locator("#studioGo").click()
    page.wait_for_selector("#studioResults .dcell, #studioResults .design-card, #studioResults .browse-card",
                           timeout=30000)
    assert page._flavor_errors == [], f"page errors: {page._flavor_errors}"


def test_chirality_note_and_gate(page, base_url, needs_models):
    _open(page, base_url)
    page.fill("#q", "carvone")
    page.click("#go")
    page.wait_for_selector(".chiral-note", timeout=30000)
    assert page.locator(".chiral-note .gate-tag").count() == 1


def test_flavor_map_renders(page, base_url, needs_models):
    _open(page, base_url)
    page.locator("#mapLegend .leg").first.wait_for(timeout=30000)
    assert page.locator("#mapLegend .leg").count() >= 5  # legend classes populated


def test_molecular_formula_shown(page, base_url, needs_models):
    _open(page, base_url)
    page.fill("#q", "vanillin")
    page.click("#go")
    page.wait_for_selector("#formula", timeout=30000)
    page.wait_for_timeout(600)
    assert "Formula" in (page.locator("#formula").inner_text() or "")  # the 4th identifier row


def test_recipe_designer(page, base_url, needs_models):
    _open(page, base_url)
    # pick a note that is a new supplement head, then design a recipe for it
    chip = page.locator("#formTargetChips .chip", has_text="coconut").first
    chip.wait_for(timeout=15000)
    chip.click()
    page.locator("#formDesign").click()
    page.wait_for_selector("#formResults .recipe-strip .recipe-ing", timeout=40000)
    page.wait_for_timeout(600)
    assert page.locator("#formResults .recipe-ing").count() >= 1
    assert page.locator("#recipeLoad").count() == 1  # the "load into rows" bridge
    assert page._flavor_errors == [], f"page errors: {page._flavor_errors}"


def test_recipe_export(page, base_url, needs_models):
    _open(page, base_url)
    chip = page.locator("#formTargetChips .chip", has_text="coconut").first
    chip.wait_for(timeout=15000)
    chip.click()
    page.locator("#formDesign").click()
    page.wait_for_selector("#formResults .recipe-strip .recipe-ing", timeout=40000)
    # both export affordances present (parity with the flavor-card export)
    assert page.locator("#recipeCsv").count() == 1
    assert page.locator("#recipeCard").count() == 1
    # CSV downloads as a text/csv bench sheet
    with page.expect_download(timeout=15000) as dl_csv:
        page.locator("#recipeCsv").click()
    assert dl_csv.value.suggested_filename.endswith(".csv")
    # recipe card downloads as a PNG
    with page.expect_download(timeout=30000) as dl_png:
        page.locator("#recipeCard").click()
    assert dl_png.value.suggested_filename.endswith(".png")
    assert page._flavor_errors == [], f"page errors: {page._flavor_errors}"
