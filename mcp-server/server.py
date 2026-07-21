"""Flavormancer MCP server.

Exposes the on-prem Flavormancer models as Model-Context-Protocol tools, so any MCP
client (Claude Desktop, an agent, another app) can call the *trained* taste/aroma
models to read a molecule, analyze a whole formulation, screen a mixture, predict
reaction products, find substitutes, explore stereoisomers, search by note or flavor,
browse the library, and query the flavor-space map.

It is a thin, well-behaved adapter: it forwards to the local Flavormancer HTTP API,
which runs the RandomForest heads **on-prem** (no cloud calls in a read). Nothing
about the molecule or formulation leaves the box. Point it at a different instance
with FLAVORMANCER_URL (default http://127.0.0.1:8000).

Run:  python server.py           # stdio transport, for Claude Desktop / MCP clients
"""
from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = os.environ.get("FLAVORMANCER_URL", "http://127.0.0.1:8000")
_TIMEOUT = float(os.environ.get("FLAVORMANCER_TIMEOUT", "60"))

mcp = FastMCP("flavormancer")

_HEAVY = ("svg", "svg3d", "png", "structure_svg")  # render blobs — never useful to an LLM


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=_TIMEOUT)


def _strip(obj):
    """Recursively drop heavy render blobs (SVG/PNG) so tool output stays token-lean."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _HEAVY}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _get(path: str, **params):
    with _client() as c:
        return c.get(path, params=params)


def _post(path: str, payload: dict):
    with _client() as c:
        return c.post(path, json=payload)


# ── single molecule ──────────────────────────────────────────────────────────
def _read_flavor(molecule: str) -> dict:
    r = _post("/api/predict", {"smiles": molecule})
    if r.status_code != 200:
        return {"error": f"could not read '{molecule}' (HTTP {r.status_code})"}
    d = r.json()
    with _client() as c:
        names = c.post("/api/names", json={"smiles": molecule}).json()
        aroma = c.post("/api/aroma", json={"smiles": molecule}).json()
    safety = d.get("safety", {}) or {}
    taste = {k: d.get(k) for k in ("sweet", "bitter", "umami", "tasteless")
             if isinstance(d.get(k), (int, float))}
    if isinstance(d.get("sour_predicted"), (int, float)):
        taste["sour"] = d["sour_predicted"]
    if isinstance(d.get("salty_predicted"), (int, float)):
        taste["salty"] = d["salty_predicted"]
    descriptors = (aroma.get("predicted") or {}).get("descriptors", [])
    return {
        "name": names.get("common"), "iupac": names.get("iupac"),
        "formula": names.get("formula"), "smiles": d.get("smiles"),
        "taste_probabilities": {k: round(v, 3) for k, v in taste.items()},
        "sour_rule": d.get("sour"), "salty_rule": d.get("salty"),
        "confident_aromas": [x["odor"] for x in descriptors if x.get("confident")],
        "all_aroma_scores": {x["odor"]: round(x["score"], 3) for x in descriptors},
        "gras_status": safety.get("gras_status"),
        "structural_alerts": safety.get("structural_alerts"),
        "in_applicability_domain": (d.get("applicability") or {}).get("in_domain"),
        "disclaimer": "Flavor prediction only — NOT a safety, GRAS, or regulatory determination.",
    }


def _read_full(molecule: str) -> dict:
    """The complete read — everything /api/predict returns (physchem, stability, chemesthesis,
    chirality, analytical, labeling, references, safety) plus names + full aroma scores."""
    r = _post("/api/predict", {"smiles": molecule})
    if r.status_code != 200:
        return {"error": f"could not read '{molecule}' (HTTP {r.status_code})"}
    d = _strip(r.json())
    with _client() as c:
        d["names"] = c.post("/api/names", json={"smiles": molecule}).json()
        d["aroma"] = _strip(c.post("/api/aroma", json={"smiles": molecule}).json())
    return d


def _find_substitutes(molecule: str, k: int) -> dict:
    r = _post("/api/neighbors", {"smiles": molecule, "k": k})
    if r.status_code != 200:
        return {"error": f"substitute search failed (HTTP {r.status_code})"}
    return _strip(r.json())


def _list_stereoisomers(molecule: str) -> dict:
    r = _post("/api/stereoisomers", {"smiles": molecule})
    if r.status_code != 200:
        return {"error": f"stereoisomer search failed (HTTP {r.status_code})"}
    d = _strip(r.json())
    d["note"] = ("The trained models are achiral (2D fingerprints), so any odor/taste DIFFERENCE "
                 "between stereoisomers here is DOCUMENTED (keyed by full InChIKey), not predicted. "
                 "A chirality-aware model would need stereo-resolved sensory data.")
    return d


# ── formulation & mixtures ───────────────────────────────────────────────────
def _analyze_formulation(ingredients: list, target: list, processes: list) -> dict:
    r = _post("/api/formulation",
              {"ingredients": ingredients, "target": target or [], "processes": processes or []})
    if r.status_code != 200:
        return {"error": f"formulation analysis failed (HTTP {r.status_code})"}
    return _strip(r.json())


def _design_recipe(flavors: list, notes: list, food_safe: bool) -> dict:
    r = _post("/api/design_recipe",
              {"flavors": flavors or [], "notes": notes or [], "food_safe": food_safe})
    if r.status_code != 200:
        return {"error": f"recipe design failed (HTTP {r.status_code})"}
    return _strip(r.json())


def _screen_mixture(ingredients: list, processes: list) -> dict:
    r = _post("/api/mixture", {"ingredients": ingredients, "processes": processes or []})
    if r.status_code != 200:
        return {"error": f"mixture screen failed (HTTP {r.status_code})"}
    return _strip(r.json())


def _predict_reactions(ingredients: list, processes: list) -> dict:
    r = _post("/api/mixture", {"ingredients": ingredients, "processes": processes or []})
    if r.status_code != 200:
        return {"error": f"reaction prediction failed (HTTP {r.status_code})"}
    d = _strip(r.json())
    return {
        "reactions": d.get("reactions", []),
        "active_hazards": d.get("active_hazards", []),
        "conditional_hazards": d.get("conditional_hazards", []),
        "note": ("Template-based, indicative — the functional groups to form these ARE present; "
                 "this is not a claim the reaction proceeds, nor a yield/stability assay."),
    }


# ── search & discovery ───────────────────────────────────────────────────────
def _find_by_notes(notes: list, food_safe: bool) -> dict:
    r = _get("/api/studio", terms=",".join(notes), gras=1 if food_safe else 0, limit=12)
    if r.status_code != 200:
        return {"error": f"note search failed (HTTP {r.status_code})"}
    d = r.json()
    return {
        "query_notes": notes, "food_safe_only": food_safe, "total_matches": d.get("total_matches"),
        "matches": [{"name": h.get("name"), "smiles": h.get("smiles"), "gras": h.get("gras"),
                     "matched_notes": h.get("matched"), "other_notes": h.get("other"),
                     "n_matched": h.get("n_matched")} for h in d.get("items", [])],
    }


def _find_by_flavor(flavor: str) -> dict:
    r = _get("/api/flavor", name=flavor)
    if r.status_code != 200:
        return {"error": f"flavor lookup failed (HTTP {r.status_code})"}
    d = r.json()
    return {
        "flavor": d.get("flavor"), "category": d.get("category"),
        "molecules": [{"name": m.get("name"), "smiles": m.get("smiles"),
                       "gras": m.get("gras"), "tags": m.get("tags")}
                      for m in _strip(d.get("molecules", []))],
    }


def _interpret_request(text: str) -> dict:
    r = _get("/api/nl", q=text)
    if r.status_code != 200:
        return {"error": f"interpretation failed (HTTP {r.status_code})"}
    return r.json()


def _list_categories() -> dict:
    r = _get("/api/categories")
    if r.status_code != 200:
        return {"error": f"category list failed (HTTP {r.status_code})"}
    return r.json()


def _browse_category(category: str, limit: int) -> dict:
    r = _get("/api/top", category=category, limit=limit)
    if r.status_code != 200:
        return {"error": f"browse failed (HTTP {r.status_code})"}
    return _strip(r.json())


def _flavor_map(label: str, limit: int, full: bool) -> dict:
    r = _get("/api/map")
    if r.status_code != 200:
        return {"error": f"map fetch failed (HTTP {r.status_code})"}
    pts = r.json().get("points", [])
    axes_note = ("x/y are a 2D UMAP embedding by structure; x3/y3/z3 the 3D one. "
                 "label = taste class, aroma = odor class; mw/logp/tpsa are descriptors.")
    # full dump: EVERY point with all its fields (data-rich; can be large — ~8k rows)
    if full:
        return {"total_molecules": len(pts), "axes_note": axes_note, "points": pts}
    taste_counts, aroma_counts = {}, {}
    for p in pts:
        taste_counts[p.get("label")] = taste_counts.get(p.get("label"), 0) + 1
        aroma_counts[p.get("aroma")] = aroma_counts.get(p.get("aroma"), 0) + 1
    out = {
        "total_molecules": len(pts),
        "taste_label_counts": dict(sorted(taste_counts.items(), key=lambda kv: -kv[1])),
        "aroma_label_counts": dict(sorted(aroma_counts.items(), key=lambda kv: -kv[1])),
        "axes_note": axes_note,
        "hint": "Pass full=true for every point, or a label to get the molecules carrying it.",
    }
    if label:
        lab = label.lower()
        match = [p for p in pts if lab in (str(p.get("label")).lower(), str(p.get("aroma")).lower())]
        out["filtered_label"] = label
        out["n_matching"] = len(match)
        out["molecules"] = match[:limit]  # full point objects (coords + descriptors), not a summary
    return out


# ── MCP tool surface ─────────────────────────────────────────────────────────
@mcp.tool()
def read_flavor(molecule: str) -> dict:
    """Predict the taste + aroma of a single molecule (name or SMILES).

    Returns the six taste-head probabilities, confident aromas plus all 24 aroma scores,
    GRAS status, structural alerts, and applicability-domain flag. Prediction only.
    """
    return _read_flavor(molecule)


@mcp.tool()
def read_full(molecule: str) -> dict:
    """The COMPLETE read of a molecule: everything read_flavor gives PLUS physicochemical
    properties (logP/MW/TPSA/solubility/volatility), stability watch-flags, chemesthesis
    (cooling/pungent/astringent), chirality, analytical (retention index), regulatory
    labeling (EU allergens), spectra/reference links, and the full safety block.
    """
    return _read_full(molecule)


@mcp.tool()
def find_substitutes(molecule: str, k: int = 8) -> dict:
    """Find the k structurally-nearest molecules (drop-in swaps / reformulation candidates)
    to the given molecule, each with its similarity and known tastes.
    """
    return _find_substitutes(molecule, k)


@mcp.tool()
def list_stereoisomers(molecule: str) -> dict:
    """List every stereoisomer (R/S centers and E/Z bonds) of a molecule, with any
    ISOMER-SPECIFIC documented odor/taste (e.g. R-carvone spearmint vs S-carvone caraway).
    The models are achiral, so differences here are documented, not predicted.
    """
    return _list_stereoisomers(molecule)


@mcp.tool()
def analyze_formulation(ingredients: list[dict], target: list[str] = [], processes: list[str] = []) -> dict:
    """Read a whole formulation before you pour.

    ingredients: list of {"name": <name or SMILES>, "ppm": <optional dose>}.
    target: optional aroma notes you're aiming for (drives the gap analysis).
    processes: optional "high_heat" | "refining" | "fermentation".

    Returns the blended note-profile (weighted by odor impact) with the driving ingredient
    per note, an overpowering-component flag, a target gap analysis with food-safe add/cut
    suggestions, a documented-hazard screen, and honest data-gate notes.
    """
    return _analyze_formulation(ingredients, target, processes)


@mcp.tool()
def design_recipe(flavors: list[str] = [], notes: list[str] = [], food_safe: bool = True) -> dict:
    """Design a STARTING formulation for a target profile — the inverse of analyze_formulation.

    Give target flavors (e.g. ["lemon"]) and/or aroma notes (e.g. ["citrus","fresh"]). Returns a
    food-safe recipe: character-impact molecules for the flavors + GRAS carriers for the notes,
    deduped, dosed by INVERSE VOLATILITY (volatile top-notes start low ~33 ppm, heavy base-notes
    higher ~100 ppm) to aim for balanced contributions in the directional model — already run
    through the analyzer so you see the predicted profile + gap. Doses are a bench starting point;
    calibrated dosing needs odor-threshold / panel data (a data-gate).
    """
    return _design_recipe(flavors, notes, food_safe)


@mcp.tool()
def design_recipe_csv(flavors: list[str] = [], notes: list[str] = [], food_safe: bool = True) -> str:
    """Same as design_recipe, but returns the recipe as a CSV bench sheet (a string) instead of
    JSON — parity with the workbench's 'Export CSV' and the skill CLI's `design --csv`. Columns:
    ingredient, smiles, ppm, volatility, carries, dose_basis. Hand this straight to a formulator
    or drop it into a spreadsheet."""
    import csv as _csv
    import io as _io
    out = _design_recipe(flavors, notes, food_safe)
    if isinstance(out, dict) and out.get("error"):
        return "error," + str(out["error"])
    rec = (out or {}).get("recipe") or []
    buf = _io.StringIO()
    buf.write("# Flavormancer formulation - directional starting recipe (tune on the bench)\n")
    w = _csv.writer(buf)
    w.writerow(["ingredient", "smiles", "ppm", "volatility", "carries", "dose_basis"])
    for i in rec:
        w.writerow([i.get("name", ""), i.get("smiles", ""), i.get("ppm", ""),
                    i.get("volatility", ""), "; ".join(i.get("carries") or []), i.get("dose_basis", "")])
    return buf.getvalue()


@mcp.tool()
def screen_mixture(ingredients: list[str], processes: list[str] = []) -> dict:
    """Screen a mixture of ingredients for DOCUMENTED food hazards (e.g. benzoate + ascorbate
    -> benzene), gated on process. Includes per-ingredient reads, a palette match, and
    indicative reaction products. Curated screen, NOT a reaction predictor or safety clearance.
    """
    return _screen_mixture(ingredients, processes)


@mcp.tool()
def predict_reactions(ingredients: list[str], processes: list[str] = []) -> dict:
    """Predict indicative reaction-template products for a set of ingredients (with each
    product's own predicted taste + aroma), plus any documented combination hazards.
    Template-based and indicative — not a claim the reaction proceeds.
    """
    return _predict_reactions(ingredients, processes)


@mcp.tool()
def find_molecules_by_notes(notes: list[str], food_safe: bool = True) -> dict:
    """Find molecules that carry a set of taste/aroma NOTES (e.g. ["citrus","fresh","sweet"]),
    ranked by how many notes they hit. food_safe restricts to GRAS/food-reference molecules.
    """
    return _find_by_notes(notes, food_safe)


@mcp.tool()
def find_molecules_by_flavor(flavor: str) -> dict:
    """Find the molecule(s) behind a named FLAVOR you know (e.g. "banana", "saffron", "vanilla")
    — the compound(s) that MAKE that flavor, with their tags and GRAS status.
    """
    return _find_by_flavor(flavor)


@mcp.tool()
def interpret_request(text: str) -> dict:
    """Parse a free-text brief (e.g. "a food-safe cherry flavoring with fruity, sweet notes")
    into structured Studio picks (flavor + note terms + food-safe flag) using the offline parser.
    """
    return _interpret_request(text)


@mcp.tool()
def list_flavor_categories() -> dict:
    """List the browsable taste/aroma categories (each with a key and a molecule count) —
    use a key with browse_category to list that category's top molecules.
    """
    return _list_categories()


@mcp.tool()
def browse_category(category: str, limit: int = 24) -> dict:
    """List the model-ranked top molecules for a category key (from list_flavor_categories,
    e.g. "aroma:citrus" or "taste:sweet").
    """
    return _browse_category(category, limit)


@mcp.tool()
def flavor_map(label: str = "", limit: int = 200, full: bool = False) -> dict:
    """Query the flavor-space map (every labeled molecule embedded by structure).

    full=True dumps EVERY point with all its fields (name, smiles, taste label, aroma label,
    mw, logp, tpsa, 2D x/y and 3D x3/y3/z3 coords) — the complete ~8k-molecule dataset, data-rich
    but large. With a label (a taste like "sweet" or an aroma like "citrus"), returns the full
    point objects for the molecules carrying it (up to limit). With neither, returns the taste/
    aroma label distribution across the whole map.
    """
    return _flavor_map(label, limit, full)


if __name__ == "__main__":
    mcp.run()
