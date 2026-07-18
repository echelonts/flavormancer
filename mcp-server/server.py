"""Flavormancer MCP server.

Exposes the on-prem Flavormancer models as Model-Context-Protocol tools, so any MCP
client (Claude Desktop, an agent, another app) can ask the *trained* taste/aroma
models to read a molecule, analyze a whole formulation, screen a mixture for
documented hazards, or find molecules that carry a set of notes.

It is a thin adapter: it calls the local Flavormancer HTTP API (which runs the
RandomForest heads on-prem), so nothing leaves the box. Point it at a different
instance with the FLAVORMANCER_URL env var (default http://127.0.0.1:8000).

Run:  python server.py           # stdio transport, for Claude Desktop / MCP clients
"""
from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

BASE_URL = os.environ.get("FLAVORMANCER_URL", "http://127.0.0.1:8000")
_TIMEOUT = float(os.environ.get("FLAVORMANCER_TIMEOUT", "60"))

mcp = FastMCP("flavormancer")


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, timeout=_TIMEOUT)


def _read_flavor(molecule: str) -> dict:
    """Core logic (kept plain so it is unit-testable without an MCP client)."""
    with _client() as c:
        r = c.post("/api/predict", json={"smiles": molecule})
        if r.status_code != 200:
            return {"error": f"could not read '{molecule}' (HTTP {r.status_code})"}
        d = r.json()
        names = c.post("/api/names", json={"smiles": molecule}).json()
        aroma = c.post("/api/aroma", json={"smiles": molecule}).json()
    safety = d.get("safety", {}) or {}
    # All six taste heads as probabilities (sour/salty expose the model view; the rule gives the verdict)
    taste = {k: d.get(k) for k in ("sweet", "bitter", "umami", "tasteless")
             if isinstance(d.get(k), (int, float))}
    if isinstance(d.get("sour_predicted"), (int, float)):
        taste["sour"] = d["sour_predicted"]
    if isinstance(d.get("salty_predicted"), (int, float)):
        taste["salty"] = d["salty_predicted"]
    descriptors = (aroma.get("predicted") or {}).get("descriptors", [])
    confident = [x["odor"] for x in descriptors if x.get("confident")]
    return {
        "name": names.get("common"),
        "iupac": names.get("iupac"),
        "smiles": d.get("smiles"),
        "taste_probabilities": {k: round(v, 3) for k, v in taste.items()},
        "sour_rule": d.get("sour"),
        "salty_rule": d.get("salty"),
        "confident_aromas": confident,
        "gras_status": safety.get("gras_status"),
        "structural_alerts": safety.get("structural_alerts"),
        "in_applicability_domain": (d.get("applicability") or {}).get("in_domain"),
        "disclaimer": "Flavor prediction only — NOT a safety, GRAS, or regulatory determination.",
    }


def _analyze_formulation(ingredients: list, target: list, processes: list) -> dict:
    with _client() as c:
        r = c.post("/api/formulation",
                   json={"ingredients": ingredients, "target": target or [], "processes": processes or []})
    if r.status_code != 200:
        return {"error": f"formulation analysis failed (HTTP {r.status_code})"}
    d = r.json()
    # Return the parts an agent actually reasons over (drop heavy SVG blobs).
    return {
        "ingredients": [{k: ing.get(k) for k in ("name", "smiles", "ppm", "weight", "aromas")}
                        for ing in d.get("ingredients", [])],
        "profile": d.get("profile"),
        "overpowering": d.get("overpowering"),
        "weighting": d.get("weighting"),
        "balance_warnings": d.get("balance_warnings"),
        "gap": d.get("gap"),
        "active_hazards": d.get("active_hazards"),
        "conditional_hazards": d.get("conditional_hazards"),
        "data_gates": d.get("data_gates"),
        "scope_note": d.get("scope_note"),
    }


def _screen_mixture(ingredients: list, processes: list) -> dict:
    with _client() as c:
        r = c.post("/api/mixture", json={"ingredients": ingredients, "processes": processes or []})
    if r.status_code != 200:
        return {"error": f"mixture screen failed (HTTP {r.status_code})"}
    d = r.json()
    return {
        "ingredients": [{k: ing.get(k) for k in ("name", "smiles", "tastes", "aromas", "gras", "alerts")}
                        for ing in d.get("ingredients", [])],
        "active_hazards": d.get("active_hazards"),
        "conditional_hazards": d.get("conditional_hazards"),
        "reactions": [{k: rx.get(k) for k in ("name", "smiles", "reaction", "tastes", "aromas")}
                      for rx in d.get("reactions", [])],
        "scope_note": d.get("scope_note"),
    }


def _find_molecules(notes: list, food_safe: bool) -> dict:
    with _client() as c:
        r = c.get("/api/studio", params={"terms": ",".join(notes),
                                          "gras": 1 if food_safe else 0, "limit": 12})
    if r.status_code != 200:
        return {"error": f"molecule search failed (HTTP {r.status_code})"}
    d = r.json()
    return {
        "query_notes": notes,
        "food_safe_only": food_safe,
        "total_matches": d.get("total_matches"),
        "matches": [{"name": h.get("name"), "smiles": h.get("smiles"), "gras": h.get("gras"),
                     "matched_notes": h.get("matched"), "other_notes": h.get("other"),
                     "n_matched": h.get("n_matched")}
                    for h in d.get("items", [])],
    }


@mcp.tool()
def read_flavor(molecule: str) -> dict:
    """Predict the taste and aroma of a single molecule from its structure.

    molecule: a common name, IUPAC name, or SMILES string (e.g. "vanillin" or
    "O=Cc1ccc(O)c(OC)c1"). Returns the six taste-head probabilities, the confident
    aroma descriptors, GRAS status, structural alerts, and whether the molecule is
    inside the models' applicability domain. Prediction only — not a safety clearance.
    """
    return _read_flavor(molecule)


@mcp.tool()
def analyze_formulation(ingredients: list[dict], target: list[str] = [], processes: list[str] = []) -> dict:
    """Read a whole formulation before you pour.

    ingredients: list of {"name": <name or SMILES>, "ppm": <optional dose>} objects.
    target: optional list of aroma notes you're aiming for (drives the gap analysis).
    processes: optional list of "high_heat" | "refining" | "fermentation".

    Returns the blended note-profile (weighted by odor impact) with the driving
    ingredient per note, an overpowering-component flag, a target gap analysis with
    food-safe add/cut suggestions, a documented-hazard screen, and honest data-gate
    notes (directional today; calibrated with your odor-threshold / panel data).
    """
    return _analyze_formulation(ingredients, target, processes)


@mcp.tool()
def screen_mixture(ingredients: list[str], processes: list[str] = []) -> dict:
    """Screen a mixture of ingredients for DOCUMENTED food hazards.

    ingredients: list of names or SMILES. processes: optional "high_heat" |
    "refining" | "fermentation". Returns per-ingredient reads, active and
    conditional documented hazards (e.g. benzoate + ascorbate -> benzene), and
    indicative reaction-template products. Curated screen, NOT a reaction predictor
    or a safety clearance.
    """
    return _screen_mixture(ingredients, processes)


@mcp.tool()
def find_molecules(notes: list[str], food_safe: bool = True) -> dict:
    """Find molecules that carry a set of taste/aroma notes.

    notes: list of flavor/aroma terms (e.g. ["citrus", "fresh", "sweet"]).
    food_safe: when true, restrict to GRAS/food-reference molecules. Returns ranked
    molecules with their tastes, aromas, and match score.
    """
    return _find_molecules(notes, food_safe)


if __name__ == "__main__":
    mcp.run()
