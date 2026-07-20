#!/usr/bin/env python3
"""Flavormancer CLI — a zero-dependency (stdlib only) client for the Flavormancer API.

Every subcommand prints JSON to stdout so it's easy to read and parse. Point at a
different instance with FLAVORMANCER_URL (default http://127.0.0.1:8000).

Examples:
  python flavormancer.py read vanillin
  python flavormancer.py read-full "ethyl maltol"
  python flavormancer.py formulate "limonene:250" "citral:40" "linalool:25" --target citrus,fresh
  python flavormancer.py mixture "sodium benzoate" "ascorbic acid"
  python flavormancer.py reactions "acetic acid" "ethanol"
  python flavormancer.py notes citrus fresh sweet
  python flavormancer.py flavor banana
  python flavormancer.py substitutes vanillin -k 6
  python flavormancer.py stereoisomers carvone
  python flavormancer.py map --label citrus
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("FLAVORMANCER_URL", "http://127.0.0.1:8000").rstrip("/")
_TIMEOUT = float(os.environ.get("FLAVORMANCER_TIMEOUT", "60"))
_HEAVY = ("svg", "svg3d", "png", "structure_svg")


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in _HEAVY}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _req(path, method="GET", body=None, params=None):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return _strip(json.load(r))
    except Exception as e:  # noqa: BLE001 — surface a clean error to the caller
        return {"error": f"{type(e).__name__}: {e}", "url": url}


def _ingredients(items):
    """Parse "name:ppm" tokens into [{"name":..., "ppm":...}] (ppm optional)."""
    out = []
    for it in items:
        if ":" in it:
            name, ppm = it.rsplit(":", 1)
            try:
                out.append({"name": name.strip(), "ppm": float(ppm)})
                continue
            except ValueError:
                pass
        out.append({"name": it.strip()})
    return out


def cmd_read(a):
    d = _req("/api/predict", "POST", {"smiles": a.molecule})
    names = _req("/api/names", "POST", {"smiles": a.molecule})
    aroma = _req("/api/aroma", "POST", {"smiles": a.molecule})
    taste = {k: round(d[k], 3) for k in ("sweet", "bitter", "umami", "tasteless")
             if isinstance(d.get(k), (int, float))}
    for k, lbl in (("sour_predicted", "sour"), ("salty_predicted", "salty")):
        if isinstance(d.get(k), (int, float)):
            taste[lbl] = round(d[k], 3)
    desc = (aroma.get("predicted") or {}).get("descriptors", [])
    return {"name": names.get("common"), "iupac": names.get("iupac"),
            "formula": names.get("formula"), "smiles": d.get("smiles"),
            "taste_probabilities": taste, "sour_rule": d.get("sour"), "salty_rule": d.get("salty"),
            "confident_aromas": [x["odor"] for x in desc if x.get("confident")],
            "gras_status": (d.get("safety") or {}).get("gras_status"),
            "in_applicability_domain": (d.get("applicability") or {}).get("in_domain")}


def cmd_read_full(a):
    d = _req("/api/predict", "POST", {"smiles": a.molecule})
    d["names"] = _req("/api/names", "POST", {"smiles": a.molecule})
    d["aroma"] = _req("/api/aroma", "POST", {"smiles": a.molecule})
    return d


def cmd_formulate(a):
    return _req("/api/formulation", "POST", {
        "ingredients": _ingredients(a.ingredients),
        "target": a.target.split(",") if a.target else [],
        "processes": a.process or [],
    })


def cmd_mixture(a):
    return _req("/api/mixture", "POST", {"ingredients": a.ingredients, "processes": a.process or []})


def cmd_reactions(a):
    d = _req("/api/mixture", "POST", {"ingredients": a.ingredients, "processes": a.process or []})
    return {k: d.get(k) for k in ("reactions", "active_hazards", "conditional_hazards")}


def cmd_notes(a):
    return _req("/api/studio", params={"terms": ",".join(a.notes),
                                       "gras": 0 if a.any_source else 1, "limit": a.limit})


def cmd_flavor(a):
    return _req("/api/flavor", params={"name": a.flavor})


def cmd_substitutes(a):
    return _req("/api/neighbors", "POST", {"smiles": a.molecule, "k": a.k})


def cmd_stereoisomers(a):
    return _req("/api/stereoisomers", "POST", {"smiles": a.molecule})


def cmd_design(a):
    return _req("/api/design_recipe", "POST",
                {"flavors": a.flavors or [], "notes": a.notes or [], "food_safe": not a.any_source})


def cmd_interpret(a):
    return _req("/api/nl", params={"q": a.text})


def cmd_categories(_a):
    return _req("/api/categories")


def cmd_browse(a):
    return _req("/api/top", params={"category": a.category, "limit": a.limit})


def cmd_map(a):
    d = _req("/api/map")
    pts = d.get("points", [])
    if a.full:
        return {"total": len(pts), "points": pts}
    if a.label:
        lab = a.label.lower()
        m = [p for p in pts if lab in (str(p.get("label")).lower(), str(p.get("aroma")).lower())]
        return {"label": a.label, "n_matching": len(m), "molecules": m[:a.limit]}
    tc, ac = {}, {}
    for p in pts:
        tc[p.get("label")] = tc.get(p.get("label"), 0) + 1
        ac[p.get("aroma")] = ac.get(p.get("aroma"), 0) + 1
    return {"total": len(pts),
            "taste_counts": dict(sorted(tc.items(), key=lambda kv: -kv[1])),
            "aroma_counts": dict(sorted(ac.items(), key=lambda kv: -kv[1]))}


def main():
    p = argparse.ArgumentParser(description="Flavormancer CLI (JSON output).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def mol(name):
        s = sub.add_parser(name)
        s.add_argument("molecule")
        return s

    mol("read").set_defaults(fn=cmd_read)
    mol("read-full").set_defaults(fn=cmd_read_full)
    mol("stereoisomers").set_defaults(fn=cmd_stereoisomers)
    s = mol("substitutes")
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(fn=cmd_substitutes)

    s = sub.add_parser("formulate")
    s.add_argument("ingredients", nargs="+")
    s.add_argument("--target", default="", help="comma-separated aroma notes to aim for")
    s.add_argument("--process", action="append", help="high_heat | refining | fermentation")
    s.set_defaults(fn=cmd_formulate)

    for nm, fn in (("mixture", cmd_mixture), ("reactions", cmd_reactions)):
        s = sub.add_parser(nm)
        s.add_argument("ingredients", nargs="+")
        s.add_argument("--process", action="append")
        s.set_defaults(fn=fn)

    s = sub.add_parser("notes")
    s.add_argument("notes", nargs="+")
    s.add_argument("--any-source", action="store_true", help="don't restrict to GRAS/food-safe")
    s.add_argument("--limit", type=int, default=12)
    s.set_defaults(fn=cmd_notes)

    s = sub.add_parser("flavor")
    s.add_argument("flavor")
    s.set_defaults(fn=cmd_flavor)

    s = sub.add_parser("design")
    s.add_argument("--flavor", action="append", dest="flavors", default=[], help="target flavor (repeatable)")
    s.add_argument("--note", action="append", dest="notes", default=[], help="target aroma note (repeatable)")
    s.add_argument("--any-source", action="store_true", help="allow non-GRAS carriers")
    s.set_defaults(fn=cmd_design)

    s = sub.add_parser("interpret")
    s.add_argument("text", help="a free-text brief, e.g. 'food-safe cherry with fruity notes'")
    s.set_defaults(fn=cmd_interpret)

    sub.add_parser("categories").set_defaults(fn=cmd_categories)

    s = sub.add_parser("browse")
    s.add_argument("category", help="a category key from `categories`, e.g. aroma:citrus")
    s.add_argument("--limit", type=int, default=24)
    s.set_defaults(fn=cmd_browse)

    s = sub.add_parser("map")
    s.add_argument("--label", default="")
    s.add_argument("--full", action="store_true")
    s.add_argument("--limit", type=int, default=60)
    s.set_defaults(fn=cmd_map)

    a = p.parse_args()
    out = a.fn(a)
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    if isinstance(out, dict) and out.get("error"):
        sys.exit(1)


if __name__ == "__main__":
    main()
