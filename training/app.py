"""
app.py — serving layer for the flavor workbench (demo).

Run:
    pip install fastapi "uvicorn[standard]" pydantic   # plus the SETUP.md env
    uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://<r620-ip>:8000/

Endpoints:
    GET  /              -> the workbench UI (workbench.html)
    POST /api/predict   -> {smiles|name}      -> full flavor read   (predict.predict)
    POST /api/neighbors -> {smiles|name, k}   -> substitution search (predict.substitute)

Both endpoints delegate to predict.py — one source of truth for the flavor read AND
the substitution search (Tanimoto/Morgan nearest-neighbor over the labeled molecules;
runnable today, no aroma model required). Auth / per-seat is stubbed (single open
instance) for the demo; deployment puts this behind login + per-user history, and the
prediction core doesn't change.
"""

import threading
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rdkit import Chem

import predict as P  # the unified flavor read + substitution search

app = FastAPI(title="Flavor Workbench (demo)")


def _resolve(text: str):
    """Accept a SMILES or a compound name; return canonical SMILES or None."""
    text = (text or "").strip()
    if Chem.MolFromSmiles(text):
        return text
    try:
        import pubchempy as pcp
        hits = pcp.get_compounds(text, "name")
        if hits and hits[0].canonical_smiles:
            return hits[0].canonical_smiles
    except Exception:
        pass
    return None


def _svg(smi, w=320, h=220):
    """2D structure SVG; None if drawing is unavailable (headless box w/o libXrender)."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception:  # noqa: BLE001 — missing X11 libs etc.; degrade gracefully
        return None


@lru_cache(maxsize=8192)
def _names(smi):
    """(common, IUPAC) names from PubChem for a SMILES — cached, best-effort, short timeout."""
    import json
    import urllib.parse
    import urllib.request
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
           f"{urllib.parse.quote(smi)}/property/Title,IUPACName/JSON")
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            p = json.load(r)["PropertyTable"]["Properties"][0]
        return p.get("Title"), p.get("IUPACName")
    except Exception:  # noqa: BLE001 — not found / timeout / throttled
        return None, None


class Query(BaseModel):
    smiles: str
    k: int = 8


@app.post("/api/predict")
def api_predict(q: Query):
    smi = _resolve(q.smiles)
    if not smi:
        return {"error": f"Couldn't resolve '{q.smiles}' to a structure. "
                         f"Enter a valid SMILES or a recognized compound name."}
    return P.predict(smi, include_aroma=False)


@app.post("/api/neighbors")
def api_neighbors(q: Query):
    smi = _resolve(q.smiles)
    if not smi:
        return {"neighbors": []}
    res = P.substitute(smi, k=q.k)
    for n in res.get("neighbors", []):  # enrich each candidate with a structure + a name
        n["svg"] = _svg(n["smiles"], 132, 96)
        n["name"] = _names(n["smiles"])[0]
    return res


@app.post("/api/names")
def api_names(q: Query):
    """Common (PubChem Title) + IUPAC names for the queried molecule."""
    smi = _resolve(q.smiles)
    common, iupac = _names(smi) if smi else (None, None)
    return {"common": common, "iupac": iupac, "smiles": smi}


@app.post("/api/structure")
def api_structure(q: Query):
    """2D structure depiction (SVG); {svg: None} if drawing is unavailable."""
    return {"svg": _svg(_resolve(q.smiles))}


class MixtureQuery(BaseModel):
    ingredients: list[str]
    processes: list[str] = []


@app.post("/api/mixture")
def api_mixture(m: MixtureQuery):
    """Per-ingredient reads + documented-hazard screen + a single-molecule palette match."""
    smis = [s for s in (_resolve(x) for x in m.ingredients) if s]
    out = P.check_mixture(smis, m.processes)
    reads, palette = [], set()
    for s in smis:
        r = P.predict(s)
        tp = r.get("taste_profile", [])
        tastes = [t for t in ("sweet", "bitter", "umami") if isinstance(r.get(t), (int, float)) and r[t] >= 0.5]
        if r.get("sour"):
            tastes.append("sour")
        if r.get("salty") is True:
            tastes.append("salty")
        for t in (r.get("known_tastes") or []):
            if t not in tastes:
                tastes.append(t)
        palette.update(tastes)
        reads.append({"smiles": r["smiles"], "name": _names(s)[0], "svg": _svg(s, 110, 80),
                      "top_taste": tp[0]["taste"] if tp else None, "tastes": tastes,
                      "gras": r["safety"]["gras_status"], "alerts": r["safety"]["structural_alerts"],
                      "tox_flags": r["safety"]["tox_screen"].get("flags", []) if r["applicability"]["in_domain"] else []})
    pal = P.palette_match(sorted(palette), k=5)
    for mt in pal.get("matches", []):
        mt["svg"] = _svg(mt["smiles"], 110, 80)
        mt["name"] = _names(mt["smiles"])[0]
    out["ingredients"] = reads
    out["palette"] = pal
    return out


def _load_suggest():
    import csv
    try:
        with open("flavor_volatiles.csv", encoding="utf-8") as f:
            return [(r["name"], r["smiles"]) for r in csv.DictReader(f)]
    except Exception:  # noqa: BLE001 — no file / bad rows; typeahead just stays empty
        return []


_SUGGEST = _load_suggest()
_IUPAC = {}  # smiles -> IUPAC name, filled in the background (PubChem, cached)


def _precompute_iupac():
    for _, s in _SUGGEST:
        _IUPAC[s] = _names(s)[1]


if _SUGGEST:
    threading.Thread(target=_precompute_iupac, daemon=True).start()


@app.get("/api/suggest")
def api_suggest(qs: str = ""):
    """Rich typeahead over the curated flavor-volatile list — name + SMILES + structure + IUPAC."""
    t = qs.strip().lower()
    if len(t) < 2:
        return {"items": []}
    items = [{"name": n, "smiles": s} for n, s in _SUGGEST if t in n.lower()][:8]
    for it in items:
        it["svg"] = _svg(it["smiles"], 90, 64)
        it["iupac"] = _IUPAC.get(it["smiles"])
    return {"items": items}


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("workbench.html").read_text()
