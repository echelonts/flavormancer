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


def _load_name_table():
    """inchikey-skeleton -> (common, IUPAC) from the precomputed enrichment table, so the
    whole labeled set resolves instantly and offline. Empty until build_properties.py has
    written name columns; live PubChem stays the fallback for anything not in the table."""
    try:
        import pandas as pd
        df = pd.read_parquet("properties.parquet")
        if "common_name" not in df.columns:
            return {}
        out = {}
        for ik, c, u in zip(df["inchikey"], df["common_name"], df["iupac_name"]):
            if isinstance(ik, str) and (isinstance(c, str) or isinstance(u, str)):
                out[ik.split("-")[0]] = (c if isinstance(c, str) else None,
                                         u if isinstance(u, str) else None)
        return out
    except Exception:  # noqa: BLE001 — no table / no pandas; just fall back to live lookups
        return {}


_NAME_TABLE = _load_name_table()


@lru_cache(maxsize=8192)
def _names(smi):
    """(common, IUPAC) names — from the precomputed table first (instant), else live PubChem."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is not None:
        hit = _NAME_TABLE.get(Chem.MolToInchiKey(mol).split("-")[0])
        if hit:
            return hit
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
    raw = (q.smiles or "").strip()
    smi = _resolve(raw)
    common, iupac = _names(smi) if smi else (None, None)
    # If the user searched by NAME (not a SMILES), that IS the best common name — PubChem's
    # Title for a flattened structure is often the systematic name (e.g. "cinnamaldehyde"
    # resolves to Title "3-Phenylprop-2-Enal"), which then looks like the IUPAC name repeated.
    if smi and raw and Chem.MolFromSmiles(raw) is None:
        common = raw[:1].upper() + raw[1:]
    return {"common": common, "iupac": iupac, "smiles": smi}


@app.post("/api/structure")
def api_structure(q: Query):
    """2D structure depiction (SVG); {svg: None} if drawing is unavailable."""
    return {"svg": _svg(_resolve(q.smiles))}


@app.post("/api/structure3d")
def api_structure3d(q: Query):
    """3D conformer as an SDF mol block — RDKit ETKDG embed + MMFF optimize. Rendered
    interactively in the browser (3Dmol.js). {molblock: None} if a 3D embed isn't possible."""
    smi = _resolve(q.smiles)
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return {"molblock": None}
    try:
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42  # deterministic conformer
        if AllChem.EmbedMolecule(mol, params) != 0 and AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            return {"molblock": None}  # embedding failed (e.g. tricky cage/macrocycle)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:  # noqa: BLE001 — no MMFF params for some atoms; unoptimized still fine
            pass
        return {"molblock": Chem.MolToMolBlock(mol)}
    except Exception:  # noqa: BLE001 — RDKit build without embedding etc.; degrade gracefully
        return {"molblock": None}


@app.get("/static/3Dmol-min.js")
def _threedmol_js():
    """Serve the vendored 3Dmol.js (BSD-3-Clause) locally so the demo stays self-contained."""
    from fastapi.responses import FileResponse
    return FileResponse("static/3Dmol-min.js", media_type="application/javascript")


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


# --- Aroma: REAL documented odor only (public-domain HSDB/CAMEO) ---------------
# Hand-set illustrative descriptor "scores" were removed on purpose: made-up numbers
# have no place in the read. The aroma card now shows only real, cited documented odor
# (odor_notes.parquet, built by build_odor_notes.py). A trained per-molecule descriptor
# model — presence/absence learned from these same descriptions, or intensity from a
# customer's expert-labeled data — is the next step (see docs/AROMA.md).
def _load_odor_table():
    """inchikey-skeleton -> {odor, odor_source, threshold_ppm, threshold_source} from
    odor_notes.parquet — real, cited, public-domain (HSDB/Haz-Map/CAMEO) data. Empty until
    build_odor_notes.py has run; tolerant of older tables without the threshold columns."""
    try:
        import pandas as pd
        df = pd.read_parquet("odor_notes.parquet")

        def col(name):
            return df[name] if name in df.columns else [None] * len(df)

        out = {}
        for ik, odor, osrc, thr, tsrc in zip(df["inchikey"], col("odor"), col("odor_source"),
                                             col("odor_threshold_ppm"),
                                             col("odor_threshold_source")):
            if not isinstance(ik, str):
                continue
            rec = {}
            if isinstance(odor, str):
                rec["odor"] = odor
                rec["odor_source"] = osrc if isinstance(osrc, str) else None
            if isinstance(thr, (int, float)) and thr == thr:  # numeric and not NaN
                rec["threshold_ppm"] = float(thr)
                rec["threshold_source"] = tsrc if isinstance(tsrc, str) else None
            if rec:
                out[ik.split("-")[0]] = rec
        return out
    except Exception:  # noqa: BLE001 — no table / no pandas
        return {}


_ODOR_TABLE = _load_odor_table()


@app.post("/api/aroma")
def api_aroma(q: Query):
    """Real, cited documented odor (public-domain HSDB/CAMEO) when available — NOT a trained
    model and NOT invented scores. A trained descriptor model comes next (see docs/AROMA.md)."""
    smi = _resolve(q.smiles)
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return {"available": False}
    rec = _ODOR_TABLE.get(Chem.MolToInchiKey(mol).split("-")[0])
    if not rec:
        return {"available": False}
    out = {"available": True}
    if rec.get("odor"):
        notes = [s.strip() for s in rec["odor"].split("\n") if s.strip()]
        concise = [n for n in notes if len(n) <= 90] or notes  # lead with punchy descriptors
        out["documented"] = {"notes": concise[:4], "source": rec.get("odor_source")}
    if rec.get("threshold_ppm") is not None:
        out["threshold"] = {"ppm": rec["threshold_ppm"], "source": rec.get("threshold_source")}
    return out


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("workbench.html").read_text()
