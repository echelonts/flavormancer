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
    return P.substitute(smi, k=q.k)


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("workbench.html").read_text()
