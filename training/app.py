"""
app.py — serving layer for the flavor workbench (demo).

Run:
    pip install fastapi "uvicorn[standard]" pydantic   # plus the SETUP.md env
    uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://<r620-ip>:8000/

Endpoints:
    GET  /              -> the workbench UI (workbench.html)
    POST /api/predict   -> {smiles|name}      -> full flavor read (predict.predict)
    POST /api/neighbors -> {smiles|name, k}   -> substitution candidates

Substitution runs today on Morgan-fingerprint Tanimoto similarity over the
labeled molecules — runnable now, no aroma model required. It upgrades to the
learned aroma-embedding space (embeddings.parquet from train_odor.py) once that
exists: swap the fingerprint index below for those vectors + cosine distance.

Auth / per-seat is stubbed (single open instance) for the demo. For deployment,
put this behind login + per-user history as in the plan; the prediction core
doesn't change.
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

import predict as P  # reuse the unified flavor read

app = FastAPI(title="Flavor Workbench (demo)")

_FPS, _SMI, _KNOWN = [], [], []
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _build_index():
    path = Path("taste_master.parquet")
    if not path.exists():
        print("note: taste_master.parquet not found — substitution search disabled")
        return
    m = pd.read_parquet(path)
    basic = [t for t in ("sweet", "bitter", "umami", "sour", "salty") if t in m.columns]
    for _, r in m.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        if mol is None:
            continue
        _FPS.append(_MORGAN.GetFingerprint(mol))
        _SMI.append(r["smiles"])
        _KNOWN.append([t for t in basic if r[t] == 1])
    print(f"substitution index built: {len(_FPS)} molecules")


_build_index()


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
    if not smi or not _FPS:
        return {"neighbors": []}
    mol = Chem.MolFromSmiles(smi)
    fp = _MORGAN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(fp, _FPS)
    self_smi = Chem.MolToSmiles(mol)
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    out = []
    for i in ranked:
        if _SMI[i] == self_smi:
            continue
        out.append({"smiles": _SMI[i], "similarity": round(sims[i], 3),
                    "known_tastes": _KNOWN[i]})
        if len(out) >= q.k:
            break
    return {"neighbors": out}


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("workbench.html").read_text()
