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

import contextlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import predict as P  # the unified flavor read + substitution search
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from rdkit import Chem

app = FastAPI(title="Flavor Workbench (demo)")

# --- warming-up gate -------------------------------------------------------------------------
# Model heads load on a background thread (predict._load_all_models) so uvicorn binds instantly.
# Until they're ready we serve a friendly self-refreshing page (HTML nav) / a clean 503 (API),
# instead of the old ~34 s startup 502. /api/status and /healthz stay open so the page can poll.
_WARMING_OPEN = {"/api/status", "/healthz", "/favicon.ico"}

_WARMING_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Flavormancer — warming up</title>
<link rel=icon type=image/png href="/static/favicon.png">
<meta http-equiv=refresh content=15>
<style>
 @font-face{font-family:'Cinzel Decorative';src:url('/static/wordmark.ttf') format('truetype');font-weight:700;font-display:swap}
 @font-face{font-family:'Grenze Gotisch';src:url('/static/headerfont.ttf') format('truetype');font-weight:700;font-display:swap}
 :root{--brand-1:#8A6BE0;--brand-2:#2BC4C4;--accent:#E0913C;--cream:#D9AB74;--ink:#0B0F14;--muted:#9BA6B0}
 *{box-sizing:border-box}html,body{margin:0;height:100%}
 body{background:radial-gradient(1200px 640px at 50% -12%,#161d29,#080B10 62%);color:var(--cream);
   font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;display:flex;align-items:center;justify-content:center;padding:24px}
 .box{max-width:460px;width:100%;text-align:center}
 header{display:flex;flex-direction:column;align-items:center;gap:6px;margin-bottom:6px}
 header img{width:54px;height:54px;object-fit:contain;filter:drop-shadow(0 2px 8px rgba(0,0,0,.5))}
 .wordmark{font-family:'Grenze Gotisch','Cinzel Decorative',Georgia,serif;font-size:38px;line-height:1;
   letter-spacing:.02em;background:linear-gradient(100deg,#8A6BE0,#4E84C8 46%,#2BC4C4);-webkit-background-clip:text;
   background-clip:text;color:transparent;margin:2px 0 0}
 .tagline{font-family:'Cinzel Decorative',Georgia,serif;font-size:12.5px;letter-spacing:.05em;color:var(--muted)}
 .flask{width:150px;height:150px;margin:14px auto 6px;display:block}
 .loader-ring{transform-origin:70px 70px;animation:ringspin 1.15s linear infinite}
 @keyframes ringspin{to{transform:rotate(360deg)}}
 .bub{opacity:0;transform-box:fill-box;transform-origin:center;animation:bub 1.7s ease-in infinite}
 .b1{animation-delay:0s}.b2{animation-delay:.5s}.b3{animation-delay:.9s}.b4{animation-delay:1.3s}
 @keyframes bub{0%{opacity:0;transform:translateY(6px) scale(.4)}20%{opacity:.9}100%{opacity:0;transform:translateY(-18px) scale(.95)}}
 .wisp{stroke-dasharray:5 9;transform-box:fill-box;transform-origin:bottom;animation:wisp 2.2s linear infinite,wispRise 3.4s ease-in-out infinite}
 .w1{opacity:.85;animation-delay:0s,0s}.w2{opacity:.6;animation-delay:.5s,.4s}
 .w3{opacity:.55;animation-delay:1s,.9s}.w4{opacity:.5;animation-delay:1.5s,1.3s}
 @keyframes wisp{to{stroke-dashoffset:-28}}
 @keyframes wispRise{0%,100%{transform:translateY(2px) scaleY(.96)}50%{transform:translateY(-2px) scaleY(1.02)}}
 h1{font-family:'Cinzel Decorative',Georgia,serif;font-size:18px;letter-spacing:.06em;color:var(--cream);margin:2px 0 4px}
 .sub{color:var(--muted);font-size:13px;margin:0 0 20px}
 .bar{height:10px;border-radius:6px;background:#141b26;overflow:hidden;border:1px solid #2a3644}
 .fill{height:100%;width:0;border-radius:6px;background:linear-gradient(90deg,var(--brand-1),var(--brand-2));transition:width .5s ease}
 .stat{display:flex;justify-content:space-between;margin-top:10px;font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}
 .cantrip{margin-top:18px;min-height:1.2em;font-family:'Cinzel Decorative',Georgia,serif;font-size:12.5px;color:var(--brand-2);opacity:.9}
 @media(prefers-reduced-motion:reduce){.loader-ring,.bub,.wisp{animation:none}}
</style></head><body>
<div class=box>
 <header>
   <img src="/static/logo.png" alt="">
   <div class=wordmark>Flavormancer</div>
   <div class=tagline>taste &amp; aroma prediction from chemical structure</div>
 </header>
 <svg class=flask viewBox="0 0 140 140" aria-hidden=true>
   <defs><linearGradient id=g x1=0 y1=0 x2=1 y2=1><stop offset=0 stop-color=#7C5CBF /><stop offset=1 stop-color=#2BC4C4 /></linearGradient></defs>
   <circle cx=70 cy=70 r=62 fill=none stroke="url(#g)" stroke-width=3 opacity=.55 />
   <circle class=loader-ring cx=70 cy=70 r=62 fill=none stroke="url(#g)" stroke-width=3 stroke-linecap=round stroke-dasharray="80 320"/>
   <path d="M58 44 h24 v14 l16 34 a6 6 0 0 1 -5.5 8.4 h-45 a6 6 0 0 1 -5.5 -8.4 l16 -34 z" fill="rgba(43,196,196,.10)" stroke="url(#g)" stroke-width=3 stroke-linejoin=round/>
   <path d="M56 44 h28" stroke="url(#g)" stroke-width=3.4 stroke-linecap=round/>
   <path d="M50.5 74 L89.5 74 L98 92 a6 6 0 0 1 -5.5 8.4 h-45 a6 6 0 0 1 -5.5 -8.4 Z" fill="url(#g)" opacity=.72 />
   <circle class="bub b1" cx=64 cy=90 r=2.4 fill=#EAF6F4 /><circle class="bub b2" cx=73 cy=93 r=1.8 fill=#EAF6F4 />
   <circle class="bub b3" cx=77 cy=87 r=2.1 fill=#EAF6F4 /><circle class="bub b4" cx=68 cy=95 r=1.5 fill=#EAF6F4 />
   <g fill=none stroke-linecap=round>
     <path class="wisp w1" d="M69 72 C63 64 75 58 69 50 C63 43 77 36 70 28 C65 22 73 16 69 9" stroke=#D9AB74 stroke-width=2.4 />
     <path class="wisp w2" d="M63 71 C57 64 69 59 62 52 C56 46 66 40 62 33 C59 28 64 24 62 19" stroke=#2BC4C4 stroke-width=2 />
     <path class="wisp w3" d="M76 71 C82 64 70 59 77 52 C83 46 73 41 77 34 C79 30 75 26 77 22" stroke=#8A6BE0 stroke-width=2 />
     <path class="wisp w4" d="M70 73 C66 68 74 63 70 57 C67 52 72 48 70 43" stroke=#D9AB74 stroke-width=1.7 />
   </g>
 </svg>
 <h1>Warming the cauldron…</h1>
 <p class=sub>Summoning the flavor &amp; aroma heads into memory. This happens once, at startup.</p>
 <div class=bar><div class=fill id=fill></div></div>
 <div class=stat><span id=count>Loading models…</span><span id=eta></span></div>
 <div class=cantrip id=cantrip>Stoking the athanor…</div>
</div>
<script>
 var CANTRIPS=['Stoking the athanor…','Unrolling the aroma grimoire…','Awakening the descriptor heads…',
   'Charging the olfactory runes…','Tempering the taste engines…','Aligning the flavor lattice…','Distilling first essences…'];
 var ci=0;setInterval(function(){ci=(ci+1)%CANTRIPS.length;document.getElementById('cantrip').textContent=CANTRIPS[ci];},5000);
 function poll(){
   fetch('/api/status',{cache:'no-store'}).then(function(r){return r.json();}).then(function(s){
     if(s.ready){location.reload();return;}
     var t=s.total||0,l=s.loaded||0,pct=t?Math.round(l/t*100):0;
     document.getElementById('fill').style.width=pct+'%';
     document.getElementById('count').textContent=t?(l+' / '+t+' heads summoned'):'Discovering heads…';
     var eta='';
     if(l>0&&t>l&&s.elapsed){var per=s.elapsed/l;eta='~'+Math.max(1,Math.round(per*(t-l)))+'s remaining';}
     document.getElementById('eta').textContent=eta;
   }).catch(function(){}).finally(function(){setTimeout(poll,1000);});
 }
 poll();
</script></body></html>"""


@app.get("/api/status")
def api_status():
    """Model-load progress for the warming-up page (loaded/total heads, phase, ready, elapsed)."""
    return P.load_status()


@app.get("/healthz")
def healthz():
    """Liveness/readiness for systemd + proxies: ok once the process is up, ready once models load."""
    return {"ok": True, "ready": P.MODELS_READY.is_set()}


@app.get("/api/heads")
@lru_cache(maxsize=1)
def api_heads():
    """The full head catalog grouped by category (taste / aroma / mouthfeel / safety) with AUROC —
    for the modal Heads card and the library category pickers."""
    return P.head_catalog()


@app.middleware("http")
async def _warming_gate(request: Request, call_next):
    _p = request.url.path
    # /static/* stays open so the warming page's own fonts/logo/favicon load while models warm
    if not P.MODELS_READY.is_set() and _p not in _WARMING_OPEN and not _p.startswith("/static/"):
        wants_html = request.method == "GET" and (
            request.url.path == "/" or "text/html" in request.headers.get("accept", ""))
        if wants_html:
            return HTMLResponse(_WARMING_HTML, status_code=503, headers={"Retry-After": "5"})
        return JSONResponse({"warming": True, **P.load_status()}, status_code=503,
                            headers={"Retry-After": "5"})
    return await call_next(request)


def _load_name2smiles():
    """Local name -> SMILES index (instant, offline) so library/demo ingredients resolve without a
    PubChem round-trip. Built from master_enrichment.parquet (~8k named molecules) + the suggest
    CSV. Only genuinely-unknown names fall through to live PubChem in _resolve()."""
    idx = {}
    with contextlib.suppress(Exception):  # table absent / no pandas; live lookup still covers it
        import pandas as pd
        df = pd.read_parquet(P.artifact("master_enrichment.parquet"))
        for nm, smi in zip(df["name"], df["smiles"]):
            if isinstance(nm, str) and isinstance(smi, str) and nm.strip() and smi.strip():
                idx.setdefault(nm.strip().lower(), smi)
    with contextlib.suppress(Exception):  # no suggest file; fine
        import csv
        with open(P.artifact("flavor_volatiles.csv"), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("name") and r.get("smiles"):
                    idx.setdefault(r["name"].strip().lower(), r["smiles"])
    return idx


_NAME2SMILES = _load_name2smiles()


@lru_cache(maxsize=8192)
def _resolve(text: str):
    """Accept a SMILES or a compound name; return canonical SMILES or None. Memoized. Tries a
    local name index first (instant, offline) so library/demo molecules never touch the network;
    only unknown names hit PubChem live (~1-2 s), which is why caching + the index matter."""
    text = (text or "").strip()
    if Chem.MolFromSmiles(text):
        return text
    hit = _NAME2SMILES.get(text.lower())
    if hit and Chem.MolFromSmiles(hit):
        return hit
    with contextlib.suppress(Exception):
        import pubchempy as pcp
        hits = pcp.get_compounds(text, "name")
        if hits and hits[0].canonical_smiles:
            return hits[0].canonical_smiles
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
    """skeleton -> (common, IUPAC), sourced from master_enrichment — the ONE base table (#224).

    This used to read properties.parquet and then merge iupac_backfill.parquet itself, which
    duplicated work build_enrichment.py had already done and silently diverged from it in three
    ways. The enrichment build resolves a name by curated list first, then PubChem common name,
    then IUPAC name, then a molecular-formula fallback — and it un-inverts CAS-style ordering
    along the way. None of that reached this table, so 754 molecules the grid displayed by name
    (cedrol, fenchyl alcohol, hydroxycitronellal, musk ketone...) resolved to nothing here. The
    old merge also read only `iupac_name` from the backfill and ignored the `common_name` column,
    so every PubChem Title the crawler recovered was thrown away.

    Reading the resolved name straight off the base table makes the UI's lookup and the grid
    agree by construction rather than by coincidence.
    """
    try:
        import pandas as pd
        df = pd.read_parquet(P.artifact("master_enrichment.parquet"))
    except Exception:  # noqa: BLE001 — no table / no pandas; live lookups still cover it
        return {}
    out = {}
    iupac = dict(zip(df.get("inchikey_skel", []), df.get("iupac_name", [])))  # optional column
    for skel, name in zip(df["inchikey_skel"], df["name"]):
        if isinstance(skel, str) and isinstance(name, str) and name.strip():
            u = iupac.get(skel)
            out[skel] = (name.strip(), u if isinstance(u, str) else None)
    return out


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


def _name_local(smi):
    """Common name from the precomputed table ONLY (instant, no network) — for hot loops like
    the Formulation Studio's candidate ranking, where a live PubChem call per candidate would
    stall the request. Returns None for molecules not in the table (they're simply skipped)."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return None
    hit = _NAME_TABLE.get(Chem.MolToInchiKey(mol).split("-")[0])
    return hit[0] if hit else None


class Query(BaseModel):
    smiles: str
    k: int = 50  # a generous cap; neighbors/substitutes return every match above a similarity floor
                 # (up to k), so the UI scrolls the qualifying set instead of a fixed short list


@app.post("/api/predict")
def api_predict(q: Query):
    smi = _resolve(q.smiles)
    if not smi:
        return {"error": f"Couldn't resolve '{q.smiles}' to a structure. "
                         f"Enter a valid SMILES or a recognized compound name."}
    out = P.predict(smi, include_aroma=True)
    out["flavor_tags"] = _read_tags(smi, out)
    out["references"] = _references(smi)
    return out


def _load_spectra():
    """inchikey-skeleton -> [available spectra types] from spectra.parquet (build_spectra.py):
    public-domain PubChem availability metadata. Empty until the crawl has run."""
    try:
        import pandas as pd
        df = pd.read_parquet(P.artifact("spectra.parquet"))
        labels = [("has_ms", "MS"), ("has_ir", "IR"), ("has_nmr", "NMR"),
                  ("has_uv", "UV"), ("has_raman", "Raman")]
        out = {}
        for _, r in df.iterrows():
            ik = r.get("inchikey")
            if isinstance(ik, str):
                out[ik.split("-")[0]] = [lab for col, lab in labels if bool(r.get(col))]
        return out
    except Exception:  # noqa: BLE001 — not crawled yet
        return {}


_SPECTRA = _load_spectra()


def _spectra_flags(inchikey):
    return _SPECTRA.get(inchikey.split("-")[0], [])


def _references(smi):
    """Deep links to the authoritative public pages for this molecule — where the spectra
    (IR / MS / UV / NMR), GC retention indices, and full literature live. We LINK rather than
    host: PubChem is public domain, but NIST WebBook data is licensed for individual use only,
    so redistribution isn't clean — a hyperlink always is."""
    import urllib.parse
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    ik = Chem.MolToInchiKey(mol)
    have = _spectra_flags(ik)  # which spectra PubChem actually has (public-domain availability metadata)
    note = ("PubChem has " + " · ".join(have) if have else "identity, properties, spectra")
    refs = [{"label": "PubChem", "note": note,
             "url": f"https://pubchem.ncbi.nlm.nih.gov/#query={urllib.parse.quote(ik)}",
             "spectra": have}]
    with contextlib.suppress(Exception):  # InChI generation can fail on odd valences
        inchi = Chem.MolToInchi(mol)
        if inchi:
            refs.append({"label": "NIST WebBook", "note": "IR / MS spectra, GC retention index",
                         "url": "https://webbook.nist.gov/cgi/cbook.cgi?InChI="
                                + urllib.parse.quote(inchi) + "&Units=SI"})
    return refs


def _read_tags(smi, out):
    """Plain-folk 'what is this?' tags for the read: the tastes it carries, the aroma notes it
    reads as, and any everyday flavor it's the character molecule of (banana, saffron…). So a
    non-chemist sees 'banana · fruity · sweet' at a glance instead of only probabilities."""
    tastes = [t for t in ("sweet", "bitter", "umami")
              if isinstance(out.get(t), (int, float)) and out[t] >= 0.5]
    if out.get("sour"):
        tastes.append("sour")
    if out.get("salty"):
        tastes.append("salty")
    aromas = [d["odor"] for d in _aroma_tags(smi)]  # documented-or-confident aroma notes
    mol = Chem.MolFromSmiles(smi)
    flavors = []
    if mol is not None:
        flavors = _FLAVOR_BY_SKEL.get(Chem.MolToInchiKey(mol).split("-")[0], [])
    # a word can be BOTH a curated flavor and an aroma descriptor (coconut, banana, citrus…) —
    # show it once, as the richer flavor tag; also don't repeat a taste as an aroma
    seen = set(flavors) | set(tastes)
    aromas = [a for a in aromas if a not in seen]
    return {"tastes": tastes, "aromas": aromas[:6], "flavors": flavors}


def _aroma_tags(smi, k=3):
    """A few aroma descriptor tags for a molecule: keyword-derived from documented HSDB odor
    when the molecule is in the corpus (source 'found'), else the model's confident predictions."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return []
    rec = _ODOR_TABLE.get(Chem.MolToInchiKey(mol).split("-")[0])
    if rec and rec.get("odor"):
        with contextlib.suppress(Exception):  # vocab module missing; fall through to predicted
            from build_aroma_dataset import tag as _odor_tag
            found = sorted(_odor_tag(rec["odor"]))[:k]
            if found:
                return [{"odor": t, "source": "found"} for t in found]
    pa = P.predict_aroma(smi)
    return [{"odor": d["odor"], "source": "predicted"}
            for d in pa.get("descriptors", []) if d.get("confident")][:k]


def _aroma_tags_cheap(smi, precomputed, k=3):
    """Same as _aroma_tags — documented HSDB odor first ('found') — but for the PREDICTED
    fallback it reuses aromas already computed in the substitution index instead of re-running
    the 24 heads. Identical result to _aroma_tags, without the per-molecule model cost."""
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is not None:
        rec = _ODOR_TABLE.get(Chem.MolToInchiKey(mol).split("-")[0])
        if rec and rec.get("odor"):
            with contextlib.suppress(Exception):  # vocab module missing; fall through to predicted
                from build_aroma_dataset import tag as _odor_tag
                found = sorted(_odor_tag(rec["odor"]))[:k]
                if found:
                    return [{"odor": t, "source": "found"} for t in found]
    return [{"odor": a, "source": "predicted"} for a in (precomputed or [])][:k]


@app.post("/api/neighbors")
def api_neighbors(q: Query):
    smi = _resolve(q.smiles)
    if not smi:
        return {"neighbors": []}
    res = P.substitute(smi, k=q.k, min_similarity=0.30)  # every structural look-alike above the floor
    for n in res.get("neighbors", []):  # enrich each candidate: structure + names + aroma + GRAS
        n["svg"] = _svg(n["smiles"], 132, 96)
        nm = _names(n["smiles"])
        n["name"], n["iupac"] = nm[0], nm[1]
        # documented odor first (fast lookup); predicted fallback reuses the index's precomputed
        # aromas so the heads never re-run (was ~1.3s x k). Same result as _aroma_tags.
        n["aroma"] = _aroma_tags_cheap(n["smiles"], n.pop("aromas", []))
        _m = Chem.MolFromSmiles(n["smiles"])
        n["gras"] = bool(_m is not None and Chem.MolToInchiKey(_m).split("-")[0] in P._GRAS)
    return res


@app.post("/api/substitutes")
def api_substitutes(q: Query):
    """Profile-based substitutes: molecules whose predicted taste+aroma head scores line up
    closest with the query — the taste/smell-alikes (vs /api/neighbors' structural look-alikes)."""
    smi = _resolve(q.smiles)
    if not smi:
        return {"substitutes": []}
    res = P.substitutes(smi, k=q.k, min_match=0.45)  # every taste/aroma-alike above the floor
    for n in res.get("substitutes", []):  # same enrichment as neighbors: structure + names + aroma + GRAS
        n["svg"] = _svg(n["smiles"], 132, 96)
        nm = _names(n["smiles"])
        n["name"], n["iupac"] = nm[0], nm[1]
        n["aroma"] = _aroma_tags_cheap(n["smiles"], n.pop("aromas", []))
        _m = Chem.MolFromSmiles(n["smiles"])
        n["gras"] = bool(_m is not None and Chem.MolToInchiKey(_m).split("-")[0] in P._GRAS)
    return res


@app.post("/api/precomputed")
def api_precomputed(q: Query):
    """Fast check: is this molecule's profile already in the index (instant read) or does it need a
    fresh 183-head compute? Lets the UI show a 'conjuring a fresh reading' note for novel molecules."""
    smi = _resolve(q.smiles)
    return {"precomputed": bool(smi and P.is_precomputed(smi))}


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
    return {"common": common, "iupac": iupac, "smiles": smi, "formula": _formula(smi)}


def _formula(smi):
    """Hill-system molecular formula (e.g. C9H16O2) — a compact 4th identifier alongside
    common / IUPAC / SMILES. None for an unparseable SMILES."""
    from rdkit.Chem import rdMolDescriptors
    m = Chem.MolFromSmiles(smi) if smi else None
    return rdMolDescriptors.CalcMolFormula(m) if m is not None else None


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
        with contextlib.suppress(Exception):  # no MMFF params for some atoms; unoptimized still fine
            AllChem.MMFFOptimizeMolecule(mol)
        return {"molblock": Chem.MolToMolBlock(mol)}
    except Exception:  # noqa: BLE001 — RDKit build without embedding etc.; degrade gracefully
        return {"molblock": None}


@app.post("/api/stereoisomers")
def api_stereoisomers(q: Query):
    """Every stereoisomer of the queried molecule — all R/S centers AND E/Z double bonds — each
    as a card: stereo label, structure SVG, name, and any ISOMER-SPECIFIC documented odor/taste
    (keyed by full InChIKey, so R-carvone spearmint vs S-carvone caraway show through where
    PubChem records them). The trained models are achiral, so the *difference* is documented, not
    predicted — this makes that difference explorable instead of hidden."""
    smi = _resolve(q.smiles)
    if not smi:
        return {"isomers": []}
    isos = P.stereoisomers(smi)
    for it in isos:
        it["svg"] = _svg(it["smiles"], 150, 108)
        it["name"] = (_names(it["smiles"]) or (None, None))[0]
        doc = _documented_by_full(it["inchikey"])
        if doc.get("odor"):
            it["odor"] = doc["odor"]
        if doc.get("taste"):
            it["taste"] = doc["taste"]
    n_doc = sum(1 for it in isos if it.get("odor") or it.get("taste"))
    return {"isomers": isos, "n": len(isos), "n_documented": n_doc}


@app.get("/static/{fname}")
def _static(fname: str):
    """Serve vendored static assets (3Dmol.js, the logo) locally so the demo is self-contained."""
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    p = Path("static") / Path(fname).name  # basename only — no path traversal
    if p.exists():
        return FileResponse(str(p))
    raise HTTPException(status_code=404)


_TASTE_RGB = {"sweet": (232, 169, 74), "bitter": (168, 138, 224), "umami": (224, 128, 94),
              "sour": (191, 210, 78), "salty": (99, 166, 224), "tasteless": (184, 192, 198)}


@app.get("/api/card")
def api_card(q: str = "", dl: int = 0):
    """A branded, shareable 'flavor card' PNG for a molecule — structure + the read + the
    share URL. Self-contained (RDKit draws the structure, Pillow composes)."""
    from fastapi import HTTPException
    from fastapi.responses import Response
    smi = _resolve(q)
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        raise HTTPException(status_code=404)
    out = P.predict(smi, include_aroma=False)
    tags = _read_tags(smi, out)
    common, iupac = _names(smi)
    name = common or (q[:1].upper() + q[1:] if q else smi)

    import io

    from PIL import Image, ImageDraw, ImageFont
    from rdkit.Chem.Draw import rdMolDraw2D

    def font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 — font file missing; fall back to default
            return ImageFont.load_default()
    DJ = "/usr/share/fonts/truetype/dejavu/"
    f_title = font("static/headerfont.ttf", 46)
    f_tag = font("static/wordmark.ttf", 17)
    f_name = font(DJ + "DejaVuSans-Bold.ttf", 34)
    f_body = font(DJ + "DejaVuSans.ttf", 17)
    f_mono = font(DJ + "DejaVuSansMono.ttf", 15)
    f_pill = font(DJ + "DejaVuSans-Bold.ttf", 16)
    f_lab = font(DJ + "DejaVuSans-Bold.ttf", 15)

    f_cell = font(DJ + "DejaVuSans.ttf", 12)
    W = 1200
    x = 508  # right column (upper band)
    ink, muted, cream, teal = (231, 237, 234), (148, 162, 169), (217, 171, 116), (43, 196, 196)

    # ---- the FULL readout: all 6 taste heads + all 24 aroma heads (nothing truncated) ----
    taste_src = {"sweet": out.get("sweet"), "bitter": out.get("bitter"), "umami": out.get("umami"),
                 "sour": out.get("sour_predicted"), "salty": out.get("salty_predicted"),
                 "tasteless": out.get("tasteless")}
    taste_cells = sorted(((t, float(v) if isinstance(v, (int, float)) else 0.0)
                          for t, v in taste_src.items()), key=lambda kv: -kv[1])
    # The card is a shareable SNAPSHOT — a PNG can't scroll, and there are 195 heads. So: the 6
    # tastes ALWAYS render (a complete, fixed row you can compare across cards), while aroma,
    # mouthfeel and safety show only what actually FIRES, capped. The labels say "N of M" so a
    # reader knows they're seeing the firing subset, not the whole model.
    AROMA_CAP = 18                                   # 3 rows of 6 — keeps the card readable
    pa = P.predict_aroma(smi)
    # rank by score but decide "fired" from the head's OWN calibrated threshold, which
    # predict_aroma already applied — re-thresholding at a flat 0.5 here would disagree with the
    # modal for exactly the thin heads that needed calibrating
    _descs = sorted(pa.get("descriptors", []), key=lambda d: -d["score"])
    _all_aroma = [(d["odor"], d["score"]) for d in _descs]
    _fired = [(d["odor"], d["score"]) for d in _descs if d.get("confident")]
    aroma_cells = (_fired or _all_aroma[:3])[:AROMA_CAP]   # nothing firing -> top 3, never a blank card
    aroma_total, aroma_fired = len(_all_aroma), len(_fired)

    _mol = Chem.MolFromSmiles(smi)
    mouth_cells = [(d["sensation"], d["score"])
                   for d in (P.predict_mouthfeel(_mol).get("descriptors", []) if _mol else [])
                   if d.get("confident")]
    tox_cells = [(a["assay"], a["probability"])
                 for a in ((out.get("safety") or {}).get("tox_screen") or {}).get("assays", [])
                 if (a.get("probability") or 0) >= 0.5]

    pill_items = ([(fl, cream) for fl in tags.get("flavors", [])[:3]]
                  + [(t, _TASTE_RGB.get(t, teal)) for t in tags.get("tastes", [])]
                  + [(a, teal) for a in tags.get("aromas", [])[:6]])

    # ---- measure pill wrapping (right column) to know where the full-width grid starts ----
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    py = 306
    px = x
    for text, _c in pill_items:
        w = scratch.textlength(text, font=f_pill)
        if px + w + 22 > W - 40:
            px, py = x, py + 40
        px += w + 30
    pills_bottom = py + 40

    # ---- full-width readout grid: 6 columns; TASTE row (6) then AROMA rows (24 = 4 x 6) ----
    GX0, COLS, ROW_H = 40, 6, 40
    colw = (W - 80) / COLS
    grid_top = max(500, pills_bottom + 14)
    taste_row_y = grid_top + 22
    aroma_label_y = taste_row_y + ROW_H + 12
    aroma_grid_y = aroma_label_y + 22
    aroma_rows = (len(aroma_cells) + COLS - 1) // COLS
    content_bottom = aroma_grid_y + aroma_rows * ROW_H
    # mouthfeel + safety only take space when something actually fires
    mouth_label_y = content_bottom + 12 if mouth_cells else None
    mouth_row_y = (mouth_label_y + 22) if mouth_cells else None
    if mouth_cells:
        content_bottom = mouth_row_y + ((len(mouth_cells) + COLS - 1) // COLS) * ROW_H
    tox_label_y = content_bottom + 12 if tox_cells else None
    tox_row_y = (tox_label_y + 22) if tox_cells else None
    if tox_cells:
        content_bottom = tox_row_y + ((len(tox_cells) + COLS - 1) // COLS) * ROW_H
    H = max(560, content_bottom + 56)

    img = Image.new("RGB", (W, H), (15, 19, 25))
    dr = ImageDraw.Draw(img)
    # corner aura
    for cx, cy, col in [(0, 0, (138, 107, 224)), (W, H, (43, 196, 196))]:
        glow = Image.new("RGB", (W, H), (15, 19, 25))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([cx - 380, cy - 320, cx + 380, cy + 320], fill=col)
        img = Image.blend(img, glow, 0.06)
        dr = ImageDraw.Draw(img)

    # header
    with contextlib.suppress(Exception):
        emblem = Image.open("static/logo.png").convert("RGBA").resize((58, 58))
        img.paste(emblem, (40, 30), emblem)
    dr.text((110, 30), "Flavormancer", font=f_title, fill=ink)
    dr.text((112, 82), "taste & aroma from chemical structure", font=f_tag, fill=cream)
    dr.line([40, 122, W - 40, 122], fill=(42, 50, 60), width=1)

    # structure panel (white) — sits in the upper band, above the grid
    panel_bottom = grid_top - 14
    dr.rounded_rectangle([40, 150, 470, panel_bottom], radius=14, fill=(245, 247, 245))
    struct_h = min(340, panel_bottom - 172)
    d2 = rdMolDraw2D.MolDraw2DCairo(410, struct_h)
    d2.drawOptions().padding = 0.12
    d2.DrawMolecule(mol)
    d2.FinishDrawing()
    struct = Image.open(io.BytesIO(d2.GetDrawingText())).convert("RGBA")
    img.paste(struct, (50, 150 + (panel_bottom - 150 - struct_h) // 2), struct)

    # right column — name / IUPAC / SMILES + READS AS pills
    dr.text((x, 158), name[:34], font=f_name, fill=ink)
    if iupac and iupac.lower() != name.lower():
        dr.text((x, 206), ("IUPAC  " + iupac)[:64], font=f_body, fill=muted)
    dr.text((x, 232), smi[:58], font=f_mono, fill=muted)
    dr.text((x, 254), "formula  " + (_formula(smi) or ""), font=f_mono, fill=muted)
    dr.text((x, 282), "READS AS", font=f_lab, fill=muted)
    px, py = x, 306
    for text, c in pill_items:
        w = dr.textlength(text, font=f_pill)
        if px + w + 22 > W - 40:
            px, py = x, py + 40
        dr.rounded_rectangle([px, py, px + w + 22, py + 30], radius=15, outline=c, width=2)
        dr.text((px + 11, py + 6), text, font=f_pill, fill=c)
        px += w + 30

    # a compact meter cell in the grid: label, %, and a mini bar
    def cell(col, cy, label, v, color):
        cx = GX0 + int(col * colw)
        inner = int(colw) - 12
        ptxt = f"{round(v * 100)}%"
        pw = dr.textlength(ptxt, font=f_cell)
        lab = label
        while lab and dr.textlength(lab, font=f_cell) > inner - pw - 8:
            lab = lab[:-1]
        dr.text((cx, cy), lab, font=f_cell, fill=ink)
        dr.text((cx + inner - pw, cy), ptxt, font=f_cell, fill=muted)
        dr.rounded_rectangle([cx, cy + 17, cx + inner, cy + 24], radius=3, fill=(36, 44, 53))
        if v > 0:
            dr.rounded_rectangle([cx, cy + 17, cx + max(2, int(inner * v)), cy + 24], radius=3, fill=color)

    # TASTE MODEL — all 6 heads across one row
    dr.text((GX0, grid_top), "TASTE MODEL · all 6 heads", font=f_lab, fill=muted)
    for i, (t, v) in enumerate(taste_cells):
        cell(i, taste_row_y, t, v, _TASTE_RGB.get(t, teal))

    # AROMA MODEL — only the heads that fire, capped; label states the subset honestly
    _shown = len(aroma_cells)
    _alab = (f"AROMA MODEL · {_shown} of {aroma_total} heads firing"
             + (f" (top {_shown} shown)" if aroma_fired > _shown else "")
             if aroma_fired else f"AROMA MODEL · none of {aroma_total} heads firing · strongest {_shown}")
    dr.text((GX0, aroma_label_y), _alab, font=f_lab, fill=muted)
    for i, (a, v) in enumerate(aroma_cells):
        cell(i % COLS, aroma_grid_y + (i // COLS) * ROW_H, a, v, teal)

    # MOUTHFEEL — trigeminal sensations, only what fires
    if mouth_cells:
        dr.text((GX0, mouth_label_y), f"MOUTHFEEL · {len(mouth_cells)} of 5 sensations firing",
                font=f_lab, fill=muted)
        for i, (m, v) in enumerate(mouth_cells):
            cell(i % COLS, mouth_row_y + (i // COLS) * ROW_H, m, v, cream)

    # SAFETY — Tox21 assays, only when flagged; caution-only, never a determination
    if tox_cells:
        dr.text((GX0, tox_label_y), f"SAFETY · {len(tox_cells)} Tox21 assay(s) flagged — caution-only, "
                "indicative in-vitro activity, NOT a toxicity determination", font=f_lab, fill=muted)
        for i, (a, v) in enumerate(tox_cells):
            cell(i % COLS, tox_row_y + (i // COLS) * ROW_H, a, v, (192, 85, 58))

    # footer
    fy = H - 54
    dr.line([40, fy, W - 40, fy], fill=(42, 50, 60), width=1)
    share = "flavormancer.echelonts.net/?q=" + (common or q or smi)
    dr.text((40, fy + 14), share, font=f_mono, fill=teal)
    tw = dr.textlength("before you pour", font=f_tag)
    dr.text((W - 40 - tw, fy + 12), "before you pour", font=f_tag, fill=cream)

    buf = io.BytesIO()
    img.save(buf, "PNG")
    data = buf.getvalue()
    fn = "".join(ch for ch in (common or "molecule") if ch.isalnum() or ch in "-_") or "molecule"
    headers = {"Content-Disposition": f'attachment; filename="flavormancer-{fn}.png"'} if dl else {}
    return Response(content=data, media_type="image/png", headers=headers)


class RecipeIngredient(BaseModel):
    name: str = ""
    smiles: str = ""
    ppm: float | None = None
    carries: list[str] = []
    volatility: str = ""
    dose_basis: str = ""


class RecipeCardQuery(BaseModel):
    ingredients: list[RecipeIngredient] = []
    flavors: list[str] = []          # target flavors this recipe was designed for
    notes: list[str] = []            # target aroma notes
    title: str = ""                  # optional recipe name
    note: str = ""                   # the "how it was dosed" caption


@app.post("/api/recipe_card")
def api_recipe_card(rc: RecipeCardQuery, dl: int = 0):
    """A branded, shareable 'recipe card' PNG for a designed/analyzed formulation —
    the bench-sheet parity of /api/card. Lists each ingredient (structure swatch,
    name, formula, ppm, volatility, what it carries) under the target profile, with
    the same honest 'directional, not calibrated' footing as the Studio itself.

    SMILES/formula are recomputed server-side from each ingredient's structure, so the
    card is authoritative even though the recipe rows are posted from the client."""
    import io

    from fastapi.responses import Response
    from PIL import Image, ImageDraw, ImageFont
    from rdkit.Chem.Draw import rdMolDraw2D

    def font(path, size):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001 — font file missing; fall back to default
            return ImageFont.load_default()
    DJ = "/usr/share/fonts/truetype/dejavu/"
    f_title = font("static/headerfont.ttf", 46)
    f_tag = font("static/wordmark.ttf", 17)
    f_sub = font(DJ + "DejaVuSans-Bold.ttf", 22)
    f_name = font(DJ + "DejaVuSans-Bold.ttf", 21)
    f_body = font(DJ + "DejaVuSans.ttf", 15)
    f_mono = font(DJ + "DejaVuSansMono.ttf", 13)
    f_ppm = font(DJ + "DejaVuSans-Bold.ttf", 26)
    f_lab = font(DJ + "DejaVuSans-Bold.ttf", 13)
    f_pill = font(DJ + "DejaVuSans-Bold.ttf", 15)

    ink, muted, cream, teal = (231, 237, 234), (148, 162, 169), (217, 171, 116), (43, 196, 196)
    W = 1000

    # canonicalise ingredients server-side (authoritative formula + structure)
    rows = []
    for ing in rc.ingredients[:14]:  # a bench recipe is a handful of ingredients; cap defensively
        smi = (ing.smiles or "").strip()
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None and ing.name:
            smi2 = _resolve(ing.name)
            mol = Chem.MolFromSmiles(smi2) if smi2 else None
            if mol is not None:
                smi = smi2
        rows.append({"mol": mol, "smiles": Chem.MolToSmiles(mol) if mol else smi,
                     "formula": _formula(smi) if mol else "",
                     "name": ing.name or (_name_local(smi) if mol else "") or smi,
                     "ppm": ing.ppm, "carries": ing.carries, "volatility": ing.volatility})

    ROW_H = 92
    head_h = 210 if (rc.flavors or rc.notes) else 168
    H = max(420, head_h + len(rows) * ROW_H + 78)

    img = Image.new("RGB", (W, H), (15, 19, 25))
    dr = ImageDraw.Draw(img)
    for cx, cy, col in [(0, 0, (138, 107, 224)), (W, H, (43, 196, 196))]:
        glow = Image.new("RGB", (W, H), (15, 19, 25))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([cx - 340, cy - 300, cx + 340, cy + 300], fill=col)
        img = Image.blend(img, glow, 0.06)
        dr = ImageDraw.Draw(img)

    # header
    with contextlib.suppress(Exception):
        emblem = Image.open("static/logo.png").convert("RGBA").resize((58, 58))
        img.paste(emblem, (40, 30), emblem)
    dr.text((110, 30), "Flavormancer", font=f_title, fill=ink)
    dr.text((112, 82), "taste & aroma from chemical structure", font=f_tag, fill=cream)
    dr.line([40, 122, W - 40, 122], fill=(42, 50, 60), width=1)

    # sub-title + target profile pills
    title = rc.title.strip() or "Formulation"
    dr.text((40, 138), title[:52], font=f_sub, fill=ink)
    if rc.flavors or rc.notes:
        dr.text((40, 174), "TARGET", font=f_lab, fill=muted)
        px, py = 118, 170
        for text, c in ([(f, cream) for f in rc.flavors] + [(n, teal) for n in rc.notes]):
            w = dr.textlength(text, font=f_pill)
            if px + w + 22 > W - 40:
                px, py = 118, py + 34
            dr.rounded_rectangle([px, py, px + w + 20, py + 28], radius=14, outline=c, width=2)
            dr.text((px + 10, py + 5), text, font=f_pill, fill=c)
            px += w + 28

    # ingredient rows
    y = head_h
    for r in rows:
        dr.rounded_rectangle([40, y, W - 40, y + ROW_H - 12], radius=12, outline=(42, 50, 60), width=1)
        # structure swatch (white)
        dr.rounded_rectangle([52, y + 10, 152, y + ROW_H - 22], radius=8, fill=(245, 247, 245))
        if r["mol"] is not None:
            d2 = rdMolDraw2D.MolDraw2DCairo(96, ROW_H - 36)
            d2.drawOptions().padding = 0.1
            d2.DrawMolecule(r["mol"])
            d2.FinishDrawing()
            sw = Image.open(io.BytesIO(d2.GetDrawingText())).convert("RGBA")
            img.paste(sw, (54, y + 12), sw)
        # name + identifiers
        dr.text((168, y + 12), r["name"][:40], font=f_name, fill=ink)
        ident = r["formula"] + ("   " + r["smiles"][:40] if r["smiles"] else "")
        dr.text((168, y + 40), ident[:70], font=f_mono, fill=muted)
        carries = ", ".join(r["carries"]) if r["carries"] else ""
        if carries:
            dr.text((168, y + 60), ("carries: " + carries)[:64], font=f_body, fill=teal)
        # ppm + volatility (right-aligned)
        if r["ppm"] is not None:
            ptxt = f"{r['ppm']:g} ppm"
            pw = dr.textlength(ptxt, font=f_ppm)
            dr.text((W - 60 - pw, y + 16), ptxt, font=f_ppm, fill=cream)
        if r["volatility"]:
            vt = r["volatility"] + " volatility"
            vw = dr.textlength(vt, font=f_body)
            dr.text((W - 60 - vw, y + 50), vt, font=f_body, fill=muted)
        y += ROW_H

    # caption (honest scope) + footer
    cap = (rc.note.strip() or "Directional starting recipe — doses balanced by inverse volatility. "
           "Tune on the bench; calibrated intensity comes with your odor-threshold / panel data.")
    dr.text((40, y + 4), ("— " + cap)[:118], font=f_body, fill=muted)
    fy = H - 46
    dr.line([40, fy, W - 40, fy], fill=(42, 50, 60), width=1)
    dr.text((40, fy + 12), "flavormancer.echelonts.net", font=f_mono, fill=teal)
    tw = dr.textlength("before you pour", font=f_tag)
    dr.text((W - 40 - tw, fy + 10), "before you pour", font=f_tag, fill=cream)

    buf = io.BytesIO()
    img.save(buf, "PNG")
    data = buf.getvalue()
    fn = "".join(ch for ch in title.lower().replace(" ", "-") if ch.isalnum() or ch in "-_") or "recipe"
    headers = {"Content-Disposition": f'attachment; filename="flavormancer-{fn}.png"'} if dl else {}
    return Response(content=data, media_type="image/png", headers=headers)


class MixtureQuery(BaseModel):
    ingredients: list[str]
    processes: list[str] = []


class BlendQuery(BaseModel):
    ingredients: list[str]
    weights: list[float] = []
    k: int = 6


@app.post("/api/mixture_to_molecule")
def api_mixture_to_molecule(b: BlendQuery):
    """Collapse a blend to single equivalent molecules: the dose-weighted mean taste+aroma
    profile of the components, then the molecules whose own profile is closest."""
    smis = [s for s in (_resolve(x) for x in b.ingredients) if s]
    if not smis:
        return {"equivalents": []}
    res = P.mixture_to_molecule(smis, weights=b.weights or None, k=b.k)
    for n in res.get("equivalents", []):  # enrich like neighbors/substitutes
        n["svg"] = _svg(n["smiles"], 132, 96)
        nm = _names(n["smiles"])
        n["name"], n["iupac"] = nm[0], nm[1]
        n["aroma"] = _aroma_tags_cheap(n["smiles"], n.pop("aromas", []))
        _m = Chem.MolFromSmiles(n["smiles"])
        n["gras"] = bool(_m is not None and Chem.MolToInchiKey(_m).split("-")[0] in P._GRAS)
    return res


@app.post("/api/mixture")
def api_mixture(m: MixtureQuery):
    """Per-ingredient reads + documented-hazard screen + a single-molecule palette match."""
    smis = [s for s in (_resolve(x) for x in m.ingredients) if s]
    out = P.check_mixture(smis, m.processes)
    reads, palette, aroma_palette = [], set(), set()
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
        aromas = [d["odor"] for d in _aroma_tags(s)]  # this ingredient's aroma notes
        aroma_palette.update(aromas)
        reads.append({"smiles": r["smiles"], "name": _names(s)[0], "svg": _svg(s, 110, 80),
                      "top_taste": tp[0]["taste"] if tp else None, "tastes": tastes, "aromas": aromas[:4],
                      "gras": r["safety"]["gras_status"], "alerts": r["safety"]["structural_alerts"],
                      "tox_flags": r["safety"]["tox_screen"].get("flags", []) if r["applicability"]["in_domain"] else []})
    pal = P.palette_match(sorted(palette), sorted(aroma_palette), k=5)
    for mt in pal.get("matches", []):
        mt["svg"] = _svg(mt["smiles"], 110, 80)
        mt["name"] = _names(mt["smiles"])[0]
    out["ingredients"] = reads
    out["palette"] = pal
    # indicative reaction-template products (augments the documented-hazard screen) — with the
    # product's own predicted taste + aroma so you see what would form, flavor-wise
    rxns = P.reaction_products(smis)
    for rx in rxns:
        rx["svg"] = _svg(rx["smiles"], 110, 80)
        rx["name"] = _names(rx["smiles"])[0]
        rx["aromas"] = [d["odor"] for d in _aroma_tags(rx["smiles"])][:3]
        pr = P.predict(rx["smiles"])
        rtastes = [t for t in ("sweet", "bitter", "umami")
                   if isinstance(pr.get(t), (int, float)) and pr[t] >= 0.5]
        if pr.get("sour"):
            rtastes.append("sour")
        if pr.get("salty") is True:
            rtastes.append("salty")
        rx["tastes"] = rtastes
    out["reactions"] = rxns
    return out


# ── Formulation Studio ──────────────────────────────────────────────────────
# A recipe (ingredients + optional ppm) -> blended note-profile, dosing-balance /
# overpowering-component flag, hazard screen, and (with a target) a gap analysis.
# The whole point: read a formulation "before you pour" and save bench runs.
_VOL_W = {"high": 3.0, "moderate": 2.0, "low": 1.0}
_PROFILE_FLOOR = 0.35  # ignore each molecule's faint (<0.35 prob) heads so noise can't stack


class FormulationQuery(BaseModel):
    ingredients: list[dict] = []   # [{name|smiles, ppm?}]
    processes: list[str] = []      # high_heat / refining / fermentation
    target: list[str] = []         # desired aroma notes for the gap analysis


@app.post("/api/formulation")
def api_formulation(f: FormulationQuery):
    """Formulation Studio engine — reads a full recipe before it is poured.

    Returns the blended note-profile (which aromas the mix reads as, and which
    ingredient drives each), the dosing balance / overpowering-component flag, a
    documented-hazard screen, and — when a target profile is supplied — a gap
    analysis with concrete add/cut moves.

    HONEST SCOPE (surfaced in `data_gates`): the profile is DIRECTIONAL. Each
    molecule's predicted notes are weighted by OAV where odor thresholds are
    loaded, else by mass x volatility. It is NOT a calibrated finished-blend
    intensity map — suppression/synergy and true intensity need the customer's
    odor-threshold / panel data (a learned mixture model)."""
    # Resolve every ingredient concurrently — a name is a live PubChem lookup (~1-2 s each), so
    # a serial loop makes a big formula crawl. HTTP + name lookups release the GIL; memoized.
    def _resolve_one(it):
        raw = (it.get("smiles") or it.get("name") or "").strip()
        if not raw:
            return None
        smi = _resolve(raw)
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return {"raw": raw, "unresolved": True}
        smi = Chem.MolToSmiles(m)  # canonical, so it keys against analyze_balance's rows
        ppm = it.get("ppm")
        try:
            ppm = float(ppm) if ppm not in (None, "") else None
        except (TypeError, ValueError):
            ppm = None
        # local-table name (instant) or the user's own input — avoids a SECOND live PubChem
        # round-trip per ingredient (_resolve already paid one); "decanal" reads fine as-is.
        return {"raw": raw, "smiles": smi, "ppm": ppm, "name": _name_local(smi) or raw}

    resolved, unresolved = [], []
    ings = [it for it in f.ingredients if (it.get("smiles") or it.get("name") or "").strip()]
    if ings:
        with ThreadPoolExecutor(max_workers=min(8, len(ings))) as ex:
            for out in ex.map(_resolve_one, ings):
                if out is None:
                    continue
                (unresolved.append(out["raw"]) if out.get("unresolved") else resolved.append(out))
    if not resolved:
        return {"error": "no resolvable ingredients", "unresolved": unresolved, "profile": []}

    # dosing balance — OAV ranking where thresholds are loaded, else volatility tier
    bal = P.analyze_balance([{"smiles": r["smiles"], "ppm": r["ppm"], "name": r["name"]}
                            for r in resolved])
    per = {row["smiles"]: row for row in bal.get("per_ingredient", []) if row.get("smiles")}

    # per-ingredient odor-impact weight (cheap, serial)
    for r in resolved:
        row = per.get(r["smiles"], {})
        oav = row.get("OAV")
        if oav:
            w = float(oav)                                    # quantitative: odor activity value
        else:
            vt = (row.get("volatility") or "moderate").split()[0]
            w = (r["ppm"] or 1.0) * _VOL_W.get(vt, 2.0)       # directional: mass x volatility tier
        r["weight"] = round(w, 3)

    # aroma prediction is the per-molecule cost (24 RF heads). It's CPU-bound and does NOT
    # release the GIL cleanly, so threading it hurts (contention) — keep it serial. Speed comes
    # from memoization (repeats/re-analyses are instant) and the startup pre-warm of demo mols.
    aromas = [P.predict_aroma(r["smiles"]) for r in resolved]

    # weighted aggregate note-profile: sum (weight x per-molecule note score) across ingredients
    profile, contrib = {}, {}
    for r, pa in zip(resolved, aromas):
        r["aromas"] = [d["odor"] for d in pa.get("top", [])][:5]
        for d in pa.get("descriptors", []):
            if d["score"] < _PROFILE_FLOOR:
                continue
            c = r["weight"] * d["score"]
            profile[d["odor"]] = profile.get(d["odor"], 0.0) + c
            contrib.setdefault(d["odor"], []).append((r["name"], c))
    total = sum(profile.values()) or 1.0
    prof = sorted(
        ({"note": n, "pct": round(100 * v / total, 1),
          "drivers": [nm for nm, _ in sorted(contrib[n], key=lambda t: -t[1])[:2]]}
         for n, v in profile.items()),
        key=lambda d: -d["pct"])

    # overpowering-component flag — the "too heavy in one item" read. Works in BOTH bases
    # because it uses the blend weights we just computed, not only the quantitative OAV branch.
    overpowering = None
    wsum = sum(r["weight"] for r in resolved) or 1.0
    if len(resolved) > 1:
        top = max(resolved, key=lambda r: r["weight"])
        share = top["weight"] / wsum
        if share > 0.55:
            overpowering = {"name": top["name"], "share": round(100 * share),
                            "drives": [p["note"] for p in prof if top["name"] in p.get("drivers", [])][:3]}

    # target gap analysis — what the brief asks for vs what the blend reads as
    gap = None
    if [t for t in f.target if t.strip()]:
        tset = [t.strip().lower() for t in f.target if t.strip()]
        pmap = {p["note"]: p for p in prof}
        under, over, on_target = [], [], []
        for t in tset:
            hit = pmap.get(t)
            pct = hit["pct"] if hit else 0.0
            if pct < 8:                                        # target note missing / too faint
                sug = P.palette_match([], [t], k=8)
                gras_adds, other_adds = [], []                 # prefer food-safe (GRAS) carriers
                for mt in sug.get("matches", []):
                    nm = _name_local(mt["smiles"])             # local-only (no network) — named carriers, fast
                    if not nm or nm in gras_adds or nm in other_adds:
                        continue
                    cmol = Chem.MolFromSmiles(mt["smiles"])     # cheap food-listed lookup — no full predict() pipeline
                    is_gras = cmol is not None and Chem.MolToInchiKey(cmol).split("-")[0] in P._GRAS
                    (gras_adds if is_gras else other_adds).append(nm)
                    if len(gras_adds) >= 2:
                        break
                under.append({"note": t, "pct": pct, "add": (gras_adds + other_adds)[:2]})
            else:
                on_target.append({"note": t, "pct": pct})
        for p in prof:                                         # loud notes nobody asked for
            if p["note"] not in tset and p["pct"] >= 15:
                over.append({"note": p["note"], "pct": p["pct"], "cut": p["drivers"][:1]})
        gap = {"under": under, "over": over[:4], "on_target": on_target}

    haz = P.check_mixture([r["smiles"] for r in resolved], f.processes)
    quant = (bal.get("basis") or "").startswith("quantitative")
    return {
        "ingredients": [{"name": r["name"], "smiles": r["smiles"], "ppm": r["ppm"],
                         "weight": r["weight"], "aromas": r["aromas"],
                         "svg": _svg(r["smiles"], 110, 80)} for r in resolved],
        "unresolved": unresolved,
        "profile": prof,
        "weighting": bal.get("basis"),
        "overpowering": overpowering,
        "balance_warnings": bal.get("balance_warnings", []),
        "impact_ranking": bal.get("impact_ranking", []),
        "gap": gap,
        "active_hazards": haz.get("active_hazards", []),
        "conditional_hazards": haz.get("conditional_hazards", []),
        "data_gates": {
            "intensity": ("Directional note profile — contributions weighted by "
                          + ("OAV (odor thresholds are loaded)." if quant else
                             "mass x volatility. Load odor thresholds for quantitative OAV / calibrated intensity — comes with your data.")),
            "synergy": ("Notes are assumed to add independently. Real blends show suppression / "
                        "synergy (1+1 != 2); a learned mixture model needs formulation->panel "
                        "data (your data) or a licensed set."),
        },
        "scope_note": bal.get("scope_note"),
        "disclaimer": bal.get("disclaimer"),
    }


# ── Recipe generator ────────────────────────────────────────────────────────
# The inverse of the analyzer: pick a target (flavors + notes) and Studio proposes a starting
# formulation — food-safe carriers at rough, volatility-balanced doses — then runs it through
# the analyzer so you see the predicted profile + gap right away. Doses are a starting point.
_VOL_PPM = {"high": 33.0, "moderate": 50.0, "low": 100.0}  # inverse-volatility: aim for balanced contributions


def _load_note_carriers():
    """descriptor/note -> [(smiles, name)] of KNOWN character-impact molecules, from the curated
    flavors.csv + aroma_supplement.csv. The recipe designer prefers these (e.g. gamma-nonalactone
    for coconut, maltol for caramel) over a generic palette match, which can surface poor carriers.

    Mouthfeel agents load separately (_load_mouthfeel_carriers) and are NOT merged here: both files
    key on bare names, so aroma_supplement's `pungent` Piper amides and mouthfeel_supplement's
    TRPV1 agents would collide under one key and lose their modality."""
    import csv
    m = {}
    for path in ("flavors.csv", "aroma_supplement.csv"):
        with contextlib.suppress(Exception), open(path, encoding="utf-8") as fh:  # missing file / bad rows; just skip
            for r in csv.DictReader(fh):
                note = (r.get("flavor") or "").strip().lower()
                smi = (r.get("smiles") or "").strip()
                nm = (r.get("molecule") or "").strip()
                if note and smi:
                    m.setdefault(note, []).append((smi, nm or note))
    return m


def _load_mouthfeel_carriers():
    """sensation -> [(smiles, name)] of curated trigeminal agents (mouthfeel_supplement.csv, same
    schema as flavors.csv). Kept separate from _NOTE_CARRIERS so the modality survives: the design
    pool is built from the HSDB *odor* corpus, which barely contains these (0 of 11 tingling agents,
    1 of 11 astringent), so without folding them in those chips match nothing."""
    import csv
    m = {}
    with contextlib.suppress(Exception), open("mouthfeel_supplement.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            note = (r.get("flavor") or "").strip().lower()
            smi = (r.get("smiles") or "").strip()
            nm = (r.get("molecule") or "").strip()
            if note and smi:
                m.setdefault(note, []).append((smi, nm or note))
    return m


_NOTE_CARRIERS = _load_note_carriers()
_MOUTHFEEL_CARRIERS = _load_mouthfeel_carriers()


class DesignRecipeQuery(BaseModel):
    flavors: list[str] = []
    notes: list[str] = []
    food_safe: bool = True


@app.post("/api/design_recipe")
def api_design_recipe(d: DesignRecipeQuery):
    """Design a STARTING formulation for a desired profile. Pick target flavors + notes; get a
    food-safe recipe (carriers + rough starting ppm) already run through the analyzer, so you see
    its predicted profile and the gap. Doses are a bench starting point — calibrated dosing needs
    odor-threshold / panel data (a data-gate)."""
    picks = []  # [canonical_smiles, [carries...], name]
    seen = {}

    def add(smi, carries, name):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return
        cs = Chem.MolToSmiles(m)
        if cs in seen:
            if carries not in seen[cs][1]:
                seen[cs][1].append(carries)
            return
        row = [cs, [carries], name or _name_local(cs) or cs]
        seen[cs] = row
        picks.append(row)

    # flavors -> their character-impact molecule
    for fl in d.flavors:
        entries = _FLAVORS.get(fl.strip().lower()) or _FLAVORS.get(fl.strip())
        if entries:
            add(entries[0]["smiles"], fl, entries[0].get("molecule"))

    # notes -> a food-safe carrier (GRAS-preferred, named)
    for note in d.notes:
        chosen, fallback = None, None
        # curated character-impact molecules first, then a generic palette match as backstop
        candidates = list(_NOTE_CARRIERS.get(note.strip().lower(), []))
        candidates += [(mt["smiles"], _name_local(mt["smiles"]))
                       for mt in P.palette_match([], [note], k=8).get("matches", [])]
        for smi, nm in candidates:
            if not nm:
                continue
            cmol = Chem.MolFromSmiles(smi)
            # food-listed check: match the InChIKey skeleton against the reference set directly
            # (robust — don't string-match the human-readable status text)
            if cmol is not None and Chem.MolToInchiKey(cmol).split("-")[0] in P._GRAS:
                chosen = (smi, nm)
                break
            fallback = fallback or (smi, nm)
        pick = chosen or (None if d.food_safe else fallback)
        if pick:
            add(pick[0], note, pick[1])

    if not picks:
        return {"recipe": [], "targeted": {"flavors": d.flavors, "notes": d.notes},
                "note": "No food-safe carriers found for that target — try different flavors/notes."}

    recipe = []
    for cs, carries, name in picks:
        pc = P.physchem(Chem.MolFromSmiles(cs))
        vt = pc["qualitative"]["aroma_volatility"].split()[0]
        vp = (pc.get("measured") or {}).get("vapor_pressure_pa")
        if isinstance(vp, (int, float)) and vp > 0:
            # continuous dose from MEASURED vapor pressure: more volatile -> less needed. A decade
            # more volatile drops the starting dose ~20 ppm, clamped to a sane 20-150 ppm band.
            ppm = round(max(20.0, min(150.0, 70.0 - 20.0 * math.log10(vp))), 1)
            basis = "vapor pressure"
        else:
            ppm = _VOL_PPM.get(vt, 50.0)  # fall back to the coarse volatility tier
            basis = "volatility tier"
        recipe.append({"name": name, "smiles": cs, "ppm": ppm,
                       "carries": carries, "volatility": vt, "dose_basis": basis})

    analysis = api_formulation(FormulationQuery(
        ingredients=[{"name": r["smiles"], "ppm": r["ppm"]} for r in recipe],
        target=d.notes, processes=[]))
    return {
        "recipe": recipe,
        "analysis": analysis,
        "targeted": {"flavors": d.flavors, "notes": d.notes},
        "note": ("How these doses are set: each target flavor maps to its character-impact molecule "
                 "and each note to a food-safe (GRAS) carrier, then doses are assigned by INVERSE "
                 "VOLATILITY — from each molecule's MEASURED vapor pressure where available (a decade "
                 "more volatile ≈ 20 ppm lower, clamped 20–150 ppm), else a coarse volatility tier — "
                 "so no single ingredient's odor impact dominates the directional model. It's an "
                 "honest STARTING POINT to tune on the bench; true calibrated dosing needs odor "
                 "thresholds / panel intensities (a data-gate that comes with your data)."),
    }


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
    P.MODELS_READY.wait()  # don't compete with the model load for cores
    for _, s in _SUGGEST:
        _IUPAC[s] = _names(s)[1]


if _SUGGEST:
    threading.Thread(target=_precompute_iupac, daemon=True).start()


# Pre-warm the Formulation Studio's demo molecules (starter formulas) at startup, in the
# background, so the first click on an example is instant. predict_aroma is CPU-bound (~1.3 s
# cold per molecule) but memoized — warming these fills the cache before anyone reaches them.
_FORMULATION_WARM = [
    "vanillin", "ethyl vanillin", "ethyl maltol", "limonene", "citral", "linalool",
    "ethyl butyrate", "menthol", "eucalyptol", "methyl salicylate", "benzaldehyde",
]


def _prewarm_formulation():
    P.MODELS_READY.wait()  # the heads must be loaded before we can warm anything with them
    # Build the substitution index FIRST (the ~8k-row aroma batch) so the first neighbor search
    # never pays the one-time build; the lock in substitute() makes a concurrent request wait.
    with contextlib.suppress(Exception):  # best-effort
        P.substitute("CCO")
        # Warm the PROFILE path too: builds the normalized reference matrix (_profiles_unit, the
        # 8k×178 renormalization) once here instead of on the first /api/substitutes.
        P.substitutes("CCO")
    for n in _FORMULATION_WARM:
        with contextlib.suppress(Exception):  # best-effort warmup; a miss just means a cold first hit
            smi = _resolve(n)
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is not None:
                canon = Chem.MolToSmiles(m)
                P.predict_aroma(canon)   # fills the shared 172-head aroma cache
                P.substitutes(canon)     # taste heads + profile cosine, so the demo chips are instant


threading.Thread(target=_prewarm_formulation, daemon=True).start()


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


# --- Browsable "top" lists for the landing page (model-ranked; discovery, not just search) ---
_TOP_LISTS = {}  # category -> {"label", "items":[{name, smiles, score}]}
_TOP_N = 40      # nicely-named molecules to keep per category
_CAND = 600      # rank this many by probability, then keep the recognizable ones


def _table_name(smi):
    """Common name from the precomputed tables only (instant, offline) — the properties name
    table first, then the odor corpus's PubChem Titles — so the precompute never hits the network."""
    m = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
    if m is None:
        return None
    skel = Chem.MolToInchiKey(m).split("-")[0]
    hit = _NAME_TABLE.get(skel)
    if hit and hit[0]:
        return hit[0]
    rec = _ODOR_TABLE.get(skel)  # odor-corpus common name (PubChem Title)
    return rec.get("name") if rec else None


def _nice(name):
    """A recognizable common name — not a systematic/registry string — so browse lists read well."""
    if not name or len(name) > 38 or name[0] in "[(":
        return False
    return sum(c.isdigit() for c in name) / len(name) <= 0.22


def _rank(smiles_iter, prob_fn):
    """Top-_CAND (smiles, prob) by a per-molecule probability, structure-parsed once."""
    import numpy as np
    rows = [(s, Chem.MolFromSmiles(str(s))) for s in smiles_iter]
    rows = [(s, m) for s, m in rows if m is not None]
    if not rows:
        return []
    probs = prob_fn(np.vstack([P._fp(m) for _, m in rows]))
    order = probs.argsort()[::-1][:_CAND]
    return [(rows[i][0], round(float(probs[i]), 3)) for i in order]


def _named_top(ranked):
    """Keep the recognizable, named molecules from a ranked list, up to _TOP_N."""
    items, seen = [], set()
    for s, p in ranked:
        nm = _table_name(s)
        if _nice(nm) and nm.lower() not in seen:
            seen.add(nm.lower())
            items.append({"name": nm, "smiles": s, "score": p})
        if len(items) >= _TOP_N:
            break
    return items


def _precompute_top_lists():
    """Rank molecules for the landing-page browse lists — taste heads over the labeled set,
    aroma heads over the odor corpus. Model-derived: honest 'what the tool predicts'."""
    import pandas as pd
    with contextlib.suppress(Exception):  # no taste data; skip taste lists
        tm = pd.read_parquet(P.artifact("taste_master.parquet"))
        for taste, clf in P._CLASSIFIERS.items():
            ranked = _rank(tm["smiles"], lambda X, c=clf: c.predict_proba(X)[:, 1])
            _TOP_LISTS[f"taste:{taste}"] = {"label": f"Top {taste}", "items": _named_top(ranked)}
        # salty is a rule/known-label taste (not a model head) — list the LABELED salty molecules
        if "salty" in tm.columns:
            salty = [(s, 1.0) for s in tm.loc[tm["salty"] == 1, "smiles"]]
            items = _named_top(salty)
            if items:
                _TOP_LISTS["taste:salty"] = {"label": "Known salty", "items": items}
    with contextlib.suppress(Exception):  # no odor corpus / vocab; skip aroma lists
        # aroma lists use DOCUMENTED odor (ground truth), not model ranking: the public corpus
        # skews industrial, so ranking by a head surfaces confident-but-odd picks (cyanide under
        # "almond"). Documented examples are real, recognizable, and honest ("documented citrus").
        from build_aroma_dataset import tag as _odor_tag
        od = pd.read_parquet(P.artifact("odor_notes.parquet"))
        by_desc = {}
        for _, r in od.iterrows():
            nm, odor = r.get("name"), r.get("odor")
            if not isinstance(odor, str) or not _nice(nm):
                continue
            for d in _odor_tag(odor):
                by_desc.setdefault(d, []).append({"name": nm, "smiles": r["smiles"]})
        for d, items in by_desc.items():
            if len(items) >= 8:  # only offer descriptors with enough documented examples
                _TOP_LISTS[f"aroma:{d}"] = {"label": f"Documented {d}", "items": items[:_TOP_N]}
# NB: the thread is started at the very end of the module, after _ODOR_TABLE is defined.


@app.get("/api/categories")
def api_categories():
    """The browse categories available on the landing page, once precompute has finished."""
    return {"categories": [{"key": k, "label": v["label"], "n": len(v["items"])}
                           for k, v in _TOP_LISTS.items()]}


@app.get("/api/top")
def api_top(category: str = "", limit: int = 24):
    """A browse list: model-ranked top molecules for a taste/aroma category (each clickable)."""
    lst = _TOP_LISTS.get(category)
    if not lst:
        return {"label": None, "items": []}
    items = [dict(it) for it in lst["items"][:limit]]
    for it in items:
        it["svg"] = _svg(it["smiles"], 96, 68)
    return {"label": lst["label"], "items": items}


def _load_flavor_map():
    """The flavor-space embedding (flavor_map.parquet, built by build_flavor_map.py): 2D (x,y)
    + 3D (x3,y3,z3) coordinates normalized to 0..1 with names — an interactive scatter / cloud."""
    try:
        import pandas as pd
        df = pd.read_parquet(P.artifact("flavor_map.parquet"))
        # UMAP occasionally emits NaN coords for a few near-duplicate rows — drop them so the
        # JSON stays valid (NaN isn't JSON-compliant) and the scatter has no phantom points.
        df = df.dropna(subset=[c for c in ("x", "y", "x3", "y3", "z3") if c in df.columns]).reset_index(drop=True)

        def norm(col):
            v = df[col]
            lo, rng = v.min(), (v.max() - v.min()) or 1.0
            return ((v - lo) / rng).round(4)

        cols = {c: norm(c).tolist() for c in ("x", "y", "x3", "y3", "z3") if c in df.columns}
        smis, labs = df["smiles"].tolist(), df["label"].tolist()
        aromas = df["aroma_label"].tolist() if "aroma_label" in df.columns else [None] * len(smis)
        mouth = (df["mouthfeel_label"].tolist() if "mouthfeel_label" in df.columns
                 else [None] * len(smis))   # dominant trigeminal sensation, for color-by-mouthfeel
        # raw physicochemical values (for the interpretable MW×logP×TPSA axes view — real units)
        mw = df["mw"].tolist() if "mw" in df.columns else [None] * len(smis)
        logp = df["logp"].tolist() if "logp" in df.columns else [None] * len(smis)
        tpsa = df["tpsa"].tolist() if "tpsa" in df.columns else [None] * len(smis)
        pts = []
        for i in range(len(smis)):
            p = {"label": labs[i], "aroma": aromas[i], "mouth": mouth[i], "smiles": smis[i],
                 "name": _table_name(smis[i]) or "",
                 "mw": None if mw[i] != mw[i] else mw[i],      # NaN -> None
                 "logp": None if logp[i] != logp[i] else logp[i],
                 "tpsa": None if tpsa[i] != tpsa[i] else tpsa[i]}
            for c, v in cols.items():
                p[c] = v[i]
            pts.append(p)
        return pts
    except Exception:  # noqa: BLE001 — no map built yet
        return []


_FLAVOR_MAP = None


@app.get("/api/map")
def api_map():
    """The flavor-space map points (built + name-resolved once, then cached)."""
    global _FLAVOR_MAP
    if _FLAVOR_MAP is None:
        _FLAVOR_MAP = _load_flavor_map()
    return {"points": _FLAVOR_MAP}


# --- Flavor designer: reverse search (desired descriptors -> best food-safe molecules) ---
_DESIGN = []          # [{smiles, name, tags:set, gras:bool}]
_DESIGN_DESCS = []    # descriptors with enough molecules to offer as options
_DESIGN_MOUTHFEEL = []  # trained mouthfeel/chemesthesis sensations, offered as their own pick-list
_DESIGN_TASTE = []      # trained taste heads, offered as their own pick-list (namespaced taste:*)
_TERM_ALIASES = {}      # term -> (equivalent terms in the other modality), both directions

# Curated flavor<->note synonyms the string normalizer can't reach: genuinely the same material
# under two names, not a loose association. Deliberately conservative — "lemon"/"citrus" is NOT
# here, because citrus is broader than lemon and equating them would silently widen a search.
_TERM_SYNONYMS = {
    "blackcurrant": "cassis",       # cassis IS blackcurrant
    "tangerine": "mandarin",        # tangerine IS a mandarin
    "orange blossom": "neroli",     # neroli IS orange-blossom
    "cilantro": "coriander",        # cilantro IS coriander leaf
    "chocolate": "cocoa",
    "licorice": "anise",
    "cheese": "cheesy",
    "smoke": "smoky",
    "peppermint": "minty",
    "spearmint": "minty",
}


def _build_term_aliases():
    """flavor <-> note aliases, both directions. Auto-derives spelling/adjective pairs (bread ->
    bready, black pepper -> blackpepper) by normalizing away spacing and common suffixes, then
    folds in the curated synonym table. Auto-derivation means new vocabulary keeps linking up
    without anyone maintaining a list."""
    import re

    def norm(t):
        t = re.sub(r"[^a-z]", "", t.lower())
        for suf in ("y", "ic", "ish"):
            if t.endswith(suf) and len(t) > len(suf) + 3:
                t = t[: -len(suf)]
        return t

    by_norm = {}
    for n in _DESIGN_DESCS:
        by_norm.setdefault(norm(n), set()).add(n)
    pairs = set()
    for f in _FLAVORS:
        for n in by_norm.get(norm(f), ()):
            if n != f:
                pairs.add((f, n))
    for f, n in _TERM_SYNONYMS.items():
        if f in _FLAVORS and n in _DESIGN_DESCS:
            pairs.add((f, n))
    out = {}
    for f, n in pairs:                       # link both ways
        out.setdefault(f, set()).add(n)
        out.setdefault(n, set()).add(f)
    _TERM_ALIASES.clear()
    _TERM_ALIASES.update({k: tuple(sorted(v)) for k, v in out.items()})


def _precompute_design():
    """Index the odor corpus by descriptor (documented tags + model-confident predictions +
    taste), with GRAS status — so the designer can rank food-safe molecules for a target flavor.
    Model inference is BATCHED (one vectorized call per head over all molecules) — per-molecule
    RandomForest calls over ~2.3k molecules would take minutes."""
    with contextlib.suppress(Exception):  # no corpus/models; designer just stays empty
        from collections import Counter

        import numpy as np
        import pandas as pd
        from build_aroma_dataset import tag as _odor_tag
        od = pd.read_parquet(P.artifact("odor_notes.parquet"))
        rows = []  # (smiles, name, mol_skeleton, {documented tags})
        for smi, nm, odor in zip(od["smiles"], od.get("name", [None] * len(od)), od["odor"]):
            mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) else None
            if mol is None:
                continue
            dtags = set(_odor_tag(odor)) if isinstance(odor, str) else set()
            rows.append((smi, nm if isinstance(nm, str) else "",
                         Chem.MolToInchiKey(mol).split("-")[0], _fpvec(mol), dtags))
        if not rows:
            return
        X = np.vstack([r[3] for r in rows])
        tagsets = [set(r[4]) for r in rows]
        for name, clf in P._AROMA_MODELS.items():                 # model-confident aroma
            for i in np.where(clf.predict_proba(X)[:, 1] >= 0.5)[0]:
                tagsets[i].add(name)
        # Taste tags are NAMESPACED ("taste:sweet") for the same reason mouthfeel is: `sweet` is
        # both a taste head and an aroma head, and a bare tag conflated "tastes sweet" with
        # "smells sweet". `bitter` had additionally leaked into the aroma-notes picker.
        # EVERY trained taste head, not just sweet/bitter/umami — sour, salty and tasteless are
        # equally real targets (a neutral, tasteless carrier is a genuine formulation ask).
        for t in sorted(P._CLASSIFIERS):                          # taste
            clf = P._CLASSIFIERS.get(t)
            if clf is not None:
                for i in np.where(clf.predict_proba(X)[:, 1] >= 0.5)[0]:
                    tagsets[i].add(f"taste:{t}")
        # Mouthfeel tags are NAMESPACED ("mouthfeel:pungent"), matching how the profile index keys
        # its dims. Without this, picking `pungent` under Mouthfeel returned sharp-SMELLING
        # molecules (acetic acid, ammonia, CO2) from the far larger aroma:pungent set instead of
        # the TRPV1 burn agents — the two modalities share a name but not a meaning.
        for name, clf in P._MOUTHFEEL_MODELS.items():             # mouthfeel / chemesthesis
            for i in np.where(clf.predict_proba(X)[:, 1] >= 0.5)[0]:
                tagsets[i].add(f"mouthfeel:{name}")
        cnt, pool = Counter(), []
        for (smi, nm, skel, _, _), tags in zip(rows, tagsets):
            if not tags:
                continue
            pool.append({"smiles": smi, "name": nm, "tags": tags, "gras": skel in P._GRAS})
            cnt.update(tags)
        # fold the curated character-impact molecules (supplement + flavors) into the index so
        # their descriptors (coconut, nutty, vanilla, cinnamon...) are searchable + offerable even
        # where the industrial-skewed odor corpus is thin on them
        # aroma/flavor carriers keep their bare note; mouthfeel carriers carry the namespaced tag
        folds = [(n, c, n) for n, c in _NOTE_CARRIERS.items()]
        folds += [(n, c, f"mouthfeel:{n}") for n, c in _MOUTHFEEL_CARRIERS.items()]
        for _note, carriers, tag in folds:
            for csmi, cnm in carriers:
                cm = Chem.MolFromSmiles(csmi)
                if cm is None:
                    continue
                pool.append({"smiles": csmi, "name": cnm or "", "tags": {tag},
                             "gras": Chem.MolToInchiKey(cm).split("-")[0] in P._GRAS})
                cnt.update([tag])
        _DESIGN[:] = pool
        # Offer EVERY trained aroma head as a selectable note (even aroma-only ones with few
        # food-safe carriers), plus any design note with >=5 carriers. Namespaced mouthfeel tags
        # are excluded — they're a separate modality with their own pick-list below.
        _DESIGN_DESCS[:] = sorted(
            {d for d, n in cnt.items() if n >= 5 and ":" not in d} | set(P._AROMA_MODELS))
        # Mouthfeel and taste terms stay namespaced ("mouthfeel:cooling", "taste:sweet") so picking
        # `cooling`/`sweet` there matches the SENSATION / the TASTE, not the like-named odour note.
        # The UI shows the bare label. Only tastes with carriers in the pool are offered.
        _DESIGN_MOUTHFEEL[:] = [f"mouthfeel:{m}" for m in sorted(P._MOUTHFEEL_MODELS)]
        # EVERY trained taste head is offered, exactly like the aroma heads above — a head with few
        # confident carriers in this pool (sour, salty) must still be selectable, or the picker
        # silently hides a dimension the model can actually read.
        _DESIGN_TASTE[:] = [f"taste:{t}" for t in sorted(P._CLASSIFIERS)]
        _build_term_aliases()   # needs _FLAVORS + the finished _DESIGN_DESCS


def _fpvec(mol):
    """1-D model feature row for the taste/aroma heads (fingerprint + physicochemical block),
    matching how they were trained (predict._feat returns a (1, N) row; we want the flat vector)."""
    return P._feat(mol)[0]


@app.get("/api/design_options")
def api_design_options():
    """The descriptors the designer can actually match (enough molecules in the corpus)."""
    return {"descriptors": _DESIGN_DESCS}


@app.get("/api/design")
def api_design(descriptors: str = "", gras: int = 0, offset: int = 0, limit: int = 20):
    """Reverse search: given desired descriptors (+ optional food-safe filter), rank the
    best-matching molecules — with GRAS status and drop-in substitutes. Paginated via offset."""
    want = [d.strip().lower() for d in descriptors.split(",") if d.strip()]
    if not want or not _DESIGN:
        return {"items": [], "requested": want, "total_matches": 0, "offset": offset, "limit": limit}
    scored = []
    for m in _DESIGN:
        if gras and not m["gras"]:
            continue
        matched = [d for d in want if d in m["tags"]]
        if matched:
            scored.append((len(matched), m, matched))
    scored.sort(key=lambda x: (-x[0], not x[1]["gras"], x[1]["name"] == ""))
    items = []
    for n, m, matched in scored[offset:offset + limit]:  # substitutes for the visible page only
        subs = []
        for s in P.substitute(m["smiles"], k=6).get("neighbors", []):
            sm = Chem.MolFromSmiles(s["smiles"])
            if sm is not None and Chem.MolToInchiKey(sm).split("-")[0] in P._GRAS:
                subs.append({"smiles": s["smiles"], "name": _table_name(s["smiles"]) or "",
                             "similarity": s["similarity"]})
            if len(subs) >= 3:
                break
        items.append({"smiles": m["smiles"], "name": m["name"], "gras": m["gras"],
                      "matched": matched, "n_matched": n, "svg": _svg(m["smiles"], 108, 78),
                      "other": sorted(t for t in m["tags"] if t not in matched)[:5], "subs": subs})
    return {"items": items, "requested": want, "total_matches": len(scored),
            "offset": offset, "limit": limit}


# --- Flavor library: curated flavor -> character-impact molecule(s) ------------
# A small, hand-curated map from a familiar flavor (banana, saffron, smoke, pawpaw…) to the
# molecule(s) most responsible for it — "character-impact compounds" — drawn from public,
# common flavor-chemistry knowledge (no proprietary GC-MS profiles). It's the front door to the
# designer: pick a flavor you know, see the molecule that makes it, whether it's food-safe, and
# food-safe drop-ins. flavors.csv is a committed curated input (not a crawl artifact).
_FLAVORS = {}         # flavor -> [{molecule, smiles, category}]
_FLAVOR_CATS = []     # [{category, flavors:[...]}] for the picker UI


def _load_flavors(path="flavors.csv"):
    with contextlib.suppress(Exception):  # no csv; library just stays empty
        import csv
        from collections import OrderedDict
        p = Path(path)
        if not p.exists():
            return
        by_flavor, by_cat = OrderedDict(), OrderedDict()
        with p.open() as f:
            for row in csv.DictReader(f):
                fl, cat = row["flavor"], row.get("category", "other")
                by_flavor.setdefault(fl, []).append(
                    {"molecule": row["molecule"], "smiles": row["smiles"], "category": cat})
                by_cat.setdefault(cat, [])
                if fl not in by_cat[cat]:
                    by_cat[cat].append(fl)
        _FLAVORS.clear()
        _FLAVORS.update(by_flavor)
        _FLAVOR_CATS[:] = [{"category": c, "flavors": fs} for c, fs in by_cat.items()]


_load_flavors()


def _build_flavor_by_skel():
    """InChIKey-skeleton -> [everyday flavor names it's a character molecule of], for the read tags."""
    out = {}
    for flavor, entries in _FLAVORS.items():
        for e in entries:
            m = Chem.MolFromSmiles(e["smiles"])
            if m is not None:
                out.setdefault(Chem.MolToInchiKey(m).split("-")[0], [])
                if flavor not in out[Chem.MolToInchiKey(m).split("-")[0]]:
                    out[Chem.MolToInchiKey(m).split("-")[0]].append(flavor)
    return out


_FLAVOR_BY_SKEL = _build_flavor_by_skel()


def _flavor_card(smi, molname):
    """One character molecule as a display card: structure, name, GRAS status, its documented/
    predicted taste+aroma tags, and food-safe drop-in substitutes (same logic as the designer)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    skel = Chem.MolToInchiKey(mol).split("-")[0]
    tags = sorted(_design_tags(smi))
    subs = []
    for s in P.substitute(smi, k=8).get("neighbors", []):
        sm = Chem.MolFromSmiles(s["smiles"])
        if sm is not None and Chem.MolToInchiKey(sm).split("-")[0] in P._GRAS:
            subs.append({"smiles": s["smiles"], "name": _table_name(s["smiles"]) or "",
                         "similarity": s["similarity"]})
        if len(subs) >= 3:
            break
    return {"smiles": smi, "molecule": molname, "name": _table_name(smi) or molname,
            "gras": skel in P._GRAS, "tags": tags[:6], "svg": _svg(smi, 108, 78), "subs": subs}


def _design_tags(smi):
    """The tag set the designer index holds for this molecule (documented odor + model-confident
    aroma + taste), computed on the fly for molecules that aren't in the odor corpus."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return set()
    skel = Chem.MolToInchiKey(mol).split("-")[0]
    for m in _DESIGN:                                  # reuse the precomputed index when present
        dm = Chem.MolFromSmiles(m["smiles"])
        if dm is not None and Chem.MolToInchiKey(dm).split("-")[0] == skel:
            return set(m["tags"])
    tags = set()                                        # else predict directly
    with contextlib.suppress(Exception):
        X = _fpvec(mol).reshape(1, -1)
        for name, clf in P._AROMA_MODELS.items():
            if clf.predict_proba(X)[0, 1] >= 0.5:
                tags.add(name)
        for t in ("sweet", "bitter", "umami"):
            clf = P._CLASSIFIERS.get(t)
            if clf is not None and clf.predict_proba(X)[0, 1] >= 0.5:
                tags.add(t)
    return tags


@app.get("/api/flavors")
def api_flavors():
    """The curated flavor picker: flavors grouped by category (fruit, spice, floral, savory…)."""
    return {"categories": _FLAVOR_CATS, "count": len(_FLAVORS)}


@app.get("/api/flavor")
def api_flavor(name: str = ""):
    """One flavor -> its character-impact molecule(s), each with GRAS status, taste/aroma tags,
    and food-safe drop-in substitutes. The bridge from 'I want banana' to the actual chemistry."""
    entries = _FLAVORS.get(name.strip().lower()) or _FLAVORS.get(name.strip())
    if not entries:
        return {"flavor": name, "molecules": []}
    cards = [c for c in (_flavor_card(e["smiles"], e["molecule"]) for e in entries) if c]
    return {"flavor": name, "category": entries[0]["category"], "molecules": cards}


# --- Flavor Studio: ONE unified search over flavors AND notes -------------------
# A flavor IS a set of notes, so the Studio offers a single pick-list mixing curated flavors
# (banana, saffron…) with model/documented descriptors (citrus, floral…). Pick any combination;
# a molecule is ranked by how many of your picks it carries — the character molecule of a flavor,
# a molecule that has a note, or (best) one that satisfies several at once.
def _gras_subs(smi, k=3):
    subs = []
    for s in P.substitute(smi, k=6).get("neighbors", []):
        sm = Chem.MolFromSmiles(s["smiles"])
        if sm is not None and Chem.MolToInchiKey(sm).split("-")[0] in P._GRAS:
            subs.append({"smiles": s["smiles"], "name": _table_name(s["smiles"]) or "",
                         "similarity": s["similarity"]})
        if len(subs) >= k:
            break
    return subs


@app.get("/api/studio_terms")
def api_studio_terms():
    """The unified pick-list, one entry per modality the studios can target: curated flavors
    (grouped by category), aroma-note descriptors, mouthfeel sensations, and basic tastes."""
    return {"flavors": _FLAVOR_CATS, "notes": _DESIGN_DESCS,
            "mouthfeel": _DESIGN_MOUTHFEEL, "taste": _DESIGN_TASTE}


@app.get("/api/nl")
def api_nl(q: str = ""):
    """Natural-language intent -> Studio picks. A lightweight, offline, commercial-clean parser:
    it scans free text ('make a food-safe cherry flavoring with fruity notes') for the flavor and
    note words the Studio already knows, plus a food-safe (GRAS) intent. No model, no GPU — a
    keyword/whole-word match. (A real on-prem LLM emitting structured intent is the future
    upgrade; see the roadmap — this covers the 'type what you want' story today.)"""
    import re
    ql = q.lower()
    known = set(_FLAVORS.keys()) | set(_DESIGN_DESCS)
    # match longest terms first so 'bubble gum' / 'green pea' win over 'gum' / 'green'
    hits = []
    for t in sorted(known, key=len, reverse=True):
        # skip a term wholly inside an already-matched longer term's span
        if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", ql) \
                and not any(t != h and t in h for h in hits):
            hits.append(t)
    # order them as they appear in the query, dedup
    seen, terms = set(), []
    for t in sorted(hits, key=lambda x: ql.find(x)):
        if t not in seen:
            seen.add(t)
            terms.append(t)
    gras = bool(re.search(r"food[\s-]?safe|gras|edible|safe to eat", ql))
    return {"query": q, "terms": terms, "gras": gras,
            "understood": bool(terms),
            "note": ("Matched to what the Studio knows; a full natural-language model is on the "
                     "roadmap." if terms else "No known flavors or notes recognized — try words "
                     "like 'cherry', 'fruity', 'citrus', 'food-safe'.")}


# --- Master enrichment table: one rich row per molecule (browse the whole universe) ---------
# Columns each carry a "why it matters" note so the table is legible to a non-chemist. Built by
# build_enrichment.py (master_enrichment.parquet); gets richer as the PubChem crawl fills in
# names / melting / boiling points.
_ENRICH = []
ENRICH_COLUMNS = [
    {"key": "svg", "label": "Structure", "why": "2D depiction drawn straight from the structure (RDKit).", "sort": False},
    {"key": "name", "label": "Common name", "why": "Everyday name (PubChem Title, public domain) where one exists."},
    {"key": "iupac", "label": "IUPAC name", "why": "Systematic IUPAC name (public-domain PubChem) — the unambiguous identity."},
    {"key": "smiles", "label": "SMILES", "why": "The machine-readable structure string the models actually read."},
    {"key": "formula", "label": "Formula", "why": "Hill-system molecular formula (e.g. C9H16O2) — a compact 4th identifier."},
    {"key": "taste", "label": "Taste", "why": "Documented taste where known, else the model's call — what it tastes like."},
    {"key": "aroma_top", "label": "Aroma", "why": "The strongest predicted odor descriptor — the note it most reads as."},
    {"key": "mw", "label": "MW", "why": "Molecular weight (Da) — size. Heavier molecules are generally less volatile, so aroma fades."},
    {"key": "logp", "label": "logP", "why": "Lipophilicity (oil↔water). Sets solubility, which carrier a flavor needs, and how it partitions in a product."},
    {"key": "tpsa", "label": "TPSA", "why": "Polar surface area (Å²) — polarity / H-bonding. High TPSA ⇒ more water-soluble, less volatile."},
    {"key": "hbd", "label": "HBD", "why": "H-bond donors — drive water solubility and lower volatility."},
    {"key": "hba", "label": "HBA", "why": "H-bond acceptors — same story: solubility and volatility."},
    {"key": "rot_bonds", "label": "RotB", "why": "Rotatable bonds — molecular flexibility (rigidity often tracks with a sharper odor)."},
    {"key": "melting_point_c", "label": "MP °C", "why": "Melting point (measured) — solid vs liquid at room temperature."},
    {"key": "boiling_point_c", "label": "BP °C", "why": "Boiling point (measured) — a direct handle on volatility, hence aroma strength."},
    {"key": "gras", "label": "Food-listed", "why": "Listed in a food-use reference (FDA Substances-Added-to-Food / EU flavourings) — a listing flag, not a GRAS or safety clearance."},
]


def _load_enrichment():
    """Rows from master_enrichment.parquet with taste collapsed to a display string."""
    try:
        import pandas as pd
        df = pd.read_parquet(P.artifact("master_enrichment.parquet"))
    except Exception:  # noqa: BLE001 — not built yet
        return []
    def _s(v):  # NaN (a truthy float) -> "" ; keep real strings
        return v if isinstance(v, str) else ""

    rows = []
    for _, r in df.iterrows():
        taste = _s(r.get("taste_documented")) or _s(r.get("taste_predicted"))
        skel = r.get("inchikey_skel")
        iupac = (_NAME_TABLE.get(skel) or (None, None))[1] if isinstance(skel, str) else None
        rows.append({
            "smiles": r["smiles"], "name": _s(r.get("name")), "iupac": iupac or "",
            "formula": _formula(r["smiles"]) or "",
            "taste": taste, "aroma_top": _s(r.get("aroma_top")),
            "mw": _num(r.get("mw")), "logp": _num(r.get("logp")), "tpsa": _num(r.get("tpsa")),
            "hbd": _num(r.get("hbd")), "hba": _num(r.get("hba")), "rot_bonds": _num(r.get("rot_bonds")),
            "melting_point_c": _num(r.get("melting_point_c")), "boiling_point_c": _num(r.get("boiling_point_c")),
            "gras": bool(r.get("gras")), "is_isomer": bool(r.get("is_isomer")),
        })
    return rows


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else (int(f) if f == int(f) else round(f, 2))
    except (TypeError, ValueError):
        return None


@app.get("/api/enrichment_meta")
def api_enrichment_meta():
    """Column definitions (label + why-it-matters) for the enrichment table."""
    return {"columns": ENRICH_COLUMNS, "count": len(_ENRICH)}


@app.get("/api/enrichment")
def api_enrichment(q: str = "", sort: str = "name", desc: int = 0, offset: int = 0, limit: int = 50):
    """Sortable, searchable, paginated master enrichment table — the whole molecule universe."""
    global _ENRICH
    if not _ENRICH:
        _ENRICH = _load_enrichment()
    rows = _ENRICH
    ql = q.strip().lower()
    if ql:
        rows = [r for r in rows if ql in (r["name"] or "").lower() or ql in r["smiles"].lower()
                or ql in (r["taste"] or "").lower() or ql in (r["aroma_top"] or "").lower()]
    sortable = {c["key"] for c in ENRICH_COLUMNS if c.get("sort", True)}
    if sort not in sortable:
        sort = "name"

    def sortkey(r):
        v = r.get(sort)
        if isinstance(v, bool):
            return (0, "", 1 if v else 0)
        if isinstance(v, (int, float)):
            return (0, "", v)
        return (1 if not v else 0, str(v).lower(), 0)  # blanks last

    rows = sorted(rows, key=sortkey, reverse=bool(desc))
    total = len(rows)
    page = [dict(r, svg=_svg_cell(r["smiles"])) for r in rows[offset:offset + limit]]
    return {"rows": page, "total": total, "offset": offset, "limit": limit,
            "columns": ENRICH_COLUMNS}


@lru_cache(maxsize=4096)
def _svg_cell(smi):
    """A compact 2D depiction for one enrichment-table row (cached; rendered per visible page)."""
    return _svg(smi, 104, 62)


@app.get("/api/map_members")
@lru_cache(maxsize=512)
def api_map_members(term: str = "", threshold: float | None = None):
    """Every molecule whose head `term` fires >= threshold — not just the ones where it happens to
    be the DOMINANT label.

    The map colours each molecule by a single winning label, so a head can read "0" in the legend
    while still being trained on plenty of molecules — they're simply displayed under a rarer
    co-occurring label. Highlighting from the legend used to match that winner-take-all label, so
    those heads lit up nothing and looked broken. This returns true membership from the profile
    index (which carries every head score for every molecule), so no head can ever look empty.

    `term` may be bare ("clarysage") or dimension-qualified ("aroma:clarysage", "mouthfeel:cooling")
    — bare names resolve to aroma first, matching how the map legend labels them.
    """
    t = (term or "").strip().lower()
    if not t:
        return {"term": term, "smiles": [], "n": 0}
    P._ensure_sub_index()
    smis, profiles, dims = P._SUB_INDEX[1], P._SUB_INDEX[4], P._SUB_INDEX[5]
    if profiles is None or not smis:
        return {"term": term, "smiles": [], "n": 0, "note": "profile index not built"}
    dims = [str(d) for d in dims]
    col = None
    for cand in ([t] if ":" in t else [f"aroma:{t}", f"taste:{t}", f"mouthfeel:{t}"]):
        if cand in dims:
            col = dims.index(cand)
            break
    if col is None:
        return {"term": term, "smiles": [], "n": 0, "note": "no such head"}
    # Default to the head's OWN calibrated threshold, not a flat 0.5. A head that fires at 0.24
    # would otherwise have most of its molecules hidden here while the modal happily calls them
    # matches — the two surfaces disagreeing about the same head.
    thr = threshold if threshold is not None else _member_threshold(dims[col])
    hits = [smis[i] for i in range(len(smis)) if float(profiles[i][col]) >= thr]
    return {"term": term, "dim": dims[col], "threshold": thr, "n": len(hits), "smiles": hits}


def _member_threshold(dim):
    """The calibrated firing threshold for a dimension-qualified head name ("aroma:pine")."""
    kind, _, name = dim.partition(":")
    meta = {"aroma": P._AROMA_META, "taste": P._TASTE_META,
            "mouthfeel": P._MOUTHFEEL_META}.get(kind, {})
    return P._head_threshold(meta, name)


@app.get("/api/map_counts")
@lru_cache(maxsize=4)
def api_map_counts():
    """How many molecules each head actually FIRES on, for every head at once.

    The map legend used to show the count of molecules where a head wins the winner-take-all
    colour, which is a different question and produced the confusing "(0)" on heads that are
    trained on plenty of molecules. `raspberry` reads 0 there and fires on 19. This gives the
    legend the number people are actually asking for; the dominant label stays a colouring device.
    """
    P._ensure_sub_index()
    smis, profiles, dims = P._SUB_INDEX[1], P._SUB_INDEX[4], P._SUB_INDEX[5]
    if profiles is None or not smis:
        return {"counts": {}, "note": "profile index not built"}
    out = {}
    for j, d in enumerate(str(x) for x in dims):
        thr = _member_threshold(d)
        out[d] = int(sum(1 for i in range(len(smis)) if float(profiles[i][j]) >= thr))
    return {"counts": out}


@lru_cache(maxsize=8192)
def _tox_flags(smiles):
    """Tox21 assays this molecule is predicted active in (>=0.5), as a tuple. Caution-only: assay
    activity is INDICATIVE and warrants review, it is never a toxicity determination (see TOX.md).
    Cached because the studio calls it once per result row."""
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return ()
    scr = P.predict_tox(mol)
    return tuple(a["assay"] for a in scr.get("assays", []) if (a.get("probability") or 0) >= 0.5)


@app.get("/api/studio")
def api_studio(terms: str = "", gras: int = 0, no_tox: int = 0, offset: int = 0, limit: int = 20):
    """Unified search: given any mix of flavors and notes, rank the molecules that carry them.
    Each molecule scores by how many distinct picked terms it matches (a flavor's character
    molecule matches that flavor; a molecule with a note matches that note)."""
    want = [t.strip().lower() for t in terms.split(",") if t.strip()]
    if not want:
        return {"items": [], "requested": want, "total_matches": 0, "offset": offset, "limit": limit}
    flavor_terms = [t for t in want if t in _FLAVORS]
    note_terms = [t for t in want if t not in _FLAVORS]
    # cross-list dual-purpose terms: a flavor and its like-named note are the same concept spelled
    # differently ("black pepper"/"blackpepper", "butter"/"buttery", "cilantro"/"coriander"), so
    # picking either should surface the other's molecules too.
    note_terms += [a for t in want for a in _TERM_ALIASES.get(t, ()) if a not in note_terms]
    cand = {}  # skeleton -> {smiles, name, gras, matched:set}

    def add(smi, name, is_gras, term):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return
        skel = Chem.MolToInchiKey(mol).split("-")[0]
        rec = cand.get(skel)
        if rec is None:
            rec = cand[skel] = {"smiles": smi, "name": name or "", "gras": is_gras, "matched": set(),
                                "tags": set()}
        rec["matched"].add(term)

    for ft in flavor_terms:                               # flavor -> its character molecule(s)
        for e in _FLAVORS.get(ft, []):
            mol = Chem.MolFromSmiles(e["smiles"])
            g = mol is not None and Chem.MolToInchiKey(mol).split("-")[0] in P._GRAS
            add(e["smiles"], _table_name(e["smiles"]) or e["molecule"], g, ft)
    if note_terms:                                        # notes -> molecules carrying them
        for m in _DESIGN:
            hit = [nt for nt in note_terms if nt in m["tags"]]
            for nt in hit:
                add(m["smiles"], m["name"], m["gras"], nt)
                cand[Chem.MolToInchiKey(Chem.MolFromSmiles(m["smiles"])).split("-")[0]]["tags"] = m["tags"]
    scored = [r for r in cand.values() if not (gras and not r["gras"])]
    # The 12 Tox21 heads aren't design TARGETS — nobody formulates *for* assay activity — but they
    # shouldn't be invisible either, so they surface per result and can filter the list. Still
    # caution-only: an assay flag means "review this", never "this is toxic".
    if no_tox:
        scored = [r for r in scored if not _tox_flags(r["smiles"])]
    scored.sort(key=lambda r: (-len(r["matched"]), not r["gras"], r["name"] == ""))
    items = []
    for r in scored[offset:offset + limit]:
        matched = sorted(r["matched"])
        items.append({"smiles": r["smiles"], "name": r["name"], "gras": r["gras"],
                      "matched": matched, "n_matched": len(matched),
                      "svg": _svg(r["smiles"], 108, 78),
                      "other": sorted(t for t in r["tags"] if t not in r["matched"])[:5],
                      "tox_flags": _tox_flags(r["smiles"]),
                      "subs": _gras_subs(r["smiles"])})
    return {"items": items, "requested": want, "flavor_terms": flavor_terms, "note_terms": note_terms,
            "total_matches": len(scored), "offset": offset, "limit": limit}


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
        df = pd.read_parquet(P.artifact("odor_notes.parquet"))

        def col(name):
            return df[name] if name in df.columns else [None] * len(df)

        out = {}
        for ik, name, odor, osrc, thr, tsrc in zip(
                df["inchikey"], col("name"), col("odor"), col("odor_source"),
                col("odor_threshold_ppm"), col("odor_threshold_source")):
            if not isinstance(ik, str):
                continue
            rec = {}
            if isinstance(name, str):
                rec["name"] = name
            if isinstance(odor, str):
                rec["odor"] = odor
                rec["odor_source"] = osrc if isinstance(osrc, str) else None
            if isinstance(thr, (int, float)) and not math.isnan(thr):  # numeric and not NaN
                rec["threshold_ppm"] = float(thr)
                rec["threshold_source"] = tsrc if isinstance(tsrc, str) else None
            if rec:
                out[ik.split("-")[0]] = rec
        return out
    except Exception:  # noqa: BLE001 — no table / no pandas
        return {}


_ODOR_TABLE = _load_odor_table()


def _load_documented_full():
    """{full InChIKey -> {odor?, taste?}} from odor_notes / taste_notes — keyed by the FULL key
    (stereo included) so the stereoisomer explorer can surface enantiomer-specific documented
    sensory data (e.g. R- vs S-carvone) that the skeleton-keyed tables collapse together."""
    out = {}
    with contextlib.suppress(Exception):
        import pandas as pd
        od = pd.read_parquet(P.artifact("odor_notes.parquet"))
        for ik, odor in zip(od["inchikey"], od["odor"]):
            if isinstance(ik, str) and isinstance(odor, str) and odor.strip():
                out.setdefault(ik, {})["odor"] = odor.strip().split("\n")[0][:160]
    with contextlib.suppress(Exception):
        import pandas as pd
        tn = pd.read_parquet(P.artifact("taste_notes.parquet"))
        for ik, taste in zip(tn["inchikey"], tn["taste"]):
            if isinstance(ik, str) and isinstance(taste, str) and taste.strip():
                out.setdefault(ik, {})["taste"] = taste.strip().split("\n")[0][:160]
    return out


_DOCUMENTED_FULL = _load_documented_full()


def _documented_by_full(inchikey):
    """Isomer-specific documented odor/taste for an exact InChIKey ({} if none on record)."""
    return _DOCUMENTED_FULL.get(inchikey, {})


# start the landing-page precompute now that every table it reads (_NAME_TABLE, _ODOR_TABLE,
# the models) is defined — starting it earlier would race those globals into NameErrors
def _precompute_all():
    P.MODELS_READY.wait()  # both precomputes run head inference; wait for the load to finish
    _precompute_top_lists()
    _precompute_design()


threading.Thread(target=_precompute_all, daemon=True).start()


@app.post("/api/aroma")
def api_aroma(q: Query):
    """Aroma read: (1) real cited DOCUMENTED odor + threshold (public-domain HSDB/Haz-Map) when
    the molecule is in the corpus, and (2) PREDICTED descriptors from RandomForest heads trained
    on that corpus — which work for ANY molecule, including ones with no documented entry. The
    predicted heads are presence/absence (not intensity); each carries its CV-AUROC."""
    smi = _resolve(q.smiles)
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return {"available": False}
    out = {"available": False}
    rec = _ODOR_TABLE.get(Chem.MolToInchiKey(mol).split("-")[0])
    if rec:
        if rec.get("odor"):
            notes = [s.strip() for s in rec["odor"].split("\n") if s.strip()]
            concise = [n for n in notes if len(n) <= 90] or notes  # lead with punchy descriptors
            out["documented"] = {"notes": concise[:4], "source": rec.get("odor_source")}
            out["available"] = True
        if rec.get("threshold_ppm") is not None:
            out["threshold"] = {"ppm": rec["threshold_ppm"], "source": rec.get("threshold_source")}
            out["available"] = True
    pa = P.predict_aroma(smi)
    if pa.get("available") and pa.get("descriptors"):
        out["predicted"] = {"descriptors": pa["descriptors"], "note": pa.get("note"),
                            "any_confident": pa.get("any_confident", True)}
        out["available"] = True
    return out


@app.get("/", response_class=HTMLResponse)
def home():
    return Path("workbench.html").read_text()
