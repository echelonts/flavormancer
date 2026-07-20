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

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rdkit import Chem

import predict as P  # the unified flavor read + substitution search

app = FastAPI(title="Flavor Workbench (demo)")


def _load_name2smiles():
    """Local name -> SMILES index (instant, offline) so library/demo ingredients resolve without a
    PubChem round-trip. Built from master_enrichment.parquet (~8k named molecules) + the suggest
    CSV. Only genuinely-unknown names fall through to live PubChem in _resolve()."""
    idx = {}
    try:
        import pandas as pd
        df = pd.read_parquet("master_enrichment.parquet")
        for nm, smi in zip(df["name"], df["smiles"]):
            if isinstance(nm, str) and isinstance(smi, str) and nm.strip() and smi.strip():
                idx.setdefault(nm.strip().lower(), smi)
    except Exception:  # noqa: BLE001 — table absent / no pandas; live lookup still covers it
        pass
    try:
        import csv
        with open("flavor_volatiles.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("name") and r.get("smiles"):
                    idx.setdefault(r["name"].strip().lower(), r["smiles"])
    except Exception:  # noqa: BLE001 — no suggest file; fine
        pass
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


def _merge_iupac_backfill(table):
    """Fold in IUPAC names that build_iupac_backfill.py recovered from PubChem for molecules
    the main properties crawl missed (skeleton -> keep any common name, add the IUPAC)."""
    try:
        import pandas as pd
        bf = pd.read_parquet("iupac_backfill.parquet")
    except Exception:  # noqa: BLE001 — backfill not built; nothing to merge
        return table
    for skel, u in zip(bf["inchikey_skel"], bf["iupac_name"]):
        if isinstance(skel, str) and isinstance(u, str) and u:
            common = table.get(skel, (None, None))[0]
            table[skel] = (common, u)
    return table


_NAME_TABLE = _merge_iupac_backfill(_load_name_table())


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
    k: int = 8


@app.post("/api/predict")
def api_predict(q: Query):
    smi = _resolve(q.smiles)
    if not smi:
        return {"error": f"Couldn't resolve '{q.smiles}' to a structure. "
                         f"Enter a valid SMILES or a recognized compound name."}
    out = P.predict(smi, include_aroma=False)
    out["flavor_tags"] = _read_tags(smi, out)
    out["references"] = _references(smi)
    return out


def _load_spectra():
    """inchikey-skeleton -> [available spectra types] from spectra.parquet (build_spectra.py):
    public-domain PubChem availability metadata. Empty until the crawl has run."""
    try:
        import pandas as pd
        df = pd.read_parquet("spectra.parquet")
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
    try:
        inchi = Chem.MolToInchi(mol)
        if inchi:
            refs.append({"label": "NIST WebBook", "note": "IR / MS spectra, GC retention index",
                         "url": "https://webbook.nist.gov/cgi/cbook.cgi?InChI="
                                + urllib.parse.quote(inchi) + "&Units=SI"})
    except Exception:  # noqa: BLE001 — InChI generation can fail on odd valences
        pass
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
        try:
            from build_aroma_dataset import tag as _odor_tag
            found = sorted(_odor_tag(rec["odor"]))[:k]
            if found:
                return [{"odor": t, "source": "found"} for t in found]
        except Exception:  # noqa: BLE001 — vocab module missing; fall through to predicted
            pass
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
            try:
                from build_aroma_dataset import tag as _odor_tag
                found = sorted(_odor_tag(rec["odor"]))[:k]
                if found:
                    return [{"odor": t, "source": "found"} for t in found]
            except Exception:  # noqa: BLE001 — vocab module missing; fall through to predicted
                pass
    return [{"odor": a, "source": "predicted"} for a in (precomputed or [])][:k]


@app.post("/api/neighbors")
def api_neighbors(q: Query):
    smi = _resolve(q.smiles)
    if not smi:
        return {"neighbors": []}
    res = P.substitute(smi, k=q.k)
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
        except Exception:  # noqa: BLE001
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
    pa = P.predict_aroma(smi)
    aroma_cells = sorted(((d["odor"], d["score"]) for d in pa.get("descriptors", [])),
                         key=lambda kv: -kv[1])

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
    try:
        emblem = Image.open("static/logo.png").convert("RGBA").resize((58, 58))
        img.paste(emblem, (40, 30), emblem)
    except Exception:  # noqa: BLE001
        pass
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
    dr.text((x, 280), "READS AS", font=f_lab, fill=muted)
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
        ptxt = f"{int(round(v * 100))}%"
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

    # AROMA MODEL — all 24 heads in a 6-wide grid
    dr.text((GX0, aroma_label_y), f"AROMA MODEL · all {len(aroma_cells)} heads", font=f_lab, fill=muted)
    for i, (a, v) in enumerate(aroma_cells):
        cell(i % COLS, aroma_grid_y + (i // COLS) * ROW_H, a, v, teal)

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


class MixtureQuery(BaseModel):
    ingredients: list[str]
    processes: list[str] = []


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
                    cmol = Chem.MolFromSmiles(mt["smiles"])     # cheap GRAS lookup — no full predict() pipeline
                    is_gras = cmol is not None and P._gras_status(cmol).startswith("in GRAS")
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
    for coconut, maltol for caramel) over a generic palette match, which can surface poor carriers."""
    import csv
    m = {}
    for path in ("flavors.csv", "aroma_supplement.csv"):
        try:
            with open(path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    note = (r.get("flavor") or "").strip().lower()
                    smi = (r.get("smiles") or "").strip()
                    nm = (r.get("molecule") or "").strip()
                    if note and smi:
                        m.setdefault(note, []).append((smi, nm or note))
        except Exception:  # noqa: BLE001 — missing file / bad rows; just skip
            pass
    return m


_NOTE_CARRIERS = _load_note_carriers()


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
            if cmol is not None and P._gras_status(cmol).startswith("in GRAS"):
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
    # Build the substitution index FIRST (the ~8k-row aroma batch) so the first neighbor search
    # never pays the one-time build; the lock in substitute() makes a concurrent request wait.
    try:
        P.substitute("CCO")
    except Exception:  # noqa: BLE001 — best-effort
        pass
    for n in _FORMULATION_WARM:
        try:
            smi = _resolve(n)
            m = Chem.MolFromSmiles(smi) if smi else None
            if m is not None:
                P.predict_aroma(Chem.MolToSmiles(m))
        except Exception:  # noqa: BLE001 — best-effort warmup; a miss just means a cold first hit
            pass


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
    try:
        tm = pd.read_parquet("taste_master.parquet")
        for taste, clf in P._CLASSIFIERS.items():
            ranked = _rank(tm["smiles"], lambda X, c=clf: c.predict_proba(X)[:, 1])
            _TOP_LISTS[f"taste:{taste}"] = {"label": f"Top {taste}", "items": _named_top(ranked)}
        # salty is a rule/known-label taste (not a model head) — list the LABELED salty molecules
        if "salty" in tm.columns:
            salty = [(s, 1.0) for s in tm.loc[tm["salty"] == 1, "smiles"]]
            items = _named_top(salty)
            if items:
                _TOP_LISTS["taste:salty"] = {"label": "Known salty", "items": items}
    except Exception:  # noqa: BLE001 — no taste data; skip taste lists
        pass
    try:
        # aroma lists use DOCUMENTED odor (ground truth), not model ranking: the public corpus
        # skews industrial, so ranking by a head surfaces confident-but-odd picks (cyanide under
        # "almond"). Documented examples are real, recognizable, and honest ("documented citrus").
        from build_aroma_dataset import tag as _odor_tag
        od = pd.read_parquet("odor_notes.parquet")
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
    except Exception:  # noqa: BLE001 — no odor corpus / vocab; skip aroma lists
        pass
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
        df = pd.read_parquet("flavor_map.parquet")
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
        # raw physicochemical values (for the interpretable MW×logP×TPSA axes view — real units)
        mw = df["mw"].tolist() if "mw" in df.columns else [None] * len(smis)
        logp = df["logp"].tolist() if "logp" in df.columns else [None] * len(smis)
        tpsa = df["tpsa"].tolist() if "tpsa" in df.columns else [None] * len(smis)
        pts = []
        for i in range(len(smis)):
            p = {"label": labs[i], "aroma": aromas[i], "smiles": smis[i],
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


def _precompute_design():
    """Index the odor corpus by descriptor (documented tags + model-confident predictions +
    taste), with GRAS status — so the designer can rank food-safe molecules for a target flavor.
    Model inference is BATCHED (one vectorized call per head over all molecules) — per-molecule
    RandomForest calls over ~2.3k molecules would take minutes."""
    try:
        from collections import Counter

        import numpy as np
        import pandas as pd

        from build_aroma_dataset import tag as _odor_tag
        od = pd.read_parquet("odor_notes.parquet")
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
        for t in ("sweet", "bitter", "umami"):                    # taste
            clf = P._CLASSIFIERS.get(t)
            if clf is not None:
                for i in np.where(clf.predict_proba(X)[:, 1] >= 0.5)[0]:
                    tagsets[i].add(t)
        cnt, pool = Counter(), []
        for (smi, nm, skel, _, _), tags in zip(rows, tagsets):
            if not tags:
                continue
            pool.append({"smiles": smi, "name": nm, "tags": tags, "gras": skel in P._GRAS})
            cnt.update(tags)
        _DESIGN[:] = pool
        _DESIGN_DESCS[:] = sorted(d for d, n in cnt.items() if n >= 5)
    except Exception:  # noqa: BLE001 — no corpus/models; designer just stays empty
        pass


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
    try:
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
    except Exception:  # noqa: BLE001 — no csv; library just stays empty
        pass


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
    try:
        X = _fpvec(mol).reshape(1, -1)
        for name, clf in P._AROMA_MODELS.items():
            if clf.predict_proba(X)[0, 1] >= 0.5:
                tags.add(name)
        for t in ("sweet", "bitter", "umami"):
            clf = P._CLASSIFIERS.get(t)
            if clf is not None and clf.predict_proba(X)[0, 1] >= 0.5:
                tags.add(t)
    except Exception:  # noqa: BLE001
        pass
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
    """The unified pick-list: curated flavors (grouped by category) + matchable note descriptors."""
    return {"flavors": _FLAVOR_CATS, "notes": _DESIGN_DESCS}


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
        if re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", ql):
            # skip a term wholly inside an already-matched longer term's span
            if not any(t != h and t in h for h in hits):
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
    {"key": "gras", "label": "GRAS", "why": "On the FEMA/FDA food-safe (Generally Recognized As Safe) list — a flag, not a clearance."},
]


def _load_enrichment():
    """Rows from master_enrichment.parquet with taste collapsed to a display string."""
    try:
        import pandas as pd
        df = pd.read_parquet("master_enrichment.parquet")
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
        return None if f != f else (int(f) if f == int(f) else round(f, 2))
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


@app.get("/api/studio")
def api_studio(terms: str = "", gras: int = 0, offset: int = 0, limit: int = 20):
    """Unified search: given any mix of flavors and notes, rank the molecules that carry them.
    Each molecule scores by how many distinct picked terms it matches (a flavor's character
    molecule matches that flavor; a molecule with a note matches that note)."""
    want = [t.strip().lower() for t in terms.split(",") if t.strip()]
    if not want:
        return {"items": [], "requested": want, "total_matches": 0, "offset": offset, "limit": limit}
    flavor_terms = [t for t in want if t in _FLAVORS]
    note_terms = [t for t in want if t not in _FLAVORS]
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
    scored.sort(key=lambda r: (-len(r["matched"]), not r["gras"], r["name"] == ""))
    items = []
    for r in scored[offset:offset + limit]:
        matched = sorted(r["matched"])
        items.append({"smiles": r["smiles"], "name": r["name"], "gras": r["gras"],
                      "matched": matched, "n_matched": len(matched),
                      "svg": _svg(r["smiles"], 108, 78),
                      "other": sorted(t for t in r["tags"] if t not in r["matched"])[:5],
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
        df = pd.read_parquet("odor_notes.parquet")

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
            if isinstance(thr, (int, float)) and thr == thr:  # numeric and not NaN
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
    try:
        import pandas as pd
        od = pd.read_parquet("odor_notes.parquet")
        for ik, odor in zip(od["inchikey"], od["odor"]):
            if isinstance(ik, str) and isinstance(odor, str) and odor.strip():
                out.setdefault(ik, {})["odor"] = odor.strip().split("\n")[0][:160]
    except Exception:  # noqa: BLE001
        pass
    try:
        import pandas as pd
        tn = pd.read_parquet("taste_notes.parquet")
        for ik, taste in zip(tn["inchikey"], tn["taste"]):
            if isinstance(ik, str) and isinstance(taste, str) and taste.strip():
                out.setdefault(ik, {})["taste"] = taste.strip().split("\n")[0][:160]
    except Exception:  # noqa: BLE001
        pass
    return out


_DOCUMENTED_FULL = _load_documented_full()


def _documented_by_full(inchikey):
    """Isomer-specific documented odor/taste for an exact InChIKey ({} if none on record)."""
    return _DOCUMENTED_FULL.get(inchikey, {})


# start the landing-page precompute now that every table it reads (_NAME_TABLE, _ODOR_TABLE,
# the models) is defined — starting it earlier would race those globals into NameErrors
def _precompute_all():
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
