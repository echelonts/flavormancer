# Flavormancer MCP server

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
Flavormancer's **on-prem, trained taste/aroma models** as tools any MCP client can
call — Claude Desktop, an agent, or another application.

It is a thin, well-behaved adapter: it forwards to the local Flavormancer HTTP API,
which runs the RandomForest heads **on-prem** (no cloud calls in a read). Nothing
about the molecule or formulation leaves the box.

```
MCP client ──stdio──▶ flavormancer-mcp ──HTTP──▶ Flavormancer API ──▶ trained RF models (on-prem)
```

## Tools

Full coverage of Flavormancer's capabilities — 13 tools.

**Single molecule**

| Tool | What it does |
|------|--------------|
| **`read_flavor(molecule)`** | Taste (6 heads) + aroma (confident + all 24 scores), GRAS, structural alerts, applicability-domain flag. Name or SMILES. |
| **`read_full(molecule)`** | The complete read — everything above *plus* physicochemical properties, stability flags, chemesthesis (cooling/pungent/astringent), chirality, analytical (retention index), EU-allergen labeling, spectra/reference links, full safety block. |
| **`find_substitutes(molecule, k?)`** | The k structurally-nearest molecules (drop-in swaps), with similarity + known tastes. |
| **`list_stereoisomers(molecule)`** | Every stereoisomer with any documented isomer-specific odor/taste (e.g. R-carvone spearmint vs S-carvone caraway). |

**Formulation & mixtures**

| Tool | What it does |
|------|--------------|
| **`analyze_formulation(ingredients, target?, processes?)`** | Read a whole recipe *before you pour*: blended note-profile (weighted by odor impact), overpowering-component flag, target gap analysis with food-safe add/cut, hazard screen, data-gate notes. |
| **`screen_mixture(ingredients, processes?)`** | Documented combination hazards (e.g. benzoate + ascorbate → benzene), gated on process, plus per-ingredient reads and a palette match. |
| **`predict_reactions(ingredients, processes?)`** | Indicative reaction-template products (with each product's own taste + aroma). Template-based, not a claim the reaction proceeds. |

**Search, discovery & the map**

| Tool | What it does |
|------|--------------|
| **`find_molecules_by_notes(notes, food_safe?)`** | Molecules carrying a set of notes (`["citrus","fresh"]`), ranked, food-safe by default. |
| **`find_molecules_by_flavor(flavor)`** | The molecule(s) that MAKE a named flavor (`"banana"` → isoamyl acetate). |
| **`interpret_request(text)`** | Parse a free-text brief into structured Studio picks (flavor + notes + food-safe). |
| **`list_flavor_categories()`** | The browsable taste/aroma categories (key + count). |
| **`browse_category(category, limit?)`** | Model-ranked top molecules for a category key. |
| **`flavor_map(label?, limit?)`** | Flavor-space map: label distribution across all ~8k molecules, or the molecules carrying a given taste/aroma label. |

Every tool returns model **predictions only** — never a safety, GRAS, or regulatory determination. Heavy render blobs (SVG/PNG) are stripped so responses stay token-lean.

## Run it

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# point at your Flavormancer instance (default: http://127.0.0.1:8000)
export FLAVORMANCER_URL=http://127.0.0.1:8000
python server.py          # stdio transport
```

The Flavormancer API (`training/app.py`) must be running and reachable at `FLAVORMANCER_URL`.

## Add to Claude Desktop

Add to `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "flavormancer": {
      "command": "/absolute/path/to/mcp-server/.venv/bin/python",
      "args": ["/absolute/path/to/mcp-server/server.py"],
      "env": { "FLAVORMANCER_URL": "http://127.0.0.1:8000" }
    }
  }
}
```

Then ask Claude things like *"read the flavor of ethyl maltol,"* *"analyze this soda base:
limonene 250 ppm, citral 40, linalool 25 — I'm aiming for citrus and fresh,"* or *"screen
sodium benzoate with ascorbic acid for hazards."*

## Add to Claude Code (CLI)

Claude Code speaks MCP too — no Desktop required:

```sh
claude mcp add flavormancer \
  --env FLAVORMANCER_URL=http://127.0.0.1:8000 \
  -- /absolute/path/to/mcp-server/.venv/bin/python /absolute/path/to/mcp-server/server.py
```

Then `claude mcp list` shows it, and the tools are available in your Claude Code session. Any
MCP-speaking client works the same way — the server is client-agnostic.

## Config

| Env var | Default | Meaning |
|---------|---------|---------|
| `FLAVORMANCER_URL` | `http://127.0.0.1:8000` | Base URL of the Flavormancer API. |
| `FLAVORMANCER_TIMEOUT` | `60` | Per-request timeout (seconds). |

## Why it's built this way

The server owns **no** model logic — it's a protocol adapter over the API, so the models,
data gating, and honesty guarantees all live in one place (`predict.py` / `app.py`) and the
MCP surface stays a clean, testable boundary. The tool logic is factored into plain
functions (`_read_flavor`, `_analyze_formulation`, …) so it can be unit-tested without an
MCP client.
