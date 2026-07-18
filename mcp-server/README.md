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

| Tool | What it does |
|------|--------------|
| **`read_flavor(molecule)`** | Predict a single molecule's taste (6 heads) + aroma, with GRAS status, structural alerts, and applicability-domain flag. Name or SMILES. |
| **`analyze_formulation(ingredients, target?, processes?)`** | Read a whole recipe *before you pour*: blended note-profile (weighted by odor impact), the overpowering-component flag, a target gap analysis with food-safe add/cut suggestions, a hazard screen, and honest data-gate notes. |
| **`screen_mixture(ingredients, processes?)`** | Flag **documented** food hazards when ingredients combine (e.g. benzoate + ascorbate → benzene), gated on process (heat/refining/fermentation). Curated screen, not a reaction predictor. |
| **`find_molecules(notes, food_safe?)`** | Find molecules that carry a set of taste/aroma notes (e.g. `["citrus","fresh","sweet"]`), ranked, food-safe by default. |

Every tool returns model **predictions only** — never a safety, GRAS, or regulatory determination.

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
