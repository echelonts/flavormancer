# Vendored static assets

Third-party browser assets served locally by `app.py` so the demo workbench stays
self-contained (no CDN dependency, works on an air-gapped/LAN box).

- **`3Dmol-min.js`** — [3Dmol.js](https://3dmol.csb.pitt.edu/) v2.4.2, an interactive
  WebGL molecular viewer by David Koes and contributors. **License: BSD-3-Clause**
  (permissive, commercial-OK with attribution). Renders the 3D conformer that
  `/api/structure3d` generates (RDKit ETKDG embed + MMFF optimize). `3Dmol-min.js.LICENSE.txt`
  is the upstream license header extracted by the build.
- **`wordmark.ttf`** — [Cinzel Decorative](https://fonts.google.com/specimen/Cinzel+Decorative)
  (Natanael Gama), the header wordmark font. **License: SIL Open Font License 1.1** (free for
  commercial use). Served locally via `/static/{fname}`.
- **`logo.png`** — the Flavormancer badge (256px), header logo.
- **`social.png`** — 1280×640 Open-Graph / social-preview banner (logo + gradient wordmark).
  Referenced by the demo's OG meta tags; also upload it as the GitHub repo's Social preview
  (Settings → General → Social preview) so the repo link renders richly when shared.
