# Aroma-head audit — where more data buys more heads

> **Update — supplement applied, then made food-only.** `build_aroma_supplement.py` adds a curated
> **public-domain** character-impact set (`aroma_supplement.csv`, resolved via PubChem) for the sparse
> descriptors below. Result: **42 → 164 heads** (food-safe core + non-food aroma-only tier; see the
> multi-industry note below), after open-government sourcing (see below) took the
> curated supplement to **~1,994 labelled molecules**. New heads include `coconut`, `nutty`, `caramel`,
> `winey`, `onion`, `honey`, `herbal`, `vanilla`, `buttery`, `balsamic`, `smoky`, `cinnamon`, `spicy`,
> `banana`, `musky`, `vegetable`, `anise`, `meaty`, and — from the open-gov sourcing that pushed every
> near-miss over the 10-positive bar — `burnt` (0.91), `tarry` (0.98), `apple` (0.88), `clove` (0.98),
> `cocoa` (0.94), `coffee` (0.95), and `creamy` (0.93). The n≈10 heads were **thickened** at the same
> time (e.g. `musky` n=13, `cinnamon` n=11, `rose` n=14, `woody` n=15), de-inflating the earlier
> single-scaffold scores. An **`odorless`** head was then added — the aroma parallel to taste's
> `tasteless` — from ~800 molecules HSDB documents as having no smell (water, salts, most sugars,
> involatile solids); it lands at **0.897**. Adding those clean negatives also finally pushed
> **`sweet`-odor over the bar (0.611 → 0.735)** — honest caveat: it now separates *sweet-smelling from
> no-smell* well, but is weaker at *sweet vs other sweetish odors*; treat as indicative. A further sourcing + vocab pass then added **`phenolic`, `cherry`, `cheesy`,
> `soapy`, `bready`** and fruit splits **`orange`, `waxy`, `tropical`, `peach`, `pear`**, and — by
> splitting `musty` out of `earthy` — **rescued `earthy` (0.695 → 0.94) and `fresh`** back over the bar.
> Continued sourcing + vocab passes then added a large batch — `acidic`, `grape`, `melon`, `apricot`,
> `wintergreen` (salicylates), and the floral/food sub-notes `violet` (ionones), `jasmine`, `lavender`,
> `muguet`, `ginger`, `berry`, `malty`, `marine`, `hay` — each sourced food-safe from the open-gov
> registers. `musty` finally crossed with the richer negatives; and a final niche pass added
> `cardamom`, `plum`, `tonka`, `tea`, `mushroom`, and `cassis` (the distinct blackcurrant-thiol note).
> Two more then crossed once we gave them a distinct chemical signature rather than terpene
> near-duplicates: `fennel` (0.91) and — the last hold-out — `neroli` (0.896), rescued by adding its
> **orange-blossom-specific** scaffolds (indole, methyl N-methylanthranilate, benzyl acetate,
> 2-phenylethanol) instead of the E/Z terpene twins (nerol/geraniol) that dedupe to one row under the
> connectivity-skeleton key. That was the **food-safe** ceiling at ~86.
>
> **Then we walked the road past it — the non-food aroma-only descriptor space — to 164 heads.** The
> corpus is now deliberately **multi-industry**, not food-only: odor is physical, so to read
> structure→smell well the corpus should include the fragrance space, and a per-molecule **safety flag**
> (open-gov food registers) — not corpus membership — gates edibility per mode. The same 164 heads serve
> every "Mancer" mode: **Flavormancer** (food chemistry), **Beveragemancer** (beverage), **Aromamancer**
> (fragrance), **Vapemancer** (inhalation), **Oilmancer** (essential oils). Molecule names are resolved
> to structures through PubChem and folded in flagged non-food. New heads cleared across several
> families — food (`celery` 1.0, `maple` 1.0, `saffron` 1.0, `pineapple` 0.99, `hazelnut` 0.99,
> `blueberry` 0.97, `allspice` 0.94, `cumin` 0.98, `thyme` 0.98, `oregano` 0.99, `dill` 0.92,
> `nutmeg` 0.98, `black_pepper` 0.96, `rosemary` 0.94, `coriander` 0.91, `mustard` 0.88, citrus
> `bergamot`/`mandarin`/`lime`/`grapefruit`/`yuzu` 0.89–0.99), fragrance/aroma-only (`aldehydic`,
> `resinous`, `terpenic`, `animalic`, `amber`, `leathery`, `powdery`, `cooling`, `metallic`, `ozonic`,
> `oakmoss`, `vetiver`, `patchouli`, `sandalwood`, `cedarwood`, `fir`, `myrrh`, `frankincense`,
> `labdanum`, `styrax`, `opoponax`, `davana`, `costus`, `elemi`, `turmeric`, `truffle`, `clary_sage`),
> and floral/fruit subtypes (`gardenia`, `lilac`, `osmanthus`, `ylang`, `jasmine`, `champaca`,
> `honeysuckle`, `freesia`, `cyclamen`, `linden`, `mimosa`, `narcissus`, `carnation`, `geranium`).
>
> **Honesty rails on the count.** Three caveats travel with every head and are shown in the UI:
> 1. **Subtype/structural heads are slices of space we already cover** — the floral subtypes and
>    `lactonic` (0.99, overlaps coconut/peach/creamy) clear on shared scaffolds, not orthogonal percepts.
> 2. **The ~1.00-AUROC heads are narrow single-scaffold classes** — `wintergreen` (salicylates),
>    `oakmoss` (orcinol esters), `myrrh` (furanosesquiterpenes), `cinnamon`, `musky`. We *tested* this:
>    adding diverse examples kept them at 1.00, confirming the percept **is** the chemical family (easy
>    separation, not fixable overfitting). Real robustness needs a learned representation (GNN), not more
>    data.
> 3. **`musty` does not ship** — "mould / cellar / damp" has no shared substructure a fingerprint can
>    grab (stuck ~0.66–0.68). Its molecules were **redistributed** to the heads they belong to
>    (`corky` — the distinct haloanisole cork-taint scaffold, 0.98 — plus `earthy`/`mushroom`, which
>    *sharpened* `earthy` 0.88→0.95). Likewise the redundant `tuberose` head was **consolidated** into
>    its white-floral neighbours (`jasmine`/`gardenia`/`ylang`/`champaca`), nearly doubling their example
>    counts and de-inflating their small-n scores.
>
> `sweet`-odor rides the bar (0.72, documented-only until a GNN). Net: **164 aroma heads / 170 total
> (with the 6 taste heads).** The count is inherently ±2–3 at the margin — ~15 heads sit right at the
> 10-positive / 0.70-AUROC boundary, so any data change reshuffles which marginal heads ship.
>
> **Build determinism (a real fix, not a count).** The supplement build resolves molecule NAMES via
> PubChem; a rate-limit/timeout used to silently drop a molecule and churn the marginal (n≈10) heads
> build-to-build — e.g. `cocoa` once cratered 10→4 from one flaky build, not from any data change. A
> persistent **`resolve_cache.json`** now caches every resolution, so the build is deterministic and
> network-independent; the count is stable and reproducible.
>
> **Provenance of the fragrance associations (honest).** The food-clearance data is cited to open-gov
> registers (EU/GB Union List, US 21 CFR / FDA SAF). The **fragrance** structure→odor associations,
> however, come from **general public flavor/fragrance chemistry knowledge** (including associations
> supplied by Claude Opus 4.8), resolved to public-domain PubChem structures — *not* from any proprietary
> compilation (Good Scents / Leffingwell / GS-LF are excluded everywhere). Each is a defensible public
> fact (odor descriptors are non-copyrightable, *Feist v. Rural*), but the **volume** means a
> **spot-verification + IP/provenance review is the standing pre-commercial gate**: per association, find
> ≥2 independent public sources; drop anything that traces only to a proprietary set; then counsel signs
> off before any commercial sale.
>
> **Stereochemistry (honest scope).** The corpus dedupes by **connectivity skeleton** (first InChIKey
> block), so enantiomers (R/S-limonene, R/S-carvone) collapse to one row and E/Z pairs (nerol/geraniol)
> collapse too. This is deliberate today: the Morgan fingerprint is **achiral** (and encodes no E/Z at
> all), and the corpus holds only **one labelled stereoisomer per skeleton** — no R-/S- pair with
> *different* documented odors — so splitting them would add identical feature rows, not signal. Reading
> R-carvone (spearmint) apart from S-carvone (caraway) is a real upgrade but **data-gated**: it needs
> chirality-aware fingerprints **and** enantiomer-resolved odor labels (roadmap: chirality-aware model).
> Both the whole EU/GB Union
> List (~2,200 authorised, OGL v3) **and** the FDA SAF inventory (~2,800 resolved, US public domain) are
> ingested into the browse universe (~8.6k molecules), each cited; excluded non-food aromatics are kept
> in `aroma_only_seed.csv` for the aroma-only mode.
>
> **Open-government sourcing.** Three verification passes gathered food-authorised character
> molecules per descriptor, each **confirmed on an open-government register** — the EU/GB Union List
> (`data.food.gov.uk`, OGL v3) or US 21 CFR / FDA SAF (public domain) — and recorded in
> `aroma_additions.csv` (+ per-molecule citation in `food_safe_supplement.csv`, now ~175 molecules).
> Commercial compilations (Good Scents / Leffingwell / FEMA library) were finding-aids for a public FL
> number only. Prohibited/restricted molecules were excluded (coumarin, methyleugenol, estragole,
> safrole); several fabricated FL numbers were caught and discarded by fetch-to-confirm.
>
> **Food-only + open-government provenance (this is a flavor app).** Non-food odorants were **removed
> from the corpus**, not merely flagged: `isovanillin` and `habanolide` (no food clearance),
> `methyleugenol` (delisted from 21 CFR 172.515 in 2018), and `estragole`/`methyl chavicol` (prohibited
> as an added flavouring in the EU, Reg. 1334/2008 Annex III). Heads that leaned on them (`musky`,
> `vanilla`, `spicy`) were **re-based on food-authorised molecules** and retrained above the bar. Every
> food-clearance is cited to an **open-government register** — the EU/GB Union List (`data.food.gov.uk`,
> OGL v3) or US 21 CFR / FDA SAF (public domain) — in `food_safe_supplement.csv`; **no commercial
> compilation** (Good Scents / Leffingwell / FEMA library) is a source, only a finding-aid for the
> public FL number.
>
> **Small-n caveat (honest):** the *highest*-AUROC heads are small, structurally-homogeneous classes —
> e.g. `cinnamon` and `musky` reach CV-AUROC **~1.00** only because their ~10–12 positives are one
> scaffold family (cinnamaldehyde esters; macrocyclic musk lactones/ketones) the fingerprint separates
> trivially. That is memorising a scaffold, not a superb general model: these heads are **narrow and
> high-variance** and will sharpen — or get honestly re-scored — with more diverse examples. Treat
> AUROC on n≈10 heads as indicative, not gospel. The numbers below describe the **pre-supplement**
> baseline + the standing plan.


An honest look at the aroma model: which odor descriptors we can predict from structure today,
which are *just* out of reach, and exactly what data would unlock more. Regenerate the numbers
with `python training/train_aroma.py` (deterministic, `random_state=42`).

**Bar:** an odor descriptor ships as a head only if it has **≥ 10 positive examples** and clears
**CV-AUROC ≥ 0.70** (5-fold). Of **57** candidate descriptors in the current public-domain corpus
(981 labelled molecules): **24 ship**, **2 were evaluated and dropped**, **31 were skipped for too
few positives**.

## 1. The 24 shipping heads (for reference)

AUROC 0.711 → 0.975. The weakest are small-sample (`grassy` 0.711 / n=10, `woody` 0.759 / n=12,
`green` 0.773 / n=13) — they ship, but their AUROC estimates are high-variance at n≈10–13, so more
data would **robustify** them, not just add new ones.

## 2. The near-misses today (post-sourcing)

| descriptor | n_pos | CV-AUROC | verdict |
|---|---|---|---|
| **musty** | 35 | 0.68 | **Diffuse.** Split out of `earthy` (which it *un-confused* — earthy jumped to 0.94); but "musty/stale/mould/cellar" has no shared substructure on its own, so it stays documented-only. A GNN candidate. `earthy` and `fresh` were both **rescued** over the bar this pass. |

**On the "sweet"-odor reversal.** Earlier passes had odor-sweet stuck at 0.61–0.67 and we treated it as
unlearnable-from-fingerprints. Adding the **`odorless`** head (~800 molecules with no documented smell)
put clean negatives in the pool and pushed **odor-sweet to 0.735 — it now ships as an aroma head.**
Honest caveat: much of that lift is the model cleanly separating *sweet-smelling from no-smell*; it is
still weaker at *sweet vs other sweetish odorants*, so treat the score as indicative. This is **distinct
from taste-sweet** (a separate trained head from ChemTastesDB); the card shows each under its own
**TASTE** / **AROMA** section, so a molecule can read sweet-tasting, sweet-smelling, or both.

## 3. Skipped for too few positives — the real opportunity (n_pos < 10)

These have no head *at all* because they can't even be evaluated at the 10-positive bar. This is where
data collection has the highest leverage.

- **Near the threshold (n_pos 5–9)** — a few more examples each and they get evaluated, several likely
  clearing 0.70:
  `winey(9) · buttery(8) · onion(8) · herbal(7) · balsamic(7) · honey(7) · fresh(7) · musky(7) ·
  vegetable(7) · caramel(6) · coconut(6) · anise(6) · burnt(6) · tarry(6) · apple(5)`
- **Very sparse (n_pos 1–4)** — need substantial data:
  `vanilla(4) · banana(4) · meaty(4) · nutty(3) · clove(3) · cheesy(3) · waxy(3) · soapy(3) ·
  creamy(2) · fecal(2) · smoky(1) · cherry(1) · cinnamon(1) · coffee(1) · cocoa(1)`

**This explains the "coconut — no clean carrier, data-gated" message**: `coconut` has only **6** labelled
positives, below the bar, so no head exists to predict or suggest it. The classic carrier
(γ-nonalactone) simply isn't labelled "coconut" enough times in the clean public set.

## 4. The plan (owner: Aaron / `area:training`)

1. **Quick win:** collect a modest set of labelled **`spicy`** molecules → re-train → likely a 25th head.
2. **Highest leverage:** push the **n_pos 5–9** descriptors over 10 (especially `caramel, coconut, honey,
   buttery, herbal, balsamic, vanilla`) — small, targeted labelling unlocks a batch of new heads and,
   downstream, real gap-analysis carriers for those notes (fixes the "data-gated" suggestions).
3. **Not a data fix:** `sweet`-odor (and other diffuse percepts) — revisit with a **GNN / learned
   representation**, not more fingerprint data.
4. **Robustify:** add examples for the fragile shipping heads (`grassy, woody, green, rose, rancid, pine`).

Only truly-open (CC-BY / public-domain) odor data or the customer's own labels — never the restricted
GS-LF / Leffingwell sets in the commercial edition (see `docs/DATA-SOURCES.md`).
