# Collection Explorer Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An interactive per-user HTML report to explore a collection with player-count / complexity / playtime filters, reusable as a shared fragment, plus a standalone trip report that adds a curated "menu" of games.

**Architecture:** A shared Quarto fragment (`_collection_explorer.qmd`) holds the filterable itables table + summary plots. A thin registered module (`collection_report.qmd`) includes it and renders per-user via `reports/render.py`. A standalone report (`phils_spain_trip_2026_report.qmd`) includes the same fragment and adds a menu-card section built from a hardcoded game_id list.

**Tech Stack:** Quarto → HTML, itables (DataTables), polars/pandas, vanilla JS range filter, matplotlib static plots. `uv run` for all Python.

## Global Constraints

- Use `uv run python` for all Python invocations (never bare `python`/`python3`).
- Reports must be self-contained: `embed-resources: true`. No external JS/CSS/font hosts.
- No new JS framework — vanilla JS + DataTables (bundled by itables) only.
- The `collection` report needs collection+games data only; it must render WITHOUT a finalized model (unlike predictions/model).
- Filenames use `slugify_username`; the display name flows via the `username` Quarto param.
- Offline/test renders set `BGG_REPORTS_OFFLINE=1` (stubs BQ fetchers with empty frames).

---

### Task 1: Register the `collection` report and skip its model pre-flight

**Files:**
- Modify: `reports/render.py` (the `_REPORTS` dict ~line 44; the `--report` help string; the pre-flight `select_candidate` guard ~line 275)
- Modify: `tests/reports/test_render_smoke.py` (add `collection` param)
- Create: `reports/collection_report.qmd`

**Interfaces:**
- Produces: `_REPORTS["collection"] = "collection_report.qmd"`; output path `<slug>.html` (reuses the default branch of `_output_rel_path`, no code change there).
- Consumes: existing `_render_one(...)` signature (unchanged), `_setup.qmd` params.

- [ ] **Step 1: Add the collection entry to the smoke test (failing test)**

In `tests/reports/test_render_smoke.py`, add to the parametrize list:
```python
        ("collection", "phenrickson.html"),
```

- [ ] **Step 2: Run it, expect failure**

Run: `uv run pytest tests/reports/test_render_smoke.py -v -k collection`
Expected: FAIL — either `argparse` rejects `--report collection` (not in choices) or the qmd is missing. (Skips entirely if `quarto` not on PATH — if skipped, note it and rely on the manual render in Step 7.)

- [ ] **Step 3: Register the report**

In `reports/render.py`, extend `_REPORTS`:
```python
_REPORTS = {
    "predictions": "predictions_report.qmd",
    "model": "model_report.qmd",
    "collection": "collection_report.qmd",
}
```
Update the `--report` help string to mention collection.

- [ ] **Step 4: Skip the model pre-flight for the collection report**

The pre-flight block (`reports/render.py` ~line 275) calls `select_candidate(...)` and appends to `failures` on `MissingArtifactsError`. The collection report has no model, so guard it:
```python
        if not args.fixture and args.report != "collection":
            from src.reports.collection_data import (
                MissingArtifactsError,
                select_candidate,
            )
            ...
```
(Wrap the existing block in this condition — only add `and args.report != "collection"` to the existing `if not args.fixture:`.)

- [ ] **Step 5: Create `reports/collection_report.qmd`**

Model the front matter on `predictions_report.qmd` (params, theme, `embed-resources: true`), include setup + explorer fragment. `_setup.qmd` loads the data bundle into `data`, `USERNAME`, `USERNAME_SLUG`.
```markdown
---
title: "Collection Explorer"
date: today
format:
  html:
    toc: true
    toc-location: right
    toc-depth: 2
    code-fold: true
    code-summary: "Show the code"
    embed-resources: true
    theme:
      - cosmo
      - theme.scss
    css: styles.css
    page-layout: full
execute:
  echo: false
  warning: false
  message: false
  daemon: false
params:
  username: phenrickson
  outcome: own
  source: local
  candidate: ""
  fixture: false
---

{{< include _setup.qmd >}}

```{python}
#| output: asis
#| echo: false
print(
    f'---\ntitle: "{USERNAME}\'s Collection"\nsubtitle: "Explore & Filter"\n---'
)
```

# About

Explore **`{python} USERNAME`**'s board game collection. Use the filters to
narrow by player count, complexity, and playing time.

{{< include _collection_explorer.qmd >}}
```

- [ ] **Step 6: Run the smoke test**

Run: `uv run pytest tests/reports/test_render_smoke.py -v -k collection`
Expected: PASS (or SKIP if no Quarto — then Step 7 is the real gate). NOTE: this depends on Task 2's fragment existing; if running Task 1 in isolation, create a one-line placeholder `_collection_explorer.qmd` (`# Collection` heading) so the render succeeds, and replace it in Task 2.

- [ ] **Step 7: Manual render + eyeball**

Run: `uv run python -m reports.render --report collection --username phenrickson --outcome own --fixture --output-dir /tmp/collexp`
Expected: exits 0, `/tmp/collexp/fixture_user.html` exists and opens. (Use `--fixture` to avoid needing real artifacts/creds.)

- [ ] **Step 8: Commit**

```bash
git add reports/render.py reports/collection_report.qmd tests/reports/test_render_smoke.py
git commit -m "feat: register collection report module"
```

---

### Task 2: Custom faceted-filter collection table (replaces itables)

**Status note (already landed in earlier commits, keep):** the `collection`
report renders to `collection/<slug>.html`; `_fetch_games_metadata` left-joins
`bgg-data-warehouse.<dataset>.best_player_counts` to add `best_player_counts` /
`recommended_player_counts`; `format_collection_table` adds **Best** and
**Recommended** columns (numeric-sorted via `_sort_player_counts`); the
Overview charts are removed.

**Design change:** itables/DataTables is dropped for this report. Instead we
render a **custom self-contained HTML table** with **faceted filters** driven by
a per-column config. Each column declares a filter `kind`:

- `discrete` — one chip per distinct value; click to toggle (OR within column).
  Used by: Status.
- `multi` — cell is a comma-separated string (`"2, 3, 4"`); tokenize into
  individual chips; a row matches a chip if the token is in its set.
  Used by: Best, Recommended.
- `range` — numeric min/max inputs; row matches if its value (or its range,
  for `"a–b"` cells) overlaps [min,max]; unknown/blank values are kept.
  Used by: Your rating, Playtime, Complexity.
- (no filter) — Game, Players.

Filtering is **AND across columns, OR within a column**. Column-header click
sorts (asc/desc toggle). No pagination needed initially (≈160 rows); a simple
"showing N of M" count suffices. All vanilla JS embedded in the fragment; the
table data is emitted as a JSON blob the JS reads (so no DOM scraping).

**Files:**
- Modify: `reports/_collection_explorer.qmd` (replace itables block with the
  custom table + config + JS)
- Modify: `reports/styles.css` (replace `.filter-bar` block with chip + table styles)
- Create: `src/reports/explorer.py` — builds the JSON payload the JS consumes
- Test: `tests/reports/test_explorer.py`

**Interfaces:**
- Consumes: `data.collection`, `data.games`, `format_collection_table` (already
  produces the exact display columns/HTML).
- Produces: `src.reports.explorer.build_explorer_payload(collection, games) -> dict`
  with shape:
  `{"columns": [{"label": str, "kind": "discrete"|"multi"|"range"|"none"}],
    "rows": [[cell_html_or_value, ...], ...]}`
  Row cell values are the same strings `format_collection_table` produces (Game
  is HTML; Best/Recommended are `"2, 3"`; Playtime is `"60–120 min"`, etc.).

- [ ] **Step 1: Failing test for the payload builder**

Create `tests/reports/test_explorer.py`:
```python
import polars as pl
from src.reports.explorer import build_explorer_payload


def _coll():
    return pl.DataFrame({"game_id": [1, 2], "owned": [True, False],
                         "wishlist": [False, True], "user_rating": [9.0, None]})


def _games():
    return pl.DataFrame({
        "game_id": [1, 2], "name": ["Alpha", "Beta"],
        "year_published": [2020, 2021], "min_players": [2, 1], "max_players": [4, 5],
        "min_playtime": [60, 30], "max_playtime": [120, 30], "average_weight": [3.1, 1.5],
        "best_player_counts": ["4, 3", "2"], "recommended_player_counts": ["2, 3, 4", "1, 2"],
    })


def test_payload_columns_have_kinds():
    p = build_explorer_payload(_coll(), _games())
    kinds = {c["label"]: c["kind"] for c in p["columns"]}
    assert kinds["Status"] == "discrete"
    assert kinds["Best"] == "multi"
    assert kinds["Recommended"] == "multi"
    assert kinds["Complexity"] == "range"
    assert kinds["Game"] == "none"


def test_payload_rows_match_table():
    p = build_explorer_payload(_coll(), _games())
    assert len(p["rows"]) == 2
    # Best column numeric-sorted: "4, 3" -> "3, 4"
    best_idx = [c["label"] for c in p["columns"]].index("Best")
    assert p["rows"][0][best_idx] in ("3, 4", "2")
```

- [ ] **Step 2: Run it, expect failure**

Run: `uv run pytest tests/reports/test_explorer.py -v`
Expected: FAIL — `ModuleNotFoundError: src.reports.explorer`.

- [ ] **Step 3: Implement `src/reports/explorer.py`**

```python
"""Build the JSON payload for the custom faceted-filter collection table.

Reuses format_collection_table for the exact display cells, then attaches a
per-column filter `kind` the client JS uses to render controls."""
from __future__ import annotations

import polars as pl

from src.reports.tables import format_collection_table

# label -> filter kind. Columns absent here default to "none".
_COLUMN_KINDS = {
    "Status": "discrete",
    "Your rating": "range",
    "Best": "multi",
    "Recommended": "multi",
    "Playtime": "range",
    "Complexity": "range",
}


def build_explorer_payload(collection: pl.DataFrame, games: pl.DataFrame) -> dict:
    table = format_collection_table(collection, games)
    labels = list(table.columns)
    columns = [{"label": l, "kind": _COLUMN_KINDS.get(l, "none")} for l in labels]
    rows = table.astype(object).where(table.notna(), "").values.tolist()
    return {"columns": columns, "rows": rows}
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/reports/test_explorer.py -v`
Expected: PASS.

- [ ] **Step 5: Rewrite the fragment — custom table + config + JS**

Replace the entire body of `reports/_collection_explorer.qmd` with: a heading,
a `#explorer-filters` container, a `#explorer-mount` table container, a Python
cell that emits `build_explorer_payload(...)` as a JSON script tag, and a
`{=html}` `<script>` that:
- reads the JSON payload;
- builds controls per column kind (`discrete`/`multi` → chip set of distinct or
  tokenized values, sorted numerically when all-numeric; `range` → min/max
  number inputs), plus a Reset button and a "showing N of M" counter;
- renders `<table>` rows, re-rendering on every filter change (AND across
  columns, OR within a column; `range` overlap keeps unknowns);
- sorts on header click (numeric when the column parses as numeric, else
  lexical), toggling asc/desc.
Emit the payload cell as:
```{python}
#| output: asis
import json
from src.reports.explorer import build_explorer_payload
_payload = build_explorer_payload(data.collection, data.games)
print('<script id="explorer-data" type="application/json">'
      + json.dumps(_payload) + '</script>')
```
Keep the JS defensive: run on DOMContentLoaded; if `#explorer-data` is missing
or has no rows, render an empty-state line instead of throwing.

- [ ] **Step 6: Replace filter styles**

In `reports/styles.css`, replace the `.filter-bar` block (added in the prior
commit) with styles for `.explorer-filters`, `.facet` (a labeled group),
`.chip` / `.chip.active`, `.explorer-range input`, the `#explorer-table`
(borders, sticky header, hover, `dt-center` centering for numeric cols), and
`.explorer-count`. Keep it theme-aware (works on the dark cosmo theme).

- [ ] **Step 7: Render with REAL data and eyeball**

Run: `uv run python -m reports.render --report collection --username phenrickson --outcome own --output-dir /tmp/collexp`
Expected: exits 0; open `/tmp/collexp/collection/phenrickson.html`. Verify:
Status chips, Best/Recommended player-count chips (2,3,4,…), and rating/
playtime/complexity ranges all filter; clicking "2" under Recommended narrows
to games recommended at 2 players; multiple chips OR within a facet; ranges AND
across facets; header-click sorts; Reset clears; "showing N of M" updates.
**Stop here and get user feedback before proceeding.**

- [ ] **Step 8: Commit**

```bash
git add reports/_collection_explorer.qmd reports/styles.css src/reports/explorer.py tests/reports/test_explorer.py
git commit -m "feat: custom faceted-filter collection explorer table"
```

---

### Task 3: Standalone trip report with a curated menu section

**Files:**
- Create: `reports/phils_spain_trip_2026_report.qmd`
- Create: `src/reports/menu.py` (menu-card formatting helper)
- Test: `tests/reports/test_menu.py`

**Interfaces:**
- Consumes: `data` bundle (via a small inline load, since standalone reports render without the module CLI) OR `--fixture`; `bgg_link`/`img_tag` patterns from `toms_ukge_2026_report.qmd`.
- Produces: `src.reports.menu.menu_cards_html(games_df, game_ids) -> str` — returns an HTML string of cover-tile cards for the given ids, in the given order, skipping ids not present.

- [ ] **Step 1: Write the failing test for `menu_cards_html`**

Create `tests/reports/test_menu.py`:
```python
import polars as pl
from src.reports.menu import menu_cards_html


def _games():
    return pl.DataFrame({
        "game_id": [1, 2, 3],
        "name": ["Alpha", "Beta", "Gamma"],
        "year_published": [2020, 2021, 2022],
        "image": ["http://x/1.png", "http://x/2.png", None],
    })


def test_menu_cards_orders_and_filters():
    html = menu_cards_html(_games(), [3, 1, 99])
    # id 99 absent -> skipped; order preserved: Gamma before Alpha
    assert html.index("Gamma") < html.index("Alpha")
    assert "Beta" not in html
    # links to BGG by id
    assert "boardgamegeek.com/boardgame/3" in html
    assert "boardgamegeek.com/boardgame/1" in html


def test_menu_cards_empty():
    assert menu_cards_html(_games(), []) == ""
```

- [ ] **Step 2: Run it, expect failure**

Run: `uv run pytest tests/reports/test_menu.py -v`
Expected: FAIL — `ModuleNotFoundError: src.reports.menu`.

- [ ] **Step 3: Implement `src/reports/menu.py`**

```python
"""Render a curated 'menu' of games as cover-tile cards for standalone reports."""
from __future__ import annotations

import polars as pl


def _img(url) -> str:
    if not url:
        return '<div class="menu-thumb menu-thumb-empty"></div>'
    return f'<img src="{url}" class="menu-thumb" loading="lazy" />'


def menu_cards_html(games: pl.DataFrame, game_ids: list[int]) -> str:
    """HTML card grid for `game_ids`, in order, skipping ids absent from `games`."""
    if not game_ids or games.height == 0:
        return ""
    by_id = {int(r["game_id"]): r for r in games.iter_rows(named=True)}
    cards = []
    for gid in game_ids:
        row = by_id.get(int(gid))
        if row is None:
            continue
        name = row.get("name") or row.get("game_name") or f"Game {gid}"
        year = row.get("year_published")
        yr = f" ({int(year)})" if year not in (None, 0) else ""
        link = (
            f'<a href="https://boardgamegeek.com/boardgame/{int(gid)}" '
            f'target="_blank" rel="noopener">{name}{yr}</a>'
        )
        cards.append(
            f'<div class="menu-card">{_img(row.get("image"))}'
            f'<div class="menu-card-title">{link}</div></div>'
        )
    if not cards:
        return ""
    return '<div class="menu-grid">' + "".join(cards) + "</div>"
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/reports/test_menu.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Create the standalone report**

Create `reports/phils_spain_trip_2026_report.qmd`. It loads the collection bundle inline (real data; falls back to offline stubs if `BGG_REPORTS_OFFLINE=1`), includes the explorer fragment, then renders the menu. Front matter mirrors `toms_ukge_2026_report.qmd` (self-contained, `output-file: phils_spain_trip_2026_report.html`). Set params for the fragment's expectations by defining `USERNAME`/`data` in a setup cell — reuse `_setup.qmd` via include so the same names exist:
```markdown
---
title: "Phil's Spain Trip 2026"
subtitle: "Collection to bring — and what's on the menu"
date: today
format:
  html:
    output-file: phils_spain_trip_2026_report.html
    theme: {light: cosmo, dark: darkly}
    toc: true
    toc-location: right
    embed-resources: true
    page-layout: full
    css: styles.css
execute: {echo: false, warning: false, message: false}
params:
  username: phenrickson
  outcome: own
  source: local
  candidate: ""
  fixture: false
---

{{< include _setup.qmd >}}

# Explore

{{< include _collection_explorer.qmd >}}

# On the Menu

Games I'm bringing as options for the group.

```{python}
#| output: asis
from src.reports.menu import menu_cards_html
MENU_GAME_IDS = [
    # EDIT ME: BGG game_ids for the menu, in display order.
]
print(menu_cards_html(data.games, MENU_GAME_IDS))
```
```

- [ ] **Step 6: Add menu-card styles**

Append to `reports/styles.css`:
```css
.menu-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:1rem; margin:1rem 0; }
.menu-card { border:1px solid var(--bs-border-color,#dee2e6); border-radius:.5rem; overflow:hidden; text-align:center; }
.menu-thumb { width:100%; aspect-ratio:1; object-fit:cover; display:block; }
.menu-thumb-empty { width:100%; aspect-ratio:1; background:#e9ecef; }
.menu-card-title { padding:.5rem; font-size:.85rem; }
```

- [ ] **Step 7: Render the standalone report and eyeball**

Populate `MENU_GAME_IDS` with 2-3 real ids from phenrickson's collection first (pick from an earlier render). Then:
```bash
cd reports && uv run quarto render phils_spain_trip_2026_report.qmd \
  -P username=phenrickson -P outcome=own -P source=local && cd ..
```
Expected: `reports/phils_spain_trip_2026_report.html` exists; opens with the explorer + a menu grid of the chosen games. (If no GCP creds locally, prefix `BGG_REPORTS_OFFLINE=1` — menu will be empty but the report must still render.)

- [ ] **Step 8: Commit**

```bash
git add src/reports/menu.py tests/reports/test_menu.py reports/phils_spain_trip_2026_report.qmd reports/styles.css
git commit -m "feat: add Phil's Spain Trip 2026 standalone report with menu section"
```

---

## Notes for the implementer

- `_setup.qmd` is the canonical place params get read and `data`/`USERNAME`/`USERNAME_SLUG` get defined — do not re-fetch data in the fragment; consume what `_setup.qmd` provides.
- The `collection` report intentionally renders without a finalized model. If you see a `MissingArtifactsError` for it, the Task 1 Step 4 guard is missing.
- Real-data renders need GCP creds (repo convention: `credentials/service-account-key.json` or `GOOGLE_APPLICATION_CREDENTIALS`). Without them, use `--fixture` (module) or `BGG_REPORTS_OFFLINE=1` (standalone) — both still exercise the render path.
- Column indices in the filter JS (3/4/5) are coupled to `format_collection_table`'s column order. If that order changes, update `COLS`.
