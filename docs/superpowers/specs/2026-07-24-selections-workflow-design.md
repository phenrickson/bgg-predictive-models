# Selections Workflow — Design

**Date:** 2026-07-24
**Status:** Approved

## Goal

A two-step workflow for curating a proposed list of games from a collection:

1. **Select** — a Streamlit app to filter a user's collection by criteria, assign
   per-game **selection** (lock/maybe/no) and **status** (yes/no) labels, and
   export the result to `reports/selections/<name>.yaml`.
2. **Generate** — a standalone Quarto report (`rocky_bilbao_2026_menu`) that reads
   that YAML and renders Locks / Maybes / Others card sections plus The Menu table.

Shares the existing collection data + formatting layer with the JS explorer report;
the two front-ends (static JS explorer, local Streamlit selector) are separate.

## Step 1 — Streamlit Selector

**Location:** `src/streamlit/pages/8 Selections.py` (new page in the existing
multi-page app under `src/streamlit/`).

- **Data:** reuse `src.streamlit.components.collection_loader` for the user's
  collection and `src.reports.collection_data` / `build_explorer_payload` (which
  already joins `best_player_counts`) for the universe + suggestions.
- **Filters:** players (supported), complexity, playtime, status — narrow the
  universe. Same criteria semantics as the explorer (containment for ranges).
- **Editable grid:** `streamlit-aggrid` (already a dependency, currently unused).
  Columns: Game, Status(own/wishlist/…), Players, Recommended, Playtime,
  Complexity (read-only) + two **editable dropdown** columns:
  - `selection`: `lock` | `maybe` | `no` (default `no`)
  - `status`: `yes` | `no` (default `no`)
- **Export:** a text input for `<name>` (default `rocky_bilbao_2026`) and an
  Export button that writes `reports/selections/<name>.yaml` directly (local
  Streamlit process → real file write; no browser download).
- **Round-trip:** on load, if `reports/selections/<name>.yaml` exists, pre-populate
  the grid's label columns from it so a session can be resumed/edited.

**YAML shape:**
```yaml
name: rocky_bilbao_2026
username: phenrickson
criteria:            # the filters in effect at export, informational
  players: [5]
  complexity: {min: null, max: null}
  playtime: {min: null, max: null}
games:
  205637: {selection: lock,  status: yes}
  177736: {selection: maybe, status: no}
  31260:  {selection: no,    status: no}
```

**Launch:** `just select` → `uv run streamlit run src/streamlit/Home.py` (the page
appears in the existing app nav). Matches the existing `streamlit` Makefile target.

## Step 2 — Menu Report

**Location:** `reports/rocky_bilbao_2026_menu_report.qmd` (standalone,
self-contained, like `toms_ukge_2026_report.qmd`).

Reads `reports/selections/rocky_bilbao_2026.yaml`, joins the game_ids back to the
collection/games data + predictions for Pr(Yes), and renders:

- **Locks** — `selection == lock`. Card layout: image, name, description,
  recommended player count (badges), complexity (heat chip), playtime, Pr(Yes).
- **Maybes** — `selection == maybe`. Same card layout.
- **Others** — games meeting the export criteria but `selection == no`. Same cards.
- **The Menu** — `status == yes` (independent of selection). The explorer table
  (`_collection_explorer.qmd` fragment) filtered to these games.

**Label roles:**
- `selection` (lock/maybe/no) → which **section** a game appears in.
- `status` (yes/no) → the "actually bringing it" flag → defines **The Menu**.

## Reusable Core

Factor the shared rendering into modules both reports (and the selector) use:

- `src/reports/game_cards.py` — `game_cards_html(rows) -> str`: the card grid
  (image / name / description / rec-count badges / complexity chip / playtime /
  Pr(Yes)). Reuses the badge + weight-chip logic currently inline in the explorer
  fragment, lifted into small Python helpers so both the JS fragment and the cards
  produce identical badges.
- `src/reports/selections.py` — `load_selections(path) -> Selections`: parse+validate
  the YAML; helpers `locks()`, `maybes()`, `others(universe)`, `menu()`.
- The explorer fragment stays as-is; it just gains nothing here (already reusable
  via include).

## Non-goals

- No server/hosted Streamlit — local tool only (that's what makes the seamless
  file write possible).
- Selector does not replace the JS explorer; both remain.
- No auto-render chaining — user runs step 1, then renders step 2. (A convenience
  `just` recipe may render step 2 given a `<name>`.)

## Build order

1. Reusable core: `selections.py` (+ tests), `game_cards.py` (+ tests).
2. Step 2 report against a hand-written fixture `selections.yaml` (verifiable
   without the app).
3. Step 1 Streamlit selector + export (verified by round-tripping a YAML the
   report then consumes).
