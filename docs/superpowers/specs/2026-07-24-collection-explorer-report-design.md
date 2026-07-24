# Collection Explorer Report — Design

**Date:** 2026-07-24
**Status:** Approved (iterate-on-first-render)

## Goal

An interactive HTML report to explore/investigate a user's collection, with
filters for **player count**, **complexity**, and **playing time**. Reusable as
a per-user module, and extensible into standalone one-off reports (e.g.
`phils_spain_trip_2026`) that add a curated "menu" section highlighting
specific games.

## Architecture

Three layers, following the existing Quarto report conventions
(`reports/*.qmd` → HTML via `reports/render.py`, itables for interactive
tables, `src.reports.collection_data.load` for the data bundle).

### 1. Shared fragment — `reports/_collection_explorer.qmd`

The reusable core, included by both the registered module and any standalone
report:

- **Filter controls**: range sliders for player count, complexity
  (`average_weight`), and playtime, mounted above the table.
- **Collection table**: itables table built from the existing
  `format_collection_table(data.collection, data.games)` (already surfaces
  Players / Playtime / Complexity columns).
- **Summary plots**: reuse `plot_collection_by_category_static` and
  `plot_collection_by_year_static` from `_collection.qmd`.
- **Filter mechanism**: range sliders wired to DataTables via a small custom
  `$.fn.dataTable.ext.search` range callback (~40 lines JS). Client-side only;
  keeps the report self-contained (`embed-resources: true`).

Filter fidelity (sliders vs. bucketed dropdowns vs. plain search) is the first
thing to iterate on after seeing a local render.

### 2. Registered module — `collection`

- Thin `reports/collection_report.qmd`: params via `_setup.qmd`
  (`username`, `outcome`, `source`), includes `_collection_explorer.qmd`,
  dynamic per-user title.
- Register `"collection": "collection_report.qmd"` in `render.py` `_REPORTS`.
- Output path: `<slug>.html` (like predictions).
- Renderable via `just render <user> --report collection` (or the render CLI).

### 3. Standalone — `reports/phils_spain_trip_2026_report.qmd`

- Self-contained qmd (like `toms_ukge_2026_report.qmd`).
- Includes the same `_collection_explorer.qmd` fragment.
- Adds a **Menu section**: a **hardcoded Python list of BGG `game_id`s** at the
  top of the file, rendered as cover-tile cards (reuse the Tom report's
  card/badge styling helpers). Edit the file to change the menu.

## Data

- Source: `src.reports.collection_data.load` (existing bundle:
  `collection`, `games`, `outcomes`).
- Filter fields already available via `format_collection_table`:
  `min_players`/`max_players`, `min_playtime`/`max_playtime`, `average_weight`.

## Build order

1. Build layer 1 (`_collection_explorer.qmd`) + minimal layer 2
   (`collection_report.qmd` + registry wiring).
2. Render locally for `phenrickson`, get feedback on filter UX.
3. Iterate on filters.
4. Build layer 3 (standalone trip report + menu cards).

## Non-goals

- No server-side/live filtering — static self-contained HTML only.
- No new JS framework — vanilla JS + DataTables (already in use via itables).
- No changes to the model/prediction pipeline.
