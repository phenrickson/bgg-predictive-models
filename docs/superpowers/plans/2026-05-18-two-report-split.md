# Two-Report Split + Room for General Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `reports/collection_report.qmd` into a daily-rendered predictions report and an on-promote model report, publish them under typed site namespaces (`/{user}.html`, `/model/{user}.html`, reserved `/reports/`), and add a second CI workflow plus a promoter-triggered render seam.

**Architecture:** Both `.qmd` templates `{{< include _setup.qmd >}}` and `{{< include _collection.qmd >}}` so no section logic is duplicated. `render.py` gains a required `--report {predictions,model}` arg that picks the template and the output path namespace. `build_index.py`'s discovery loop is extracted into a unit-testable pure function that detects which of the two HTMLs exists per user. The existing matrix workflow becomes the predictions pipeline; a near-identical second workflow is the model pipeline, dispatched by the promoter.

**Tech Stack:** Python 3.12, Quarto, polars, itables, pytest, GitHub Actions, uv. Use `uv run python` for all Python invocations (project convention).

---

## File Structure

**Created:**
- `reports/_setup.qmd` — shared Quarto include: frontmatter is *not* here (includes can't carry frontmatter); this holds the parameters cell + setup/import/offline/fixture/load chunk lifted verbatim from `collection_report.qmd`.
- `reports/_collection.qmd` — shared Quarto include: the "Collection" section (Types of Games, collection-by-year, collection table) lifted verbatim.
- `reports/predictions_report.qmd` — frontmatter + `{{< include _setup.qmd >}}` + About + `{{< include _collection.qmd >}}` + Predictions sections.
- `reports/model_report.qmd` — frontmatter + `{{< include _setup.qmd >}}` + About + `{{< include _collection.qmd >}}` + Modeling + Assessment + Model Details.
- `tests/reports/test_build_index.py` — unit tests for the extracted index discovery function.

**Modified:**
- `reports/render.py` — add required `--report {predictions,model}`; map it to template + output path.
- `reports/build_index.py` — extract the disk-scan loop into `discover_reports(artifacts_root) -> list[dict]` (importable, no Quarto), each dict carrying `has_predictions` / `has_model` based on what exists in the *output* tree.
- `reports/index.qmd` — render two links per user (Predictions / Model) using the discovered flags; keep the reserved Reports section empty.
- `tests/reports/test_render_smoke.py` — parametrize over `--report predictions` and `--report model`, assert the correct output path.
- `.github/workflows/build-collection-reports.yml` — Sunday cron → daily cron; pass `--report predictions`.
- `.github/workflows/build-model-reports.yml` — **new**, structurally mirrors the predictions workflow; `workflow_dispatch` with `users` + `outcome`; passes `--report model`; renders into `/model/`.

**NOT modified:** `src/reports/collection_data.py`, `src/reports/tables.py`, `src/reports/format.py`, `src/collection/viz.py`, `docker/reports.Dockerfile`, `docker-reports-build.yml` — both templates ship in the existing image; no loader/helper changes are needed because the section logic moves verbatim into includes.

**Deleted at the end (Task 9):** `reports/collection_report.qmd`, once both new templates render and nothing references it.

---

### Task 1: Extract shared setup include

**Files:**
- Create: `reports/_setup.qmd`
- Reference (read, do not yet modify): `reports/collection_report.qmd:32-115`

- [ ] **Step 1: Create `reports/_setup.qmd`**

This file holds everything from the parameters cell through the `arts = data.outcomes[OUTCOME]` line. It has **no YAML frontmatter** — Quarto includes inherit the including document's frontmatter. Copy the two code chunks verbatim from `collection_report.qmd` (the `#| tags: [parameters]` cell and the `#| label: setup` cell), i.e. lines 32 through 115 of the current file. The content is exactly:

```
```{python}
#| tags: [parameters]
#| include: false

# Default values; Quarto injects overrides from the YAML `params:` block
# above (and from `-P key=value` on the command line) by replacing this
# cell at render time. Don't read these from anywhere else — the whole
# point is that Quarto controls them.
username = "phenrickson"
outcome = "own"
source = "local"
candidate = ""
fixture = False
```

```{python}
#| label: setup
#| include: false
import os
import sys

# Ensure project root is importable regardless of Quarto's CWD.
_project_root = os.environ.get("BGG_PROJECT_ROOT") or os.getcwd()
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import polars as pl
import pandas as pd
from itables import show as itables_show

from src.reports.collection_data import load
from src.reports.tables import (
    build_status_lookup,
    build_topn_by_year_html,
    format_collection_table,
    format_eval_predictions,
    format_eval_table,
    format_model_details,
    format_predictions_with_images,
)
from src.reports.format import model_kind
from src.collection.viz import (
    metrics_table,
    plot_collection_by_category_static,
    plot_collection_by_year_static,
    plot_feature_importance,
    plot_separation_static,
)

# Backwards-compatible aliases for the previously-uppercased names. The
# rest of the qmd reads USERNAME / OUTCOME / etc.
USERNAME = username
OUTCOME = outcome
SOURCE = source
CANDIDATE = candidate or None

# Offline mode: stub BQ-backed fetchers with empty DataFrames so the
# template renders without GCP credentials. Used by the smoke test.
if os.environ.get("BGG_REPORTS_OFFLINE") == "1":
    from src.reports import collection_data as _cd

    _empty = pl.DataFrame()
    _cd._fetch_collection_snapshot = lambda username: _empty
    _cd._fetch_games_metadata = lambda: _empty
    _cd._fetch_upcoming_predictions = lambda u, o: _empty

candidates = {OUTCOME: CANDIDATE} if CANDIDATE else None

# Fixture mode: skip BQ + artifact reads and render against synthetic
# data shaped like the real loader output. Used for fast styling iteration.
if fixture:
    from src.reports.fixtures import build_fake_report_data

    data = build_fake_report_data(username=USERNAME, outcome=OUTCOME)
else:
    data = load(
        username=USERNAME,
        outcomes=OUTCOME,
        source=SOURCE,
        candidates=candidates,
    )
arts = data.outcomes[OUTCOME]
```
```

(The outer triple-backtick fences above are markdown-doc fences for this plan; the file content is the two ```` ```{python} ```` chunks themselves.)

- [ ] **Step 2: Verify no frontmatter and chunk count**

Run: `head -1 reports/_setup.qmd && grep -c '```{python}' reports/_setup.qmd`
Expected: first line is ```` ```{python} ```` (not `---`), and the count is `2`.

- [ ] **Step 3: Commit**

```bash
git add reports/_setup.qmd
git commit -m "refactor(reports): extract shared _setup.qmd include"
```

---

### Task 2: Extract shared Collection include

**Files:**
- Create: `reports/_collection.qmd`
- Reference: `reports/collection_report.qmd:130-197` (the `# Collection` heading through the end of the collection-table chunk)

- [ ] **Step 1: Create `reports/_collection.qmd`**

Copy the `# Collection` section verbatim — from the `# Collection` heading line through the closing ```` ``` ```` of the `#| label: collection-table` chunk. This is exactly the block currently in `collection_report.qmd` starting at `# Collection` and ending after the `allow_html=True,\n)` of `collection-table`. Content:

```
# Collection

The data in this project comes from BoardGameGeek. The unit of
observation is a game. We train a classification model at the user
level to learn the relationship between game features and games a user
owns — what predicts a user's collection?

## Types of Games

::: {.column-margin}
Counts of feature values across owned games. Publishers are intentionally
excluded — for many users the top entries are foreign-language
re-publishers of already-popular games.
:::

What kinds of games does the user own? The plot below shows the most
frequent designers, publishers, mechanics, categories, and so on that
appear in their collection.

```{python}
#| label: types-of-games
#| fig-width: 8
#| fig-height: 9
#| out-width: 100%
plot_collection_by_category_static(data.collection, data.games)
```

The histogram below shows the years in which games in the user's
collection were published. This often hints at when someone first
entered the hobby.

```{python}
#| label: collection-by-year
#| fig-width: 8
#| fig-height: 3
#| out-width: 100%
plot_collection_by_year_static(data.collection, data.games)
```

## `{python} f"{USERNAME}'s Collection"`

::: {.column-margin}
TODO: notes about the user's collection — what stands out about their
taste, gaps, anything worth calling out before showing the full table.
:::

Games the user has tagged on BoardGameGeek — owned, on the wishlist,
want-to-buy, preordered, or previously owned. Sortable by status,
rating, players, and playtime.

```{python}
#| label: collection-table
itables_show(
    format_collection_table(data.collection, data.games),
    paging=True,
    pageLength=15,
    classes="display compact",
    # Column order: Game | Status | Your rating | Players | Playtime | Complexity
    columnDefs=[
        {"className": "dt-center", "targets": [1, 2, 3, 4, 5]},
        {"width": "100px", "targets": [1]},
        {"width": "90px", "targets": [2]},
        {"width": "80px", "targets": [3]},
        {"width": "110px", "targets": [4]},
        {"width": "100px", "targets": [5]},
    ],
    style="width: 100%",
    allow_html=True,
)
```
```

(Again, the outermost fence is the plan-doc fence; the file is the markdown + the inner ```` ```{python} ```` chunks.)

- [ ] **Step 2: Verify**

Run: `head -1 reports/_collection.qmd && grep -c '```{python}' reports/_collection.qmd`
Expected: first line is `# Collection`; chunk count is `3`.

- [ ] **Step 3: Commit**

```bash
git add reports/_collection.qmd
git commit -m "refactor(reports): extract shared _collection.qmd include"
```

---

### Task 3: Create predictions_report.qmd

**Files:**
- Create: `reports/predictions_report.qmd`
- Reference: `reports/collection_report.qmd:1-30` (frontmatter), `:116-128` (About), `:589-678` (Predictions)

- [ ] **Step 1: Create `reports/predictions_report.qmd`**

Frontmatter (copy verbatim from `collection_report.qmd:1-30`), then a predictions-flavored About, the shared includes, and the Predictions section (verbatim from the current `# Predictions {#predictions}` heading through the end of the `predictions-older` chunk). Full file:

````
---
title: "Predicting Board Game Collections"
author: "Phil Henrickson"
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
    mainfont: "Roboto, 'Helvetica Neue', Arial, sans-serif"
    fig-align: center
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

# About

This report shows what a classification model predicts about
`{python} USERNAME`'s board game collection — which new and upcoming
releases the model thinks they'd own, and which older games closely
match their taste. The model learns only from features observable at
release (publisher, mechanics, designers, playing time) and never reads
BGG community signals like average rating or weight, so it can score
games before the community has weighed in.

For the model's findings and evaluation, see the companion **Model
Report**.

{{< include _collection.qmd >}}

# Predictions {#predictions}

## New and Upcoming Games

::: {.column-margin}
TODO: notes about specific upcoming games worth flagging — anything
the model is unusually high on, anything that surprised you.
:::

Predictions for recent and upcoming releases, generated by the deployed
model. Filtered to games published *after* the model's training window
(`finalize_through`). The **Status** column shows whether the game is
already in the user's collection.

```{python}
#| label: predictions-upcoming
#| column: page-right
status_lookup = build_status_lookup(data.collection)

# The model was trained on data through `finalize_through`, so anything
# published in or before that year is in-sample. "New and Upcoming"
# means strictly after.
finalize_through = arts.registration.get("finalize_through")
upcoming = arts.upcoming_predictions.join(data.games, on="game_id", how="inner")
if finalize_through is not None:
    upcoming = upcoming.filter(pl.col("year_published") > int(finalize_through))
upcoming = upcoming.sort("predicted_prob", descending=True)

itables_show(
    format_predictions_with_images(
        upcoming,
        status_lookup=status_lookup,
        top_n=100,
        show_actual=False,
    ),
    paging=True,
    pageLength=10,
    classes="display compact",
    columnDefs=[
        {"className": "dt-center", "targets": [0, 4, 5]},
        {"width": "60px", "targets": [0]},
        {"width": "90px", "targets": [1]},
        {"width": "200px", "targets": [2]},
        {"width": "80px", "targets": [4]},
        {"width": "100px", "targets": [5]},
    ],
    style="width: 100%",
    allow_html=True,
)
```

## Older Games

Older games that the model liked for the user during training/evaluation. These are the highest scoring games that the user does not currently own; these might be good ones to check out.

::: {.column-margin}
These are technically 'misses' by the model, but in practice they represent games that are very consistent with what is in the user's collection and can be thought of as recommendations.
:::

```{python}
#| label: predictions-older
older = (
    pl.concat(
        [arts.oof_predictions, arts.val_predictions, arts.test_predictions],
        how="diagonal_relaxed",
    ).join(data.games, on="game_id", how="inner")
    # `label` is a boolean (or 0/1) here — true = user owns it. We want
    # not-owned games for the recommendations list, so cast and negate.
    .filter(~pl.col("label").cast(pl.Boolean))
    # Drop the label column so the table doesn't render an "Owned"
    # column of all-"no" values.
    .drop("label")
)
if "users_rated" in older.columns:
    older = older.filter(pl.col("users_rated") >= 5)
older = older.sort("proba", descending=True)

itables_show(
    format_eval_predictions(older, top_n=200),
    paging=True,
    pageLength=15,
    classes="display compact",
    columnDefs=[
        {"className": "dt-center", "targets": [0, 2]},
        {"width": "60px", "targets": [0]},
    ],
    style="width: 100%",
    allow_html=True,
)
```
````

- [ ] **Step 2: Verify it renders against fixtures**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run python -m reports.render --report predictions --fixture --output-dir /tmp/twosplit`
Expected: command exits 0 (this depends on Task 4's `--report` arg; if running tasks in order, defer this verification step's *execution* to after Task 4 and just confirm the file parses with `quarto check` is not required here). If `--report` is not yet implemented, instead run the legacy path to confirm the qmd itself is valid:
`cd reports && quarto render predictions_report.qmd -P fixture=true --output /tmp/p.html && test -s /tmp/p.html && echo OK`
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add reports/predictions_report.qmd
git commit -m "feat(reports): add predictions_report.qmd"
```

---

### Task 4: Create model_report.qmd

**Files:**
- Create: `reports/model_report.qmd`
- Reference: `reports/collection_report.qmd:1-30` (frontmatter), `:116-128` (About), `:201-468` (Modeling), `:470-587` (Assessment), `:680-704` (Model Details)

- [ ] **Step 1: Create `reports/model_report.qmd`**

Same frontmatter as Task 3 (verbatim from `collection_report.qmd:1-30`), then `{{< include _setup.qmd >}}`, a model-flavored About, `{{< include _collection.qmd >}}`, then the **Modeling**, **Assessment**, and **Model Details** sections copied verbatim from `collection_report.qmd` (the `# Modeling` heading through the end of the `model-details` chunk — i.e. current lines 201–468 for Modeling, 470–587 for Assessment, 680–704 for Model Details, concatenated in that order, skipping the Predictions section entirely).

The file is: the frontmatter block, then:

```
{{< include _setup.qmd >}}

# About

This report walks through the classification model trained to predict
whether `{python} USERNAME` owns a game on BoardGameGeek — what the
model learned, how the features relate to ownership, and how well it
performs on held-out data. The model uses only features observable at
release and never reads BGG community signals.

For the model's picks for new and upcoming games, see the companion
**Predictions Report**.

{{< include _collection.qmd >}}
```

…immediately followed by the verbatim copy of these three blocks from the current `collection_report.qmd`, in this exact order, with no other edits:

1. The entire `# Modeling` section: from the `# Modeling` heading line through the closing `:::` that ends the `.panel-tabset` (the `#| label: feature-importance-missigness` chunk and its closing `:::`). Current lines 201–468.
2. The entire `# Assessment` section: from `# Assessment` through the end of the `top-games-by-year` chunk. Current lines 470–587.
3. The entire `# Model Details` section: from `# Model Details` through the end of the `model-details` chunk. Current lines 680–704.

Do not alter any chunk labels, prose, or code inside these blocks.

- [ ] **Step 2: Verify the qmd is valid**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models/reports && quarto render model_report.qmd -P fixture=true --output /tmp/m.html && test -s /tmp/m.html && echo OK`
Expected: `OK`.

- [ ] **Step 3: Confirm no Predictions section leaked in**

Run: `grep -c 'predictions-upcoming\|# Predictions' reports/model_report.qmd`
Expected: `0`.

- [ ] **Step 4: Commit**

```bash
git add reports/model_report.qmd
git commit -m "feat(reports): add model_report.qmd"
```

---

### Task 5: Add --report to render.py

**Files:**
- Modify: `reports/render.py`
- Test: `tests/reports/test_render_smoke.py`

- [ ] **Step 1: Write the failing test (parametrized smoke test)**

Replace the body of `tests/reports/test_render_smoke.py` with:

```python
"""End-to-end smoke test for reports/render.py.

Skipped if Quarto is not on PATH. Runs the render driver against the
fixture artifact tree and asserts the right HTML output path per report.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("quarto") is None, reason="Quarto not installed on PATH"
)
@pytest.mark.parametrize(
    "report,rel_out",
    [
        ("predictions", "phenrickson.html"),
        ("model", "model/phenrickson.html"),
    ],
)
def test_render_smoke(fixture_collection_root: Path, tmp_path: Path, report, rel_out):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cmd = [
        "uv", "run", "python", "-m", "reports.render",
        "--report", report,
        "--username", "phenrickson",
        "--outcome", "own",
        "--source", str(fixture_collection_root),
        "--output-dir", str(output_dir),
        "--candidate", "logistic_row_norm",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, "BGG_REPORTS_OFFLINE": "1"},
    )
    assert result.returncode == 0, f"render failed: {result.stderr}"
    out_html = output_dir / rel_out
    assert out_html.exists(), f"missing {out_html}"
    assert out_html.stat().st_size > 1000
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run pytest tests/reports/test_render_smoke.py -v`
Expected: FAIL — `render.py` does not yet accept `--report` (argparse error / non-zero exit).

- [ ] **Step 3: Implement `--report` in `render.py`**

In `reports/render.py`, add the argument and template/path mapping. Add this constant near the top (after the logger definition):

```python
# Maps --report to (qmd template filename, output path builder). The
# predictions report keeps the friendly top-level URL; the model report
# lives under model/ so the two render pipelines never collide on a
# filename in the shared output bundle.
_REPORTS = {
    "predictions": "predictions_report.qmd",
    "model": "model_report.qmd",
}


def _output_rel_path(report: str, username: str) -> Path:
    if report == "model":
        return Path("model") / f"{username}.html"
    return Path(f"{username}.html")
```

In `main`, add the argument (after the `--candidate` line):

```python
    parser.add_argument(
        "--report",
        required=True,
        choices=sorted(_REPORTS),
        help="Which report to render: predictions or model",
    )
```

Change `_render_one`'s signature to accept `report: str`, and inside it replace the hardcoded `qmd_path` and `rendered`/`target` logic:

Replace:
```python
    qmd_path = Path(__file__).parent / "collection_report.qmd"
```
with:
```python
    qmd_path = Path(__file__).parent / _REPORTS[report]
```

Replace:
```python
    rendered_name = f"{username}.html"
```
with:
```python
    rendered_name = f"{username}.html"
    rel_out = _output_rel_path(report, username)
```

Replace the final block:
```python
    rendered = qmd_path.parent / rendered_name
    if not rendered.exists():
        logger.error("Quarto reported success but %s is missing", rendered)
        return 1
    target = output_dir / rendered_name
    rendered.replace(target)
    return 0
```
with:
```python
    rendered = qmd_path.parent / rendered_name
    if not rendered.exists():
        logger.error("Quarto reported success but %s is missing", rendered)
        return 1
    target = output_dir / rel_out
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered.replace(target)
    return 0
```

Update the `_render_one(...)` call site in `main` to pass `report=args.report`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run pytest tests/reports/test_render_smoke.py -v`
Expected: PASS for both `predictions` and `model` params (or SKIPPED if Quarto absent — if skipped, run the manual check below).

Manual fallback if skipped: `uv run python -m reports.render --report model --fixture --username fixture_user --output-dir /tmp/twosplit && test -s /tmp/twosplit/model/fixture_user.html && echo OK` → expect `OK`.

- [ ] **Step 5: Commit**

```bash
git add reports/render.py tests/reports/test_render_smoke.py
git commit -m "feat(reports): add required --report arg to render.py"
```

---

### Task 6: Extract index discovery into a testable function

**Files:**
- Modify: `reports/build_index.py`
- Test: `tests/reports/test_build_index.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/reports/test_build_index.py`:

```python
"""Unit tests for reports.build_index.discover_output_reports."""

from __future__ import annotations

from pathlib import Path

from reports.build_index import discover_output_reports


def test_detects_both_links(tmp_path: Path):
    (tmp_path / "alice.html").write_text("x")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "alice.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert rows == [
        {"username": "alice", "has_predictions": True, "has_model": True}
    ]


def test_predictions_only_degrades(tmp_path: Path):
    (tmp_path / "bob.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert rows == [
        {"username": "bob", "has_predictions": True, "has_model": False}
    ]


def test_model_only(tmp_path: Path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "carol.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert rows == [
        {"username": "carol", "has_predictions": False, "has_model": True}
    ]


def test_index_html_is_not_a_user(tmp_path: Path):
    (tmp_path / "index.html").write_text("x")
    (tmp_path / "dave.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert [r["username"] for r in rows] == ["dave"]


def test_empty_dir(tmp_path: Path):
    assert discover_output_reports(tmp_path) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run pytest tests/reports/test_build_index.py -v`
Expected: FAIL with `ImportError: cannot import name 'discover_output_reports'`.

- [ ] **Step 3: Add `discover_output_reports` to `build_index.py`**

Add this pure function to `reports/build_index.py` (after the logger definition, before `main`):

```python
def discover_output_reports(output_dir: Path) -> list[dict]:
    """Scan a rendered output directory and report, per user, which of
    the two report HTMLs exist.

    Predictions reports are `{output_dir}/{user}.html`; model reports are
    `{output_dir}/model/{user}.html`. `index.html` is not a user. Returns
    one dict per user sorted by username with `has_predictions` /
    `has_model` flags so the index can degrade to a single link when one
    side has not been rendered yet.
    """
    output_dir = Path(output_dir)
    users: dict[str, dict] = {}

    for p in sorted(output_dir.glob("*.html")):
        if p.name == "index.html":
            continue
        users.setdefault(
            p.stem, {"username": p.stem, "has_predictions": False, "has_model": False}
        )["has_predictions"] = True

    model_dir = output_dir / "model"
    if model_dir.is_dir():
        for p in sorted(model_dir.glob("*.html")):
            users.setdefault(
                p.stem,
                {"username": p.stem, "has_predictions": False, "has_model": False},
            )["has_model"] = True

    return [users[u] for u in sorted(users)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run pytest tests/reports/test_build_index.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add reports/build_index.py tests/reports/test_build_index.py
git commit -m "feat(reports): add discover_output_reports() to build_index"
```

---

### Task 7: Wire two links + reserved Reports section into index.qmd

**Files:**
- Modify: `reports/index.qmd`
- Modify: `reports/build_index.py` (pass `--output-dir` through to the qmd as a param so the qmd can call `discover_output_reports`)

- [ ] **Step 1: Pass output dir to the index qmd**

In `reports/build_index.py`, in `main`, add an `output-dir` param to the Quarto command. Change the `cmd` list's `-P` block from:

```python
        "-P",
        f"source={args.source}",
    ]
```
to:
```python
        "-P",
        f"source={args.source}",
        "-P",
        f"output_dir={output_dir.resolve()}",
    ]
```

- [ ] **Step 2: Update `index.qmd` params + grid**

In `reports/index.qmd`, add `output_dir` to the frontmatter `params:` block:

```yaml
params:
  source: local
  output_dir: ""
```

And in the parameters cell (`#| tags: [parameters]`), add:

```python
output_dir = ""
```

Replace the `#| label: user-grid` chunk's card-rendering loop so each card links to whichever reports exist. Replace the existing `for u in users_summary:` print block with one that first overlays the on-disk discovery:

```python
#| label: user-grid
#| output: asis
from pathlib import Path as _P
from reports.build_index import discover_output_reports

_present = {
    r["username"]: r
    for r in (discover_output_reports(_P(output_dir)) if output_dir else [])
}


def _fmt(v, decimals=3):
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


if not users_summary:
    print("_No reports have been generated yet._")
else:
    print("```{=html}")
    print('<div class="user-grid">')
    for u in users_summary:
        name = u["username"]
        flags = _present.get(
            name, {"has_predictions": True, "has_model": False}
        )
        links = []
        if flags["has_predictions"]:
            links.append(f'<a href="{name}.html">Predictions</a>')
        if flags["has_model"]:
            links.append(f'<a href="model/{name}.html">Model</a>')
        links_html = " · ".join(links) if links else "—"
        print(
            f'<div class="user-card">'
            f'<div class="user-card-title">{name}</div>'
            f'<div class="user-card-sub">'
            f'{u["model_type"]} · ROC-AUC {_fmt(u["roc_auc"])}'
            f'</div>'
            f'<div class="user-card-links">{links_html}</div>'
            f'</div>'
        )
    print("</div>")
    print("```")
```

Then add a reserved (empty) Reports section at the end of `index.qmd`, after the `## About` block:

```
## Reports

::: {.column-margin}
Site-wide reports will appear here.
:::

_No site-wide reports yet._
```

- [ ] **Step 3: Verify the index renders with two links**

Run:
```bash
cd /Users/phenrickson/Documents/projects/bgg-predictive-models && \
  uv run python -m reports.render --report predictions --fixture --username fixture_user --output-dir /tmp/idx && \
  uv run python -m reports.render --report model --fixture --username fixture_user --output-dir /tmp/idx && \
  uv run python -m reports.build_index --source local --output-dir /tmp/idx && \
  grep -q 'model/fixture_user.html' /tmp/idx/index.html && grep -q 'No site-wide reports yet' /tmp/idx/index.html && echo OK
```
Expected: `OK`. (If Quarto is absent this whole step is N/A; rely on Task 6 unit tests for the discovery logic.)

- [ ] **Step 4: Commit**

```bash
git add reports/index.qmd reports/build_index.py
git commit -m "feat(reports): index links to predictions + model, reserve Reports section"
```

---

### Task 8: Predictions workflow → daily + --report predictions

**Files:**
- Modify: `.github/workflows/build-collection-reports.yml`

- [ ] **Step 1: Switch the cron to daily**

In `.github/workflows/build-collection-reports.yml`, replace:

```yaml
  schedule:
    # Sunday 09:00 UTC — after the weekly collection-scoring run.
    - cron: '0 9 * * 0'
```
with:
```yaml
  schedule:
    # Daily 09:00 UTC — predictions refresh after overnight scoring.
    - cron: '0 9 * * *'
```

- [ ] **Step 2: Pass `--report predictions` in the render-user step**

In the `render-user` job's `Render report` step, the `docker run` invocation currently ends with `--username`, `--source`, `--output-dir /out`. Add `--report predictions` as the first renderer argument. Change:

```bash
            us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/reports:${{ needs.discover.outputs.image_tag }} \
            --username "${{ matrix.user }}" \
            --source "gs://${{ env.GCP_PROJECT_ID }}/${{ needs.discover.outputs.env_name }}/collections" \
            --output-dir /out
```
to:
```bash
            us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/reports:${{ needs.discover.outputs.image_tag }} \
            --report predictions \
            --username "${{ matrix.user }}" \
            --source "gs://${{ env.GCP_PROJECT_ID }}/${{ needs.discover.outputs.env_name }}/collections" \
            --output-dir /out
```

- [ ] **Step 3: Pass `--output-dir` to the index render in the deploy job**

In the `deploy` job's `Render index page` step, the `docker run ... -m reports.build_index` invocation passes `--source` and `--output-dir /out`. It already passes `--output-dir /out`; confirm `build_index.py` forwards it (done in Task 7). No YAML change needed here beyond confirming the existing `--output-dir /out` argument is present. If it is not present in that step, add `--output-dir /out` to the `reports.build_index` args.

- [ ] **Step 4: Lint the workflow YAML**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/build-collection-reports.yml')); print('YAML OK')"`
Expected: `YAML OK`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/build-collection-reports.yml
git commit -m "feat(ci): predictions pipeline renders daily with --report predictions"
```

---

### Task 9: New build-model-reports.yml workflow

**Files:**
- Create: `.github/workflows/build-model-reports.yml`
- Reference: `.github/workflows/build-collection-reports.yml` (mirror its structure)

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/build-model-reports.yml` as a structural mirror of the predictions workflow: same `discover` → `render-user` → `deploy` jobs, same env/permissions/concurrency, but **no schedule** (dispatch only), an added `outcome` input, `--report model` in render-user, and the deploy job seeds/mirrors the same gs:// bundle (so it preserves the predictions namespace untouched).

```yaml
name: Build Model Reports

on:
  workflow_dispatch:
    inputs:
      users:
        description: 'Comma-separated usernames to render (blank = discover all)'
        required: false
        default: ''
      outcome:
        description: 'Outcome to render the model report for'
        required: false
        default: 'own'

env:
  GCP_PROJECT_ID: bgg-predictive-models

permissions:
  contents: read
  id-token: write
  pages: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  discover:
    runs-on: ubuntu-latest
    environment: ${{ github.ref == 'refs/heads/main' && 'PROD' || 'DEV' }}
    outputs:
      users: ${{ steps.list.outputs.users }}
      env_name: ${{ steps.env.outputs.env_name }}
      image_tag: ${{ steps.env.outputs.image_tag }}
    steps:
      - name: Set environment name + image tag
        id: env
        run: |
          if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
            echo "env_name=prod" >> $GITHUB_OUTPUT
            echo "image_tag=prod" >> $GITHUB_OUTPUT
          else
            echo "env_name=dev" >> $GITHUB_OUTPUT
            echo "image_tag=dev" >> $GITHUB_OUTPUT
          fi
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY_BGG_ML }}
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Build user list
        id: list
        run: |
          set -e
          requested='${{ inputs.users }}'
          if [ -n "$requested" ]; then
            users=$(echo "$requested" \
              | tr ',' '\n' | sed 's/^ *//; s/ *$//' | grep -v '^$' \
              | python3 -c "import sys, json; print(json.dumps([u.strip() for u in sys.stdin.read().split() if u.strip()]))")
          else
            users=$(gsutil ls "gs://${{ env.GCP_PROJECT_ID }}/${{ steps.env.outputs.env_name }}/collections/" 2>/dev/null \
              | sed -e 's|/$||' \
              | xargs -n1 basename \
              | python3 -c "import sys, json; print(json.dumps([u for u in sys.stdin.read().split() if u]))")
          fi
          echo "users=$users" >> $GITHUB_OUTPUT
          echo "Resolved users: $users"

  render-user:
    needs: discover
    runs-on: ubuntu-latest
    environment: ${{ github.ref == 'refs/heads/main' && 'PROD' || 'DEV' }}
    if: ${{ needs.discover.outputs.users != '[]' }}
    strategy:
      fail-fast: false
      max-parallel: 10
      matrix:
        user: ${{ fromJson(needs.discover.outputs.users) }}
    steps:
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY_BGG_ML }}
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Configure Docker
        run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      - name: Pull reports image
        run: |
          docker pull us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/reports:${{ needs.discover.outputs.image_tag }}
      - name: Render report
        run: |
          mkdir -p output/model
          docker run --rm \
            -v "$PWD/output:/out" \
            -v "$GOOGLE_APPLICATION_CREDENTIALS:/app/credentials/service-account-key.json:ro" \
            -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json \
            us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/reports:${{ needs.discover.outputs.image_tag }} \
            --report model \
            --username "${{ matrix.user }}" \
            --outcome "${{ inputs.outcome }}" \
            --source "gs://${{ env.GCP_PROJECT_ID }}/${{ needs.discover.outputs.env_name }}/collections" \
            --output-dir /out
      - name: Upload rendered HTML
        uses: actions/upload-artifact@v4
        with:
          name: model-report-${{ matrix.user }}
          path: output/model/${{ matrix.user }}.html
          if-no-files-found: error
          retention-days: 7

  deploy:
    needs: [discover, render-user]
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY_BGG_ML }}
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      - name: Configure Docker
        run: gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
      - name: Make output dir
        run: mkdir -p reports/_output/model
      - name: Restore previously published bundle from GCS
        run: |
          gsutil -m rsync -r \
            "gs://${{ env.GCP_PROJECT_ID }}/${{ needs.discover.outputs.env_name }}/reports/" \
            reports/_output/ 2>&1 || echo "(no prior bundle in gs://; first run)"
      - name: Download per-user artifacts
        uses: actions/download-artifact@v4
        with:
          path: artifacts
          pattern: model-report-*
          merge-multiple: true
      - name: Move per-user model HTMLs into output dir
        run: |
          if [ -d artifacts ]; then
            cp artifacts/*.html reports/_output/model/ 2>/dev/null || true
          fi
          ls -la reports/_output/ reports/_output/model/
      - name: Pull reports image
        run: |
          docker pull us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/reports:${{ needs.discover.outputs.image_tag }}
      - name: Render index page
        run: |
          docker run --rm \
            -v "$PWD/reports/_output:/out" \
            -v "$GOOGLE_APPLICATION_CREDENTIALS:/app/credentials/service-account-key.json:ro" \
            -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json \
            --entrypoint python \
            us-central1-docker.pkg.dev/${{ env.GCP_PROJECT_ID }}/bgg-predictive-models/reports:${{ needs.discover.outputs.image_tag }} \
            -m reports.build_index \
            --source "gs://${{ env.GCP_PROJECT_ID }}/${{ needs.discover.outputs.env_name }}/collections" \
            --output-dir /out
      - name: Mirror final bundle to GCS
        run: |
          gsutil -m rsync -r reports/_output/ \
            "gs://${{ env.GCP_PROJECT_ID }}/${{ needs.discover.outputs.env_name }}/reports/"
      - name: Configure GitHub Pages
        uses: actions/configure-pages@v5
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: reports/_output
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Lint the workflow YAML**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/build-model-reports.yml')); print('YAML OK')"`
Expected: `YAML OK`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/build-model-reports.yml
git commit -m "feat(ci): add build-model-reports workflow (dispatch-only, --report model)"
```

---

### Task 10: Promoter trigger seam + retire old template

**Files:**
- Modify: the collection promote recipe — find it first (Step 1)
- Delete: `reports/collection_report.qmd`

- [ ] **Step 1: Locate the promote entrypoint**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && grep -rn "promote" justfile Makefile 2>/dev/null | head && ls src/collection/promote.py 2>/dev/null`
Expected: identifies the `promote` recipe (justfile) and/or `src/collection/promote.py`. Read whichever is the user-facing promote entrypoint.

- [ ] **Step 2: Add the dispatch call to the promote recipe**

Append a model-report dispatch to the `promote` recipe so promoting a model kicks the render. In `justfile`, at the end of the `promote` recipe's command block, add (matching the recipe's existing variable names for user/outcome — they are `username` and `outcome` per the justfile-orchestration spec):

```bash
    @echo "Dispatching model report render for {{username}}/{{outcome}}"
    gh workflow run build-model-reports.yml -f users={{username}} -f outcome={{outcome}} || \
        echo "WARN: model-report dispatch failed (promote still succeeded); run it manually with: gh workflow run build-model-reports.yml -f users={{username}} -f outcome={{outcome}}"
```

If the user-facing promote path is `src/collection/promote.py` instead of (or in addition to) the justfile recipe, do not call `gh` from Python — keep the dispatch in the justfile recipe that wraps it, so the seam stays in one place. If there is no justfile `promote` recipe wrapping it, add the `gh workflow run` line as the final step of the recipe that is the user-facing promote entrypoint.

- [ ] **Step 3: Delete the obsolete template**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && grep -rn "collection_report.qmd" --include="*.py" --include="*.yml" --include="*.qmd" --include="justfile" --include="Makefile" . | grep -v docs/`
Expected: **no matches** (Task 5 repointed render.py; workflows now pass `--report`). If there are matches, fix those references before deleting. Then:

```bash
git rm reports/collection_report.qmd
```

- [ ] **Step 4: Full test suite**

Run: `cd /Users/phenrickson/Documents/projects/bgg-predictive-models && uv run pytest tests/reports/ -v`
Expected: all pass (render smoke params SKIPPED only if Quarto absent; `test_build_index.py` and `test_collection_data.py` PASS).

- [ ] **Step 5: Commit**

```bash
git add justfile reports/collection_report.qmd
git commit -m "feat(reports): promoter dispatches model-report render; drop collection_report.qmd"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task |
|---|---|
| Section split (predictions vs model) | Tasks 3, 4 |
| Shared logic factored, no duplication | Tasks 1, 2 (includes); 3, 4 consume them |
| `render.py` `--report` (required, no default) | Task 5 |
| Site layout: `/{user}.html`, `/model/{user}.html`, reserved `/reports/` | Tasks 5 (paths), 7 (reserved section) |
| Index typed sections + degrade-to-one-link | Tasks 6, 7 |
| Predictions pipeline = daily cron + `--report predictions` | Task 8 |
| New model pipeline = dispatch + `users`/`outcome` + `--report model` | Task 9 |
| Promoter triggers render (local + future job seam) | Task 10 |
| Both share the existing Docker image (no Dockerfile change) | Honored — no Dockerfile task; both qmds COPY'd via existing `reports/` copy |
| Tests: render smoke per type, build_index unit | Tasks 5, 6 |
| Out of scope items untouched | No tasks for training job / global report / workflow_run — correct |

All spec requirements map to a task.

**2. Placeholder scan:** No TBD/TODO-as-instruction; the only literal "TODO" strings are inside verbatim-copied prose blocks (`.column-margin` notes that already exist in the live template) — preserved intentionally, not plan placeholders. Every code step shows full code.

**3. Type consistency:** `discover_output_reports` returns `list[dict]` with keys `username` / `has_predictions` / `has_model` — same shape asserted in Task 6 tests and consumed in Task 7's `index.qmd`. `_REPORTS` keys (`predictions`, `model`) match `--report choices`, `_output_rel_path` branches, both workflows' `--report` args, and the smoke test params. `_output_rel_path` returns `model/{user}.html` ↔ Task 9 artifact path `output/model/${{ matrix.user }}.html` ↔ Task 6 `model/` glob. Consistent.

**Note for the executor:** Tasks 1–7 are verifiable locally (Quarto optional — unit tests cover the non-Quarto logic). Tasks 8–9 are YAML-lint-only locally; their CI behavior is validated by a real `workflow_dispatch` after merge. Task 3 Step 2 / Task 4 Step 2 depend on Task 5's `--report`; if executing strictly in order, use the documented `quarto render <file> -P fixture=true` fallback for those pre-Task-5 checks.
