# Design: Simulation Results Dashboard

## Context

The existing Experiments page (`src/streamlit/pages/2 Experiments.py`) is oriented around single-model experiments loaded from GCS. Each experiment tracks one model type (e.g., complexity, rating) with train/tune/test metrics, predictions, and feature importance.

The simulation-based evaluation workflow produces a different shape of data: a **run** that spans multiple models and test years, stored locally in `models/simulation/{run_name}/`. After running `make evaluate`, the run directory contains:

```
models/simulation/direct-2026-03-03/
  run_metadata.json          # run config, experiment lineage
  summary_metrics.csv        # one row per (test_year, outcome)
  predictions.parquet        # all years concatenated
  predictions_2021.parquet   # per-year files
  predictions_2022.parquet
  metrics_2021.json
  metrics_2022.json
  scatter_plots_2021.png
  top_games_forest_plot_2021.png
  ...
```

The predictions parquet contains columns for each outcome (complexity, rating, users_rated, geek_rating) with `_actual`, `_point`, `_median`, `_lower_90`, `_upper_90`, `_lower_50`, `_upper_50` suffixes, plus `game_id`, `name`, and `test_year`.

We need a new Streamlit page to review these results — both comparing runs against each other and drilling into individual run predictions.

## Approach

**New Streamlit page** (`src/streamlit/pages/5 Simulations.py`) alongside the existing Experiments page. Data loaded from local `models/simulation/` directory only (no GCS for now).

## Design

### Data Loading

- Scan `models/simulation/` for subdirectories containing `run_metadata.json`
- Load `run_metadata.json` for sidebar display (run name, timestamp, mode, years)
- Load `predictions.parquet` and `summary_metrics.csv` on demand
- Use `@st.cache_data` for caching

### Page Layout

**Sidebar:**
- List of discovered runs (sorted by timestamp, newest first)
- Run selection (single-select for detail view, multi-select for comparison)

**Two tabs:**

#### Tab 1: Compare Runs

Purpose: "Which run is best?" at a glance.

- **Metrics table** from `summary_metrics.csv` — rows are (run, test_year, outcome), columns are metrics
- **Metrics comparison chart** — user picks a metric from dropdown, sees it across runs faceted by outcome (same pattern as existing `create_performance_by_run_visualization`)
- **Run metadata summary** — expandable sections showing experiment lineage, mode, n_samples

Key metrics to surface: `rmse_point`, `rmse_sim`, `r2_point`, `r2_sim`, `coverage_90`, `interval_width_90`.

#### Tab 2: Run Detail

Purpose: Deep dive into one run's predictions.

Select a single run from sidebar.

- **Interactive filters** — users_rated percentile slider, year range slider, outcome selector. All metrics and plots below recompute when filters change (same pattern as existing Experiments page).
- **Metrics overview** — `st.metric` cards (RMSE, MAE, R2, coverage, interval width) computed on the fly from filtered predictions
- **Scatter plots** — predicted vs actual per outcome (plotly, interactive), with diagonal line and LOESS trend. Regenerated from data, not static PNGs.
- **Prediction intervals** — interactive forest plot for top N games showing 90%/50% intervals
- **Predictions table** — full game-level data with all columns, filterable/sortable

### Decisions

- **No static PNGs.** All plots are interactive plotly, regenerated from `predictions.parquet`. The saved PNGs from `evaluate_simulation.py` are for offline reference only.
- **Metrics computed on the fly.** The `summary_metrics.csv` is useful for quick comparison across runs, but the Run Detail tab computes metrics dynamically from filtered `predictions.parquet` data — same pattern as the existing Experiments page. Users can filter by `users_rated` percentile, year range, and outcome, and metrics update accordingly.
- **Data source is local only** (`models/simulation/`). No GCS for now.
