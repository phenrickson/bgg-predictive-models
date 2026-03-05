# Design: Local Experiments Page

## Context

The Experiments page (`src/streamlit/pages/2 Experiments.py`) loads experiment data from GCS via `ExperimentLoader`. This doesn't work locally and the app is only used for local development/analysis — not deployed.

All experiment data exists locally under `models/experiments/{outcome}/{experiment_name}/v1/` with the same structure: metadata.json, model_info.json, parameters.json, metrics JSONs, prediction parquets, coefficients.csv.

## Approach

Rewrite the Experiments page as local-only, following the pattern established by the Simulations page (`5 Simulations.py` + `components/simulation_loader.py`).

## Data Loading: `components/experiment_loader.py`

Scans `models/experiments/` for experiment directories. Each experiment is a dict:

- `outcome` — parent directory (rating, complexity, hurdle, users_rated, geek_rating)
- `name` — experiment directory name (eval-rating-2022, ard-ridge-rating)
- `version` — version subdirectory (v1, v2)
- `path` — full path to versioned directory
- `metadata` — parsed metadata.json
- `model_info` — parsed model_info.json
- `is_eval` — True if name starts with "eval-"
- `is_finalized` — True if `finalized/` subdirectory exists

Functions:

- `discover_experiments()` → list of experiment dicts (sorted by outcome, then name)
- `load_metrics(path, dataset)` → dict from `{dataset}_metrics.json`
- `load_predictions(path, dataset)` → polars DataFrame from `{dataset}_predictions.parquet`
- `load_coefficients(path)` → polars DataFrame from `coefficients.csv` (or None)

Skip the `predictions/` subdirectory under `models/experiments/` (not a model outcome).

## Page Layout: `2 Experiments.py`

Two tabs: **Eval Experiments** | **Finalized Models**

### Tab 1: Eval Experiments

- **Filters**: Outcome selector, dataset selector (train/tune/test)
- **Metrics table**: All eval experiments for selected outcome. Rows = experiments, columns = metrics (rmse, mae, r², mape). Test year shown from metadata.
- **Predictions scatter**: Select specific experiment, predicted vs actual with diagonal line + LOESS. Users_rated percentile filter when available.
- **Feature importance**: Expander per experiment with coefficients bar chart (horizontal, top N, RdBu color scale)
- **Metadata**: Expander with raw JSON

### Tab 2: Finalized Models

- **Outcome selector**
- **Model summary**: Metric cards from test_metrics.json, model info (n_features, best_params)
- **Coefficients plot**: Full coefficient bar chart
- **Metadata expander**

## Decisions

- **Local only** — no GCS, no `ExperimentLoader`. This is a development tool.
- **Polars throughout** — consistent with Simulations page.
- **Plotly for all charts** — interactive, no static PNGs.
- **Core features only** — metrics, scatter plots, feature importance. No verbose 6-tab layout. Metadata/details in expanders.
- **Same caching pattern** — `@st.cache_data` on all data loading functions.
