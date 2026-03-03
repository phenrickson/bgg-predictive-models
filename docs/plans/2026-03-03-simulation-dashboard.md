# Simulation Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** New Streamlit page for reviewing simulation evaluation runs — comparing runs side-by-side and drilling into individual run predictions with interactive filters and metrics computed on the fly.

**Architecture:** A single Streamlit page (`src/streamlit/pages/5 Simulations.py`) loads data from local `models/simulation/` directories. A helper module (`src/streamlit/components/simulation_loader.py`) handles discovery and loading. Metrics are computed on-the-fly from `predictions.parquet` using numpy, matching the formulas in `src/models/outcomes/simulation.py:compute_simulation_metrics`. All plots are interactive plotly.

**Tech Stack:** Streamlit, Polars (data loading), Plotly (charts), NumPy (metrics), statsmodels (LOESS, optional)

---

### Task 1: Simulation Loader

Create a helper module that discovers simulation runs and loads their data.

**Files:**
- Create: `src/streamlit/components/simulation_loader.py`

**Step 1: Create the loader module**

```python
"""Loader for simulation run results from local disk."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import polars as pl

logger = logging.getLogger(__name__)

SIMULATION_DIR = Path("models/simulation")
OUTCOMES = ["complexity", "rating", "users_rated", "geek_rating"]


def discover_runs(base_dir: Path = SIMULATION_DIR) -> List[Dict[str, Any]]:
    """Scan for simulation runs containing run_metadata.json.

    Returns list of dicts with keys: name, path, timestamp, metadata.
    Sorted by timestamp descending (newest first).
    """
    runs = []
    if not base_dir.exists():
        return runs

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            runs.append({
                "name": metadata.get("run_name", run_dir.name),
                "path": run_dir,
                "timestamp": metadata.get("timestamp", ""),
                "metadata": metadata,
            })
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {run_dir.name}: {e}")

    runs.sort(key=lambda r: r["timestamp"], reverse=True)
    return runs


def load_predictions(run_path: Path) -> Optional[pl.DataFrame]:
    """Load combined predictions.parquet for a run."""
    path = run_path / "predictions.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return None


def load_summary_metrics(run_path: Path) -> Optional[pl.DataFrame]:
    """Load summary_metrics.csv for a run."""
    path = run_path / "summary_metrics.csv"
    if path.exists():
        return pl.read_csv(path)
    return None
```

**Step 2: Commit**

```bash
git add src/streamlit/components/simulation_loader.py
git commit -m "feat: add simulation run loader for streamlit dashboard"
```

---

### Task 2: Metrics computation module

Create a helper that computes simulation metrics from a filtered predictions DataFrame, matching the formulas in `src/models/outcomes/simulation.py:compute_simulation_metrics` (lines 1138-1230). This operates on the flat parquet columns (`{outcome}_actual`, `{outcome}_point`, `{outcome}_median`, `{outcome}_lower_90`, etc.) rather than `SimulationResult` objects.

**Files:**
- Create: `src/streamlit/components/simulation_metrics.py`

**Step 1: Create the metrics module**

```python
"""Compute simulation metrics from predictions DataFrame.

Mirrors the formulas in src/models/outcomes/simulation.py but operates
on flat polars DataFrames (from predictions.parquet) instead of
SimulationResult objects.
"""

from typing import Dict, Any

import numpy as np
import polars as pl

OUTCOMES = ["complexity", "rating", "users_rated", "geek_rating"]


def compute_metrics_for_outcome(
    df: pl.DataFrame, outcome: str
) -> Dict[str, Any]:
    """Compute RMSE, MAE, R2, coverage, and interval width for one outcome.

    Args:
        df: Filtered predictions DataFrame.
        outcome: One of complexity, rating, users_rated, geek_rating.

    Returns:
        Dict with n, rmse_point, rmse_sim, mae_point, mae_sim,
        r2_point, r2_sim, coverage_90, coverage_50,
        interval_width_90, interval_width_50.
    """
    actual_col = f"{outcome}_actual"
    point_col = f"{outcome}_point"
    median_col = f"{outcome}_median"

    # Check columns exist
    required = [actual_col, point_col, median_col]
    if not all(c in df.columns for c in required):
        return {"n": 0}

    # Filter out invalid actuals:
    # complexity/rating/users_rated: skip 0 (means missing)
    # geek_rating: skip where users_rated_actual == 0
    if outcome in ("complexity", "rating", "users_rated"):
        df = df.filter(pl.col(actual_col) != 0)
    elif outcome == "geek_rating":
        if "users_rated_actual" in df.columns:
            df = df.filter(pl.col("users_rated_actual") != 0)

    # Also drop nulls
    df = df.drop_nulls(subset=[actual_col, point_col, median_col])

    n = len(df)
    if n == 0:
        return {"n": 0}

    actuals = df[actual_col].to_numpy()
    points = df[point_col].to_numpy()
    medians = df[median_col].to_numpy()

    # RMSE / MAE / R2 for simulation median
    rmse_sim = float(np.sqrt(np.mean((actuals - medians) ** 2)))
    mae_sim = float(np.mean(np.abs(actuals - medians)))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    ss_res_sim = float(np.sum((actuals - medians) ** 2))
    r2_sim = 1 - ss_res_sim / ss_tot if ss_tot > 0 else 0.0

    # RMSE / MAE / R2 for point estimate
    rmse_point = float(np.sqrt(np.mean((actuals - points) ** 2)))
    mae_point = float(np.mean(np.abs(actuals - points)))
    ss_res_point = float(np.sum((actuals - points) ** 2))
    r2_point = 1 - ss_res_point / ss_tot if ss_tot > 0 else 0.0

    result = {
        "n": n,
        "rmse_point": round(rmse_point, 4),
        "rmse_sim": round(rmse_sim, 4),
        "mae_point": round(mae_point, 4),
        "mae_sim": round(mae_sim, 4),
        "r2_point": round(r2_point, 4),
        "r2_sim": round(r2_sim, 4),
    }

    # Coverage and interval width for 90% and 50%
    for level in [90, 50]:
        lower_col = f"{outcome}_lower_{level}"
        upper_col = f"{outcome}_upper_{level}"
        if lower_col in df.columns and upper_col in df.columns:
            lowers = df[lower_col].to_numpy()
            uppers = df[upper_col].to_numpy()
            in_interval = (actuals >= lowers) & (actuals <= uppers)
            result[f"coverage_{level}"] = round(float(np.mean(in_interval)), 4)
            result[f"interval_width_{level}"] = round(
                float(np.mean(uppers - lowers)), 4
            )

    return result


def compute_all_metrics(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for all outcomes.

    Args:
        df: Filtered predictions DataFrame.

    Returns:
        Dict keyed by outcome name, values are metric dicts.
    """
    return {outcome: compute_metrics_for_outcome(df, outcome) for outcome in OUTCOMES}
```

**Step 2: Commit**

```bash
git add src/streamlit/components/simulation_metrics.py
git commit -m "feat: add simulation metrics computation for dashboard"
```

---

### Task 3: Simulations page — scaffolding and Compare Runs tab

Create the Streamlit page with sidebar run discovery and the Compare Runs tab.

**Files:**
- Create: `src/streamlit/pages/5 Simulations.py`

**Step 1: Create the page**

```python
"""
Streamlit page for reviewing simulation evaluation runs.

Compare runs side-by-side and drill into individual run predictions
with interactive filters and on-the-fly metrics.
"""

import sys
import os
import logging

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.streamlit.components.simulation_loader import (
    discover_runs,
    load_predictions,
    load_summary_metrics,
    OUTCOMES,
)
from src.streamlit.components.simulation_metrics import (
    compute_all_metrics,
    compute_metrics_for_outcome,
)
from src.streamlit.components.footer import render_footer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Simulations | BGG Models Dashboard", layout="wide")
st.title("Simulation Runs")

# --- Sidebar: discover and select runs ---

@st.cache_data(ttl=60)
def get_runs():
    return discover_runs()

runs = get_runs()

if not runs:
    st.warning("No simulation runs found in models/simulation/. Run `make evaluate` first.")
    st.stop()

run_names = [r["name"] for r in runs]

# --- Tabs ---
tab_compare, tab_detail = st.tabs(["Compare Runs", "Run Detail"])

# ============================
# Tab 1: Compare Runs
# ============================
with tab_compare:
    st.header("Compare Runs")

    selected_run_names = st.multiselect(
        "Select runs to compare",
        run_names,
        default=run_names,
    )

    if not selected_run_names:
        st.info("Select at least one run to compare.")
        st.stop()

    selected_runs = [r for r in runs if r["name"] in selected_run_names]

    # Load and combine summary metrics
    @st.cache_data
    def load_combined_summary(run_paths_and_names):
        frames = []
        for name, path_str in run_paths_and_names:
            from pathlib import Path
            df = load_summary_metrics(Path(path_str))
            if df is not None:
                df = df.with_columns(pl.lit(name).alias("run"))
                frames.append(df)
        if frames:
            return pl.concat(frames)
        return None

    run_paths_and_names = [(r["name"], str(r["path"])) for r in selected_runs]
    combined_metrics = load_combined_summary(run_paths_and_names)

    if combined_metrics is None:
        st.warning("No summary_metrics.csv found for selected runs.")
    else:
        # Metrics table
        st.subheader("Metrics Table")
        st.dataframe(combined_metrics.to_pandas(), use_container_width=True)

        # Metric comparison chart
        st.subheader("Metrics Comparison")
        numeric_cols = [
            c for c in combined_metrics.columns
            if c not in ("run", "test_year", "outcome")
        ]
        selected_metric = st.selectbox("Select metric", numeric_cols)

        if selected_metric:
            plot_df = combined_metrics.select(
                ["run", "test_year", "outcome", selected_metric]
            ).to_pandas()

            fig = px.bar(
                plot_df,
                x="run",
                y=selected_metric,
                color="outcome",
                barmode="group",
                facet_row="test_year",
                title=f"{selected_metric} by run",
                height=300 * plot_df["test_year"].nunique(),
            )
            fig.update_layout(xaxis_title="Run", yaxis_title=selected_metric)
            st.plotly_chart(fig, use_container_width=True)

    # Run metadata
    st.subheader("Run Metadata")
    for r in selected_runs:
        with st.expander(r["name"]):
            st.json(r["metadata"])
```

**Step 2: Commit**

```bash
git add "src/streamlit/pages/5 Simulations.py"
git commit -m "feat: add simulations page with compare runs tab"
```

---

### Task 4: Run Detail tab — filters and on-the-fly metrics

Add the Run Detail tab content to `5 Simulations.py`. This loads `predictions.parquet` for a single run, applies interactive filters, and computes metrics dynamically.

**Files:**
- Modify: `src/streamlit/pages/5 Simulations.py`

**Step 1: Add the Run Detail tab content**

Append to `5 Simulations.py`, inside the `with tab_detail:` block:

```python
# ============================
# Tab 2: Run Detail
# ============================
with tab_detail:
    st.header("Run Detail")

    selected_run_name = st.selectbox("Select run", run_names, key="detail_run")
    selected_run = next(r for r in runs if r["name"] == selected_run_name)

    @st.cache_data
    def load_run_predictions(path_str):
        from pathlib import Path
        return load_predictions(Path(path_str))

    predictions = load_run_predictions(str(selected_run["path"]))

    if predictions is None:
        st.warning("No predictions.parquet found for this run.")
        st.stop()

    # --- Filters ---
    st.subheader("Filters")
    filter_cols = st.columns(3)

    # Test year filter
    available_years = sorted(predictions["test_year"].unique().to_list())
    with filter_cols[0]:
        selected_years = st.multiselect(
            "Test years",
            available_years,
            default=available_years,
            key="detail_years",
        )

    # Outcome filter
    with filter_cols[1]:
        selected_outcomes = st.multiselect(
            "Outcomes",
            OUTCOMES,
            default=OUTCOMES,
            key="detail_outcomes",
        )

    # Users rated percentile filter
    with filter_cols[2]:
        min_users_pct = st.slider(
            "Min users rated (percentile)",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="detail_users_pct",
        )

    # Apply filters
    filtered = predictions.filter(pl.col("test_year").is_in(selected_years))

    if min_users_pct > 0 and "users_rated_actual" in filtered.columns:
        threshold = filtered["users_rated_actual"].quantile(min_users_pct / 100)
        filtered = filtered.filter(pl.col("users_rated_actual") >= threshold)

    st.caption(f"Showing {len(filtered):,} games after filters")

    # --- Metrics ---
    st.subheader("Metrics")
    metrics = compute_all_metrics(filtered)

    for outcome in selected_outcomes:
        m = metrics.get(outcome, {})
        if m.get("n", 0) == 0:
            continue

        st.markdown(f"**{outcome.replace('_', ' ').title()}** (n={m['n']:,})")
        cols = st.columns(6)
        cols[0].metric("RMSE (point)", f"{m.get('rmse_point', 0):.4f}")
        cols[1].metric("RMSE (sim)", f"{m.get('rmse_sim', 0):.4f}")
        cols[2].metric("R\u00b2 (point)", f"{m.get('r2_point', 0):.4f}")
        cols[3].metric("R\u00b2 (sim)", f"{m.get('r2_sim', 0):.4f}")
        cols[4].metric("Coverage 90%", f"{m.get('coverage_90', 0):.1%}")
        cols[5].metric("Interval Width 90%", f"{m.get('interval_width_90', 0):.3f}")
```

**Step 2: Commit**

```bash
git add "src/streamlit/pages/5 Simulations.py"
git commit -m "feat: add run detail tab with filters and on-the-fly metrics"
```

---

### Task 5: Run Detail tab — scatter plots

Add interactive plotly scatter plots (predicted vs actual) per outcome to the Run Detail tab.

**Files:**
- Modify: `src/streamlit/pages/5 Simulations.py`

**Step 1: Add scatter plots section**

Append after the metrics section inside the `with tab_detail:` block:

```python
    # --- Scatter plots ---
    st.subheader("Predicted vs Actual")

    # Try importing statsmodels for LOESS
    try:
        import statsmodels.nonparametric.smoothers_lowess as lowess_mod
        HAS_STATSMODELS = True
    except ImportError:
        HAS_STATSMODELS = False

    scatter_cols = st.columns(2)
    for i, outcome in enumerate(selected_outcomes):
        actual_col = f"{outcome}_actual"
        point_col = f"{outcome}_point"
        median_col = f"{outcome}_median"

        if not all(c in filtered.columns for c in [actual_col, point_col]):
            continue

        # Filter invalid actuals (same logic as metrics)
        plot_df = filtered
        if outcome in ("complexity", "rating", "users_rated"):
            plot_df = plot_df.filter(pl.col(actual_col) != 0)
        elif outcome == "geek_rating" and "users_rated_actual" in plot_df.columns:
            plot_df = plot_df.filter(pl.col("users_rated_actual") != 0)
        plot_df = plot_df.drop_nulls(subset=[actual_col, point_col])

        pdf = plot_df.select([actual_col, point_col, "name", "game_id"]).to_pandas()

        fig = go.Figure()

        # Scatter
        fig.add_trace(go.Scatter(
            x=pdf[point_col],
            y=pdf[actual_col],
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            text=pdf["name"],
            hovertemplate="<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
            showlegend=False,
        ))

        # Diagonal
        min_val = min(pdf[point_col].min(), pdf[actual_col].min())
        max_val = max(pdf[point_col].max(), pdf[actual_col].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", line=dict(color="red", dash="dash"),
            name="y=x", showlegend=False,
        ))

        # LOESS
        if HAS_STATSMODELS and len(pdf) > 10:
            try:
                sorted_idx = np.argsort(pdf[point_col].values)
                x_sorted = pdf[point_col].values[sorted_idx]
                y_sorted = pdf[actual_col].values[sorted_idx]
                smoothed = lowess_mod.lowess(y_sorted, x_sorted, frac=2/3, it=5)
                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0], y=smoothed[:, 1],
                    mode="lines", line=dict(color="green", width=2, dash="dot"),
                    name="LOESS", showlegend=False,
                ))
            except Exception:
                pass

        # Correlation annotation
        corr = pdf[point_col].corr(pdf[actual_col])
        label = "log users rated" if outcome == "users_rated" else outcome.replace("_", " ").title()
        fig.update_layout(
            title=f"{label} (r={corr:.3f})",
            xaxis_title="Predicted (Point)",
            yaxis_title="Actual",
            height=400,
        )

        with scatter_cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)
```

**Step 2: Commit**

```bash
git add "src/streamlit/pages/5 Simulations.py"
git commit -m "feat: add interactive scatter plots to simulation detail tab"
```

---

### Task 6: Run Detail tab — forest plot and predictions table

Add the interactive forest plot (top N games with prediction intervals) and the raw predictions table.

**Files:**
- Modify: `src/streamlit/pages/5 Simulations.py`

**Step 1: Add forest plot and predictions table**

Append after the scatter plots section inside the `with tab_detail:` block:

```python
    # --- Forest plot ---
    st.subheader("Top Games — Prediction Intervals")

    forest_outcome = st.selectbox(
        "Outcome for forest plot",
        selected_outcomes,
        key="forest_outcome",
    )
    top_n = st.slider("Number of games", 10, 200, 50, step=10, key="forest_top_n")

    actual_col = f"{forest_outcome}_actual"
    median_col = f"{forest_outcome}_median"
    lower_90 = f"{forest_outcome}_lower_90"
    upper_90 = f"{forest_outcome}_upper_90"
    lower_50 = f"{forest_outcome}_lower_50"
    upper_50 = f"{forest_outcome}_upper_50"

    forest_cols = [c for c in [actual_col, median_col, lower_90, upper_90, lower_50, upper_50, "name", "game_id"] if c in filtered.columns]

    if all(c in filtered.columns for c in [median_col, lower_90, upper_90]):
        forest_df = (
            filtered
            .select(forest_cols)
            .sort(median_col, descending=True)
            .head(top_n)
            .to_pandas()
        )
        forest_df = forest_df.iloc[::-1]  # reverse for bottom-to-top display

        fig = go.Figure()

        # 90% interval
        fig.add_trace(go.Scatter(
            x=forest_df[upper_90], y=forest_df["name"],
            mode="markers", marker=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forest_df[lower_90], y=forest_df["name"],
            mode="markers", marker=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        for _, row in forest_df.iterrows():
            fig.add_shape(
                type="line",
                x0=row[lower_90], x1=row[upper_90],
                y0=row["name"], y1=row["name"],
                line=dict(color="steelblue", width=1),
            )

        # 50% interval
        if lower_50 in forest_df.columns:
            for _, row in forest_df.iterrows():
                fig.add_shape(
                    type="line",
                    x0=row[lower_50], x1=row[upper_50],
                    y0=row["name"], y1=row["name"],
                    line=dict(color="steelblue", width=3),
                )

        # Median
        fig.add_trace(go.Scatter(
            x=forest_df[median_col], y=forest_df["name"],
            mode="markers",
            marker=dict(color="steelblue", size=6),
            name="Median",
            hovertemplate="<b>%{y}</b><br>Median: %{x:.3f}<extra></extra>",
        ))

        # Actual
        if actual_col in forest_df.columns:
            fig.add_trace(go.Scatter(
                x=forest_df[actual_col], y=forest_df["name"],
                mode="markers",
                marker=dict(color="red", size=6, symbol="circle-open"),
                name="Actual",
                hovertemplate="<b>%{y}</b><br>Actual: %{x:.3f}<extra></extra>",
            ))

        label = forest_outcome.replace("_", " ").title()
        fig.update_layout(
            title=f"Top {top_n} Games — {label}",
            xaxis_title=label,
            height=max(400, top_n * 18),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Predictions table ---
    st.subheader("Predictions")
    st.dataframe(filtered.to_pandas(), use_container_width=True)

# Footer
render_footer()
```

**Step 2: Commit**

```bash
git add "src/streamlit/pages/5 Simulations.py"
git commit -m "feat: add forest plot and predictions table to simulation detail"
```

---

### Task 7: Update Home.py and verify

Add the Simulations section to the Home page description and do a manual verification.

**Files:**
- Modify: `src/streamlit/Home.py`

**Step 1: Add Simulations to Home.py**

Add after the Experiments section in the markdown string:

```markdown
### Simulations
Review simulation evaluation runs:
- Compare runs side-by-side with summary metrics
- Drill into individual runs with interactive filters
- View scatter plots and prediction intervals
- Compute metrics on the fly with adjustable filters
```

**Step 2: Verify the page loads**

Run: `uv run streamlit run src/streamlit/Home.py`

Navigate to "Simulations" in the sidebar. Verify:
- If no runs exist in `models/simulation/`, shows a warning message
- If runs exist, sidebar shows run list, tabs work, metrics compute

**Step 3: Commit**

```bash
git add src/streamlit/Home.py
git commit -m "feat: add simulations section to dashboard home page"
```
