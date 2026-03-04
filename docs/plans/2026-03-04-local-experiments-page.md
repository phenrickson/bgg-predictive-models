# Local Experiments Page Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild the Experiments page to load from local `models/experiments/` instead of GCS, with a cleaner two-tab layout (Eval Experiments | Finalized Models).

**Architecture:** New `components/experiment_loader.py` scans local experiment directories and returns structured dicts. Rewritten `2 Experiments.py` uses polars + plotly, matching the Simulations page pattern. No GCS dependency.

**Tech Stack:** Streamlit, polars, plotly, json (stdlib)

---

### Task 1: Create `components/experiment_loader.py`

**Files:**
- Create: `src/streamlit/components/experiment_loader.py`

**Step 1: Write the loader module**

This module scans `models/experiments/` for experiment directories. Each outcome (rating, complexity, etc.) is a parent directory. Each experiment has a version subdirectory (v1, v2). Skip the `predictions/` directory (not a model outcome).

```python
"""Loader for experiment results from local disk."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import polars as pl

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("models/experiments")
SKIP_DIRS = {"predictions"}


def discover_experiments(base_dir: Path = EXPERIMENTS_DIR) -> List[Dict[str, Any]]:
    """Scan for experiments containing metadata.json.

    Returns list of dicts with keys: outcome, name, version, path, metadata,
    model_info, is_eval, is_finalized.
    Sorted by outcome then name.
    """
    experiments = []
    if not base_dir.exists():
        return experiments

    for outcome_dir in sorted(base_dir.iterdir()):
        if not outcome_dir.is_dir() or outcome_dir.name in SKIP_DIRS:
            continue

        for exp_dir in sorted(outcome_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            for version_dir in sorted(exp_dir.iterdir()):
                if not version_dir.is_dir():
                    continue

                metadata_path = version_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    model_info = None
                    model_info_path = version_dir / "model_info.json"
                    if model_info_path.exists():
                        with open(model_info_path) as f:
                            model_info = json.load(f)

                    experiments.append({
                        "outcome": outcome_dir.name,
                        "name": exp_dir.name,
                        "version": version_dir.name,
                        "path": version_dir,
                        "metadata": metadata,
                        "model_info": model_info,
                        "is_eval": exp_dir.name.startswith("eval-"),
                        "is_finalized": (version_dir / "finalized").is_dir(),
                    })
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Skipping {version_dir}: {e}")

    return experiments


def load_metrics(exp_path: Path, dataset: str = "test") -> Optional[Dict[str, Any]]:
    """Load metrics JSON for a dataset split."""
    path = exp_path / f"{dataset}_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_predictions(exp_path: Path, dataset: str = "test") -> Optional[pl.DataFrame]:
    """Load predictions parquet for a dataset split.

    Returns DataFrame with only the essential columns: game_id, name,
    year_published, users_rated, prediction, actual, plus classification
    probability columns if present.
    """
    path = exp_path / f"{dataset}_predictions.parquet"
    if not path.exists():
        return None

    df = pl.read_parquet(path)

    # Select only useful columns for display
    keep_cols = []
    for col in ["game_id", "name", "year_published", "users_rated",
                 "prediction", "actual",
                 "predicted_proba_class_0", "predicted_proba_class_1"]:
        if col in df.columns:
            keep_cols.append(col)

    return df.select(keep_cols) if keep_cols else df


def load_coefficients(exp_path: Path) -> Optional[pl.DataFrame]:
    """Load coefficients.csv if it exists."""
    path = exp_path / "coefficients.csv"
    if path.exists():
        return pl.read_csv(path)
    return None
```

**Step 2: Verify the loader works**

Run:
```bash
uv run python -c "
from src.streamlit.components.experiment_loader import discover_experiments
exps = discover_experiments()
for e in exps:
    print(f'{e[\"outcome\"]:15s} {e[\"name\"]:30s} {e[\"version\"]:4s} eval={e[\"is_eval\"]} final={e[\"is_finalized\"]}')
"
```

Expected: list of all experiments across all outcomes.

**Step 3: Commit**

```bash
git add src/streamlit/components/experiment_loader.py
git commit -m "feat: add local experiment loader component"
```

---

### Task 2: Rewrite `2 Experiments.py` — scaffolding and Eval tab metrics

**Files:**
- Modify: `src/streamlit/pages/2 Experiments.py`

**Step 1: Replace the entire file with the new scaffolding**

```python
"""
Streamlit page for exploring local model experiments.

Two tabs: Eval Experiments (time-based evaluation) and Finalized Models.
"""

import sys
import os
import logging

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import polars as pl
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.streamlit.components.experiment_loader import (
    discover_experiments,
    load_metrics,
    load_predictions,
    load_coefficients,
)
from src.streamlit.components.footer import render_footer

# Optional LOESS
try:
    import statsmodels.nonparametric.smoothers_lowess as lowess_mod
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Experiments | BGG Models Dashboard", layout="wide")
st.title("Experiments")


@st.cache_data(ttl=60)
def get_experiments():
    return discover_experiments()


experiments = get_experiments()

if not experiments:
    st.warning("No experiments found in models/experiments/. Run training first.")
    st.stop()

# Split into eval and finalized
eval_exps = [e for e in experiments if e["is_eval"]]
finalized_exps = [e for e in experiments if e["is_finalized"]]
outcomes = sorted(set(e["outcome"] for e in experiments))

# --- Tabs ---
tab_eval, tab_finalized = st.tabs(["Eval Experiments", "Finalized Models"])

# ============================
# Tab 1: Eval Experiments
# ============================
with tab_eval:
    st.header("Eval Experiments")

    if not eval_exps:
        st.info("No eval experiments found.")
    else:
        eval_outcomes = sorted(set(e["outcome"] for e in eval_exps))
        filter_cols = st.columns(2)
        with filter_cols[0]:
            selected_outcome = st.selectbox("Outcome", eval_outcomes, key="eval_outcome")
        with filter_cols[1]:
            selected_dataset = st.selectbox("Dataset", ["test", "tune", "train"], key="eval_dataset")

        outcome_exps = [e for e in eval_exps if e["outcome"] == selected_outcome]

        # --- Metrics Table ---
        st.subheader("Metrics")
        rows = []
        for exp in outcome_exps:
            m = load_metrics(exp["path"], selected_dataset)
            if m:
                row = {"experiment": exp["name"], "version": exp["version"]}
                meta = exp["metadata"].get("metadata", {})
                row["test_year"] = meta.get("test_through", "")
                row.update(m)
                rows.append(row)

        if rows:
            metrics_df = pl.DataFrame(rows)
            st.dataframe(metrics_df.to_pandas(), use_container_width=True)
        else:
            st.info(f"No {selected_dataset} metrics found for {selected_outcome} eval experiments.")

        # --- Predictions Scatter ---
        st.subheader("Predictions")
        exp_names = [e["name"] for e in outcome_exps]
        selected_exp_name = st.selectbox("Experiment", exp_names, key="eval_exp_select")
        selected_exp = next(e for e in outcome_exps if e["name"] == selected_exp_name)

        @st.cache_data
        def load_preds_cached(path_str, dataset):
            from pathlib import Path
            return load_predictions(Path(path_str), dataset)

        preds = load_preds_cached(str(selected_exp["path"]), selected_dataset)

        if preds is not None and "prediction" in preds.columns and "actual" in preds.columns:
            # Users rated percentile filter
            if "users_rated" in preds.columns:
                min_pct = st.slider("Min users rated (percentile)", 0, 100, 0, 5, key="eval_pct")
                if min_pct > 0:
                    threshold = preds["users_rated"].quantile(min_pct / 100)
                    preds = preds.filter(pl.col("users_rated") >= threshold)

            st.caption(f"Showing {len(preds):,} predictions")

            model_task = selected_exp["metadata"].get("metadata", {}).get("model_task", "regression")

            if model_task == "regression":
                pdf = preds.select(["prediction", "actual"] +
                    ([c for c in ["name", "game_id"] if c in preds.columns])
                ).to_pandas()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pdf["prediction"], y=pdf["actual"],
                    mode="markers", marker=dict(size=4, opacity=0.5),
                    text=pdf.get("name"),
                    hovertemplate="<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
                    showlegend=False,
                ))

                min_val = min(pdf["prediction"].min(), pdf["actual"].min())
                max_val = max(pdf["prediction"].max(), pdf["actual"].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode="lines", line=dict(color="red", dash="dash"),
                    showlegend=False,
                ))

                if HAS_STATSMODELS and len(pdf) > 10:
                    try:
                        sorted_idx = np.argsort(pdf["prediction"].values)
                        smoothed = lowess_mod.lowess(
                            pdf["actual"].values[sorted_idx],
                            pdf["prediction"].values[sorted_idx],
                            frac=2/3, it=5
                        )
                        fig.add_trace(go.Scatter(
                            x=smoothed[:, 0], y=smoothed[:, 1],
                            mode="lines", line=dict(color="green", width=2, dash="dot"),
                            name="LOESS", showlegend=False,
                        ))
                    except Exception:
                        pass

                corr = pdf["prediction"].corr(pdf["actual"])
                fig.update_layout(
                    title=f"{selected_outcome.replace('_', ' ').title()} — {selected_exp_name} (r={corr:.3f})",
                    xaxis_title="Predicted", yaxis_title="Actual", height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

            elif model_task == "classification":
                pdf = preds.to_pandas()
                # Metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{accuracy_score(pdf['actual'], pdf['prediction']):.2%}")
                cols[1].metric("Precision", f"{precision_score(pdf['actual'], pdf['prediction']):.2%}")
                cols[2].metric("Recall", f"{recall_score(pdf['actual'], pdf['prediction']):.2%}")
                cols[3].metric("F1", f"{f1_score(pdf['actual'], pdf['prediction']):.2%}")

                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(pdf["actual"], pdf["prediction"])
                fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Negative", "Positive"], y=["Negative", "Positive"],
                    title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)

            # Raw predictions table
            with st.expander("Raw Predictions"):
                st.dataframe(preds.to_pandas(), use_container_width=True)
        elif preds is not None:
            st.info(f"Predictions loaded but missing prediction/actual columns.")
        else:
            st.info(f"No {selected_dataset} predictions found.")

        # --- Feature Importance ---
        st.subheader("Feature Importance")
        for exp in outcome_exps:
            @st.cache_data
            def load_coeff_cached(path_str):
                from pathlib import Path
                return load_coefficients(Path(path_str))

            coeffs = load_coeff_cached(str(exp["path"]))
            if coeffs is not None:
                with st.expander(f"{exp['name']} — Coefficients"):
                    top_n = st.slider("Top N", 10, 100, 40, 5, key=f"coeff_n_{exp['name']}")

                    pdf = coeffs.to_pandas()
                    # Determine importance column
                    if "coefficient" in pdf.columns:
                        pdf["abs_val"] = pdf["coefficient"].abs()
                        plot_df = pdf.nlargest(top_n, "abs_val").sort_values("coefficient")
                        fig = px.bar(plot_df, y="feature", x="coefficient", orientation="h",
                            color="coefficient", color_continuous_scale="RdBu",
                            color_continuous_midpoint=0,
                            title=f"Top {top_n} Coefficients")
                        fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    elif "abs_coefficient" in pdf.columns:
                        plot_df = pdf.nlargest(top_n, "abs_coefficient").sort_values("abs_coefficient")
                        fig = px.bar(plot_df, y="feature", x="abs_coefficient", orientation="h",
                            color="abs_coefficient", color_continuous_scale="Viridis",
                            title=f"Top {top_n} Features")
                    else:
                        st.write("Unknown coefficient format:", pdf.columns.tolist())
                        continue

                    fig.update_layout(height=max(400, top_n * 20), yaxis_title="", xaxis_title="Effect")
                    st.plotly_chart(fig, use_container_width=True)

        # --- Metadata ---
        st.subheader("Metadata")
        for exp in outcome_exps:
            with st.expander(exp["name"]):
                st.json(exp["metadata"])
                if exp["model_info"]:
                    st.json(exp["model_info"])

# ============================
# Tab 2: Finalized Models
# ============================
with tab_finalized:
    st.header("Finalized Models")

    if not finalized_exps:
        st.info("No finalized models found.")
    else:
        final_outcomes = sorted(set(e["outcome"] for e in finalized_exps))
        selected_final_outcome = st.selectbox("Outcome", final_outcomes, key="final_outcome")

        outcome_finals = [e for e in finalized_exps if e["outcome"] == selected_final_outcome]

        for exp in outcome_finals:
            st.subheader(f"{exp['name']} ({exp['version']})")

            # Metrics
            m = load_metrics(exp["path"], "test")
            if m:
                model_task = exp["metadata"].get("metadata", {}).get("model_task", "regression")
                if model_task == "regression":
                    cols = st.columns(4)
                    cols[0].metric("RMSE", f"{m.get('rmse', 0):.4f}")
                    cols[1].metric("MAE", f"{m.get('mae', 0):.4f}")
                    cols[2].metric("R²", f"{m.get('r2', 0):.4f}")
                    cols[3].metric("MAPE", f"{m.get('mape', 0):.4f}")
                elif model_task == "classification":
                    cols = st.columns(4)
                    cols[0].metric("Accuracy", f"{m.get('accuracy', 0):.2%}")
                    cols[1].metric("F1", f"{m.get('f1', 0):.2%}")
                    cols[2].metric("AUC", f"{m.get('auc', 0):.2%}")
                    cols[3].metric("Log Loss", f"{m.get('log_loss', 0):.4f}")

            # Model info
            if exp["model_info"]:
                meta = exp["metadata"].get("metadata", {})
                info_cols = st.columns(3)
                info_cols[0].metric("Features", meta.get("feature_count", "?"))
                info_cols[1].metric("Train Size", meta.get("data_sizes", {}).get("train", "?"))
                info_cols[2].metric("Algorithm", meta.get("algorithm", "?"))

            # Coefficients
            @st.cache_data
            def load_final_coeff(path_str):
                from pathlib import Path
                return load_coefficients(Path(path_str))

            coeffs = load_final_coeff(str(exp["path"]))
            if coeffs is not None:
                top_n = st.slider("Top N", 10, 100, 40, 5, key=f"final_coeff_{exp['name']}")
                pdf = coeffs.to_pandas()
                if "coefficient" in pdf.columns:
                    pdf["abs_val"] = pdf["coefficient"].abs()
                    plot_df = pdf.nlargest(top_n, "abs_val").sort_values("coefficient")
                    fig = px.bar(plot_df, y="feature", x="coefficient", orientation="h",
                        color="coefficient", color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        title=f"Top {top_n} Coefficients — {exp['name']}")
                    fig.add_vline(x=0, line_dash="dash", line_color="gray")
                    fig.update_layout(height=max(400, top_n * 20), yaxis_title="", xaxis_title="Effect")
                    st.plotly_chart(fig, use_container_width=True)

            # Metadata
            with st.expander("Metadata"):
                st.json(exp["metadata"])
                if exp["model_info"]:
                    st.json(exp["model_info"])

render_footer()
```

**Step 2: Run the app to verify**

```bash
uv run streamlit run src/streamlit/Home.py
```

Navigate to Experiments page and verify both tabs render.

**Step 3: Commit**

```bash
git add src/streamlit/pages/2\ Experiments.py
git commit -m "feat: rebuild Experiments page with local loading"
```

---

### Task 3: Verify and iterate

**Step 1:** Run the app and check:
- Eval tab shows metrics table for each outcome
- Scatter plots render with LOESS
- Hurdle experiments show classification metrics + confusion matrix
- Feature importance expanders work
- Finalized tab shows model cards with metrics and coefficients

**Step 2:** Fix any issues found during testing.

**Step 3:** Final commit with any fixes.
