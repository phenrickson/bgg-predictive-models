"""YoY simulation experiment comparison.

Pick one or more ``simulation_name``s (each lives at
``models/bgg/snapshots/v{N}/simulations/<name>/<split>/v{M}/``), choose a
range of YoY splits, and compare metrics + predictions side-by-side.

The point of comparison is the chain-level simulation, not individual model
fits — see page 2 ("Experiments") for per-model inspection of trained
pipelines.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage  # noqa: E402

try:
    from src.streamlit.components.footer import render_footer
except ImportError:
    def render_footer():  # type: ignore
        pass


st.set_page_config(page_title="YoY Experiments | BGG Models Dashboard", layout="wide")
st.title("YoY Experiments")
st.caption(
    "Compare chain-level simulations across experiment configs and YoY splits. "
    "Each `simulation_name` is one experiment's set of YoY runs under "
    "`models/bgg/snapshots/v{N}/simulations/<name>/`."
)

# Cascade order — same order plots use elsewhere.
OUTCOMES = ["complexity", "users_rated", "rating", "geek_rating"]
LOG_MIN_RATINGS = float(np.log1p(25))

# ============================
# Discovery / loaders
# ============================


@st.cache_data(ttl=30)
def discover_snapshot_versions(base_dir: str) -> List[int]:
    base = Path(base_dir)
    if not base.exists():
        return []
    out = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("v"):
            try:
                out.append(int(child.name[1:]))
            except ValueError:
                continue
    return sorted(out)


@st.cache_data(ttl=30)
def discover_experiments(snapshot_version: int, base_dir: str) -> Dict[str, List[str]]:
    """Map simulation_name → list of split_name."""
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    return {name: storage.list_simulation_splits(name) for name in storage.list_simulation_names()}


@st.cache_data(ttl=30)
def load_simulation_artifacts(
    snapshot_version: int,
    base_dir: str,
    simulation_name: str,
    split_name: str,
    version: Optional[int] = None,
) -> Optional[Dict]:
    """Return the latest sim's registration + metrics + predictions parquet path.

    We hand back the parquet path (not the loaded frame) so individual sections
    can load only what they need.
    """
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    versions = storage.list_simulation_versions(simulation_name, split_name)
    if not versions:
        return None
    v = version if version is not None else versions[-1]
    sim_dir = storage.simulation_dir(simulation_name, split_name, v)
    reg_path = sim_dir / "registration.json"
    met_path = sim_dir / "metrics.json"
    pred_path = sim_dir / "predictions.parquet"
    if not (reg_path.exists() and met_path.exists() and pred_path.exists()):
        return None
    return {
        "version": v,
        "sim_dir": str(sim_dir),
        "registration": json.loads(reg_path.read_text()),
        "metrics": json.loads(met_path.read_text()),
        "predictions_path": str(pred_path),
    }


@st.cache_data(ttl=120)
def load_predictions_df(pred_path: str) -> pd.DataFrame:
    import polars as pl
    return pl.read_parquet(pred_path).to_pandas()


# ============================
# Selectors
# ============================

snapshot_versions = discover_snapshot_versions(DEFAULT_BASE_DIR)
if not snapshot_versions:
    st.warning("No snapshots found at `models/bgg/snapshots/`. Run `just bgg-build` first.")
    st.stop()

# Top-of-page selectors (kept off the sidebar — this page owns its filters).
top_cols = st.columns([1, 3, 3])
with top_cols[0]:
    snap = st.selectbox(
        "Snapshot",
        snapshot_versions,
        index=len(snapshot_versions) - 1,
        key="top_snap",
    )

experiments_index = discover_experiments(snap, DEFAULT_BASE_DIR)
if not experiments_index:
    st.warning(f"No simulations found under snapshot v{snap}. Run `just bgg-simulate-yoy` first.")
    st.stop()

available_names = list(experiments_index.keys())
with top_cols[1]:
    selected_names = st.multiselect(
        "Experiments",
        available_names,
        default=available_names,
        key="top_experiments",
    )
if not selected_names:
    st.info("Select at least one experiment.")
    st.stop()

# Splits that exist for ALL selected experiments (intersection) — keeps the
# comparison apples-to-apples.
common_splits = sorted(
    set.intersection(*[set(experiments_index[n]) for n in selected_names])
)
if not common_splits:
    st.warning("Selected experiments have no overlapping splits.")
    st.stop()

with top_cols[2]:
    selected_splits = st.multiselect(
        "Splits",
        common_splits,
        default=common_splits,
        key="top_splits",
    )
if not selected_splits:
    st.info("Select at least one split.")
    st.stop()


# ============================
# Load everything we'll need
# ============================

# {(name, split): artifacts}
artifacts: Dict[tuple, Dict] = {}
for name in selected_names:
    for split in selected_splits:
        a = load_simulation_artifacts(snap, DEFAULT_BASE_DIR, name, split)
        if a is not None:
            artifacts[(name, split)] = a

if not artifacts:
    st.error("No usable simulation artifacts under the current selection.")
    st.stop()


# Build the metrics dataframe once for all tabs to reuse.
rows = []
for (name, split), a in artifacts.items():
    reg = a["registration"]
    met = a["metrics"]
    for outcome_key, outcome_data in met.items():
        if not isinstance(outcome_data, dict):
            continue
        row = {
            "experiment": name,
            "split": split,
            "eval_year": reg.get("eval_year"),
            "outcome": outcome_key,
            "version": a["version"],
        }
        for k, v in outcome_data.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)

metrics_df = pd.DataFrame(rows) if rows else None

# Tabs scope the page so selectors land near the chart they affect.
tab_metrics, tab_over_time, tab_pred_actual, tab_compare = st.tabs(
    ["Metrics", "Over time", "Predicted vs Actual", "A vs B"]
)

# ============================
# Tab: Metrics table
# ============================
with tab_metrics:
    if metrics_df is None:
        st.info("No metrics found.")
    else:
        outcome_filter = st.multiselect(
            "Outcomes",
            sorted(metrics_df["outcome"].unique()),
            default=sorted(metrics_df["outcome"].unique()),
            key="metrics_outcome_filter",
        )
        show_df = metrics_df[metrics_df["outcome"].isin(outcome_filter)].copy()
        show_df = show_df.sort_values(
            ["outcome", "eval_year", "experiment"]
        ).reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True)

# ============================
# Tab: Metrics over time
# ============================
with tab_over_time:
    if metrics_df is None:
        st.info("No metrics found.")
    else:
        chart_metric = st.selectbox(
            "Metric",
            ["rmse_point", "r2_point", "rmse_sim", "r2_sim", "coverage_90", "coverage_50", "n"],
            index=0,
            key="metric_over_time",
        )
        plot_df = metrics_df.copy()
        # Drop the *_rated variants — those live in the metrics table; the
        # chart sticks to the four primary cascade outcomes.
        plot_df = plot_df[plot_df["outcome"].isin(OUTCOMES)]
        outcome_order = [o for o in OUTCOMES if o in plot_df["outcome"].unique()]
        plot_df = plot_df[plot_df[chart_metric].notna()]
        plot_df["outcome"] = pd.Categorical(
            plot_df["outcome"], categories=outcome_order, ordered=True
        )
        plot_df = plot_df.sort_values(["outcome", "eval_year"])

        # Force eval_year to be a categorical string so axes don't show
        # midpoints like "2022.5" between two integer years.
        plot_df["eval_year_str"] = plot_df["eval_year"].astype(int).astype(str)
        year_order = sorted(plot_df["eval_year_str"].unique())

        palette = px.colors.qualitative.Set1
        color_map = {n: palette[i % len(palette)] for i, n in enumerate(selected_names)}

        fig = px.line(
            plot_df,
            x="eval_year_str",
            y=chart_metric,
            color="experiment",
            color_discrete_map=color_map,
            facet_col="outcome",
            facet_col_wrap=3,
            markers=True,
            category_orders={"eval_year_str": year_order, "outcome": outcome_order},
            title=f"{chart_metric} by eval year",
        )
        fig.update_xaxes(
            matches=None,
            showticklabels=True,
            type="category",
            title="eval year",
        )
        # Shared y-axis across facets so visual differences are honest.
        fig.update_yaxes(showticklabels=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        n_rows_facets = (len(outcome_order) + 2) // 3
        fig.update_layout(
            height=320 * n_rows_facets,
            hovermode="x unified",
            margin=dict(t=60, b=40, l=40, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================
# Tab: Predicted vs Actual overlay
# ============================
with tab_pred_actual:
    scatter_cols = st.columns(3)
    with scatter_cols[0]:
        scatter_split = st.selectbox(
            "Split",
            selected_splits,
            index=len(selected_splits) - 1,
            key="scatter_split",
        )
    with scatter_cols[1]:
        scatter_outcome = st.selectbox(
            "Outcome",
            OUTCOMES,
            index=OUTCOMES.index("geek_rating"),
            key="scatter_outcome",
        )
    with scatter_cols[2]:
        only_rated = st.checkbox(
            "Only games with ≥25 ratings",
            value=False,
            key="scatter_only_rated",
            help="Filters to games with at least 25 user ratings (real-signal subset).",
        )

    pa_fig = go.Figure()
    all_min, all_max = None, None
    for name in selected_names:
        a = artifacts.get((name, scatter_split))
        if a is None:
            continue
        df = load_predictions_df(a["predictions_path"])
        pred_col = f"{scatter_outcome}_point"
        actual_col = f"{scatter_outcome}_actual"
        if pred_col not in df.columns or actual_col not in df.columns:
            continue
        sub = df.dropna(subset=[pred_col, actual_col]).copy()
        if scatter_outcome in ("complexity", "rating"):
            sub = sub[sub[actual_col] != 0]
        elif scatter_outcome == "geek_rating":
            sub = sub[sub["users_rated_actual"] != 0]
        if only_rated and "users_rated_actual" in sub.columns:
            sub = sub[sub["users_rated_actual"] >= LOG_MIN_RATINGS]
        if sub.empty:
            continue
        pred = sub[pred_col].to_numpy()
        actual = sub[actual_col].to_numpy()
        name_col = sub["name"] if "name" in sub.columns else None
        n = len(sub)
        r = float(np.corrcoef(pred, actual)[0, 1]) if n > 1 else float("nan")
        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        pa_fig.add_trace(
            go.Scatter(
                x=pred,
                y=actual,
                mode="markers",
                marker=dict(size=5, opacity=0.4),
                name=f"{name}  (n={n}, r={r:.3f}, rmse={rmse:.3f})",
                text=name_col,
                hovertemplate=(
                    "<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>"
                    if name_col is not None
                    else "Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>"
                ),
            )
        )
        lo_v = float(min(pred.min(), actual.min()))
        hi_v = float(max(pred.max(), actual.max()))
        all_min = lo_v if all_min is None else min(all_min, lo_v)
        all_max = hi_v if all_max is None else max(all_max, hi_v)

    if all_min is not None:
        pa_fig.add_trace(
            go.Scatter(
                x=[all_min, all_max],
                y=[all_min, all_max],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="y = x",
                showlegend=True,
            )
        )

    pa_fig.update_layout(
        title=f"{scatter_outcome.replace('_', ' ').title()} — {scatter_split}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=600,
        hovermode="closest",
    )
    st.plotly_chart(pa_fig, use_container_width=True)

# ============================
# Tab: A vs B per-game comparison
# ============================
with tab_compare:
    st.caption(
        "Compare two experiments' simulated predictions game-by-game for a chosen "
        "(split, outcome). The scatter plots experiment A on the x-axis and B on "
        "the y-axis — points off the diagonal are games where the two cascades "
        "disagree. Color marks low-information games (actual users_rated < 25)."
    )

    if len(selected_names) < 2:
        st.info("Select at least two experiments at the top of the page to enable A vs B comparison.")
    else:
        cmp_cols = st.columns(4)
        with cmp_cols[0]:
            cmp_split = st.selectbox(
                "Split",
                selected_splits,
                index=len(selected_splits) - 1,
                key="cmp_split",
            )
        with cmp_cols[1]:
            cmp_outcome = st.selectbox(
                "Outcome",
                OUTCOMES,
                index=OUTCOMES.index("rating"),
                key="cmp_outcome",
            )
        with cmp_cols[2]:
            exp_a = st.selectbox("Experiment A", selected_names, index=0, key="cmp_a")
        with cmp_cols[3]:
            b_default = next((i for i, n in enumerate(selected_names) if n != exp_a), 0)
            exp_b = st.selectbox("Experiment B", selected_names, index=b_default, key="cmp_b")

        a_art = artifacts.get((exp_a, cmp_split))
        b_art = artifacts.get((exp_b, cmp_split))

        if a_art is None or b_art is None or exp_a == exp_b:
            st.info("Pick two distinct experiments that both have results for this split.")
        else:
            a_df = load_predictions_df(a_art["predictions_path"])
            b_df = load_predictions_df(b_art["predictions_path"])
            pred_col = f"{cmp_outcome}_point"
            actual_col = f"{cmp_outcome}_actual"

            keep_cols = ["game_id"]
            if "name" in a_df.columns:
                keep_cols.append("name")
            # users_rated_actual lives only in the predictions parquet, log-scale.
            if "users_rated_actual" in a_df.columns:
                keep_cols.append("users_rated_actual")

            a_sub = a_df[keep_cols + [pred_col, actual_col]].rename(
                columns={pred_col: "pred_a", actual_col: "actual"}
            )
            b_sub = b_df[["game_id", pred_col]].rename(columns={pred_col: "pred_b"})
            merged = a_sub.merge(b_sub, on="game_id", how="inner").dropna(
                subset=["pred_a", "pred_b"]
            )
            merged["diff"] = merged["pred_a"] - merged["pred_b"]
            merged["abs_diff"] = merged["diff"].abs()

            # ≥25-ratings split (matching the conventions used elsewhere)
            if "users_rated_actual" in merged.columns:
                low_mask = merged["users_rated_actual"] < LOG_MIN_RATINGS
            else:
                low_mask = pd.Series(False, index=merged.index)

            n_total = len(merged)
            n_high = int((~low_mask).sum())
            n_low = int(low_mask.sum())
            r = float(np.corrcoef(merged["pred_a"], merged["pred_b"])[0, 1]) if n_total > 1 else float("nan")
            n_above_01 = int((merged["abs_diff"] > 0.1).sum())
            n_above_03 = int((merged["abs_diff"] > 0.3).sum())
            st.caption(
                f"n={n_total:,}  ·  r(A, B) = {r:.4f}  ·  "
                f"|diff|>0.1: {n_above_01:,} games ({n_above_01/n_total*100:.1f}%)  ·  "
                f"|diff|>0.3: {n_above_03:,} games ({n_above_03/n_total*100:.1f}%)"
            )

            # ---- Scatter: A vs B ----
            scatter = go.Figure()
            if n_low > 0:
                sub = merged[low_mask]
                scatter.add_trace(
                    go.Scatter(
                        x=sub["pred_a"],
                        y=sub["pred_b"],
                        mode="markers",
                        marker=dict(size=5, opacity=0.4, color="darkorange"),
                        name=f"<25 ratings (n={n_low})",
                        text=sub.get("name"),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            f"{exp_a}: %{{x:.3f}}<br>"
                            f"{exp_b}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )
            if n_high > 0:
                sub = merged[~low_mask]
                scatter.add_trace(
                    go.Scatter(
                        x=sub["pred_a"],
                        y=sub["pred_b"],
                        mode="markers",
                        marker=dict(size=5, opacity=0.5, color="steelblue"),
                        name=f"≥25 ratings (n={n_high})",
                        text=sub.get("name"),
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            f"{exp_a}: %{{x:.3f}}<br>"
                            f"{exp_b}: %{{y:.3f}}<extra></extra>"
                        ),
                    )
                )

            lo = float(min(merged["pred_a"].min(), merged["pred_b"].min()))
            hi = float(max(merged["pred_a"].max(), merged["pred_b"].max()))
            scatter.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="y = x",
                )
            )
            scatter.update_layout(
                title=f"{cmp_outcome.replace('_', ' ').title()} — {cmp_split}",
                xaxis_title=f"{exp_a}  pred",
                yaxis_title=f"{exp_b}  pred",
                height=600,
                hovermode="closest",
            )
            st.plotly_chart(scatter, use_container_width=True)

            # ---- Histogram of diffs ----
            hist_fig = px.histogram(
                merged,
                x="diff",
                color=low_mask.map({True: f"<25", False: f"≥25"}),
                color_discrete_map={"<25": "darkorange", "≥25": "steelblue"},
                nbins=60,
                title=f"`{exp_a} - {exp_b}` distribution",
                barmode="overlay",
                opacity=0.55,
            )
            hist_fig.update_layout(
                height=320,
                legend_title="users_rated_actual",
            )
            st.plotly_chart(hist_fig, use_container_width=True)

            # ---- Biggest swing table ----
            st.subheader("Biggest swings")
            show_cols = [
                c for c in ("game_id", "name", "pred_a", "pred_b", "diff", "actual", "users_rated_actual")
                if c in merged.columns
            ]
            st.dataframe(
                merged.sort_values("abs_diff", ascending=False)
                .head(500)
                [show_cols]
                .rename(columns={"pred_a": exp_a, "pred_b": exp_b}),
                use_container_width=True,
            )


render_footer()
