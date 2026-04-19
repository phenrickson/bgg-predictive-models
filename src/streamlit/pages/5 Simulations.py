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
)
from src.streamlit.components.footer import render_footer
from src.streamlit.components.simulation_explorer import (
    call_simulate_samples,
    call_explain_game,
    plot_posterior_distributions,
    plot_explanation,
    DEFAULT_MODEL_NAMES,
)
import pandas as pd


@st.cache_data(ttl=600, show_spinner=False)
def _cached_simulate(game_id: int, service_url: str, n_samples: int) -> dict:
    return call_simulate_samples(
        game_ids=[game_id], service_url=service_url, n_samples=n_samples
    )


@st.cache_data(ttl=600, show_spinner=False)
def _cached_explain(game_id: int, service_url: str) -> dict:
    return call_explain_game(game_id=game_id, service_url=service_url)


@st.cache_data(ttl=600)
def load_game_catalog_from_run(run_path_str: str) -> pd.DataFrame:
    """Load game_id/name/test_year from a local simulation run's predictions.

    Returns columns: game_id, name, test_year, geek_rating_point (nullable).
    """
    from pathlib import Path

    run_path = Path(run_path_str)
    preds = load_predictions(run_path)
    if preds is None:
        return pd.DataFrame(columns=["game_id", "name", "test_year", "geek_rating_point"])

    cols = ["game_id", "name", "test_year"]
    if "geek_rating_point" in preds.columns:
        cols.append("geek_rating_point")
    df = preds.select(cols).unique(subset=["game_id"]).to_pandas()
    if "geek_rating_point" not in df.columns:
        df["geek_rating_point"] = pd.NA
    return df

# Optional import for LOESS smoothing
try:
    import statsmodels.nonparametric.smoothers_lowess as lowess_mod

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

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
    st.warning(
        "No simulation runs found in models/simulation/. Run `make evaluate` first."
    )
    st.stop()

run_names = [r["name"] for r in runs]

# --- Tabs ---
tab_compare, tab_detail, tab_explorer = st.tabs(["Compare Runs", "Run Detail", "Game Explorer"])

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
    else:
        selected_runs = [r for r in runs if r["name"] in selected_run_names]

        # Load and combine summary metrics
        @st.cache_data
        def load_combined_summary(run_paths_and_names):
            from pathlib import Path

            frames = []
            for name, path_str in run_paths_and_names:
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
                c
                for c in combined_metrics.columns
                if c not in ("run", "test_year", "outcome", "n")
            ]
            default_idx = numeric_cols.index("rmse_point") if "rmse_point" in numeric_cols else 0
            selected_metric = st.selectbox("Select metric", numeric_cols, index=default_idx)

            if selected_metric:
                plot_df = combined_metrics.select(
                    ["run", "test_year", "outcome", selected_metric]
                ).to_pandas()
                plot_df["test_year"] = plot_df["test_year"].astype(str)

                # One chart per outcome: x=test_year, one line per run
                outcomes_in_data = plot_df["outcome"].unique()
                chart_cols = st.columns(min(len(outcomes_in_data), 2))
                for i, outcome in enumerate(outcomes_in_data):
                    outcome_df = plot_df[plot_df["outcome"] == outcome]
                    fig = px.line(
                        outcome_df,
                        x="test_year",
                        y=selected_metric,
                        color="run",
                        markers=True,
                        title=outcome.replace("_", " ").title(),
                        height=350,
                    )
                    fig.update_layout(
                        xaxis_title="Test Year",
                        yaxis_title=selected_metric,
                        xaxis_type="category",
                        yaxis_rangemode="tozero",
                    )
                    with chart_cols[i % len(chart_cols)]:
                        st.plotly_chart(fig, use_container_width=True)

        # Run metadata
        st.subheader("Run Metadata")
        for r in selected_runs:
            with st.expander(r["name"]):
                st.json(r["metadata"])

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
    elif len(predictions) == 0:
        st.warning("Predictions file is empty.")
    else:
        # --- Filters ---
        st.subheader("Filters")
        filter_cols = st.columns(3)

        # Outcome selector (single)
        with filter_cols[0]:
            selected_outcome = st.selectbox(
                "Outcome",
                OUTCOMES,
                key="detail_outcome",
            )
            selected_outcomes = [selected_outcome]

        # Test year filter
        available_years = sorted(predictions["test_year"].unique().to_list())
        with filter_cols[1]:
            selected_years = st.multiselect(
                "Test years",
                available_years,
                default=available_years,
                key="detail_years",
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

        # --- Scatter plots ---
        st.subheader("Predicted vs Actual")

        scatter_cols = st.columns(2)
        for i, outcome in enumerate(selected_outcomes):
            actual_col = f"{outcome}_actual"
            point_col = f"{outcome}_point"

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

            if len(pdf) == 0:
                continue

            fig = go.Figure()

            # Scatter
            fig.add_trace(
                go.Scatter(
                    x=pdf[point_col],
                    y=pdf[actual_col],
                    mode="markers",
                    marker=dict(size=4, opacity=0.5),
                    text=pdf["name"],
                    hovertemplate="<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )

            # Diagonal
            min_val = min(pdf[point_col].min(), pdf[actual_col].min())
            max_val = max(pdf[point_col].max(), pdf[actual_col].max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="y=x",
                    showlegend=False,
                )
            )

            # LOESS
            if HAS_STATSMODELS and len(pdf) > 10:
                try:
                    sorted_idx = np.argsort(pdf[point_col].values)
                    x_sorted = pdf[point_col].values[sorted_idx]
                    y_sorted = pdf[actual_col].values[sorted_idx]
                    smoothed = lowess_mod.lowess(y_sorted, x_sorted, frac=2 / 3, it=5)
                    fig.add_trace(
                        go.Scatter(
                            x=smoothed[:, 0],
                            y=smoothed[:, 1],
                            mode="lines",
                            line=dict(color="green", width=2, dash="dot"),
                            name="LOESS",
                            showlegend=False,
                        )
                    )
                except Exception:
                    pass

            # Correlation annotation
            corr = pdf[point_col].corr(pdf[actual_col])
            label = (
                "log users rated"
                if outcome == "users_rated"
                else outcome.replace("_", " ").title()
            )
            fig.update_layout(
                title=f"{label} (r={corr:.3f})",
                xaxis_title="Predicted (Point)",
                yaxis_title="Actual",
                height=400,
            )

            with scatter_cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

        # --- Forest plot ---
        st.subheader("Top Games \u2014 Prediction Intervals")

        if selected_outcomes:
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

            forest_cols = [
                c
                for c in [actual_col, median_col, lower_90, upper_90, lower_50, upper_50, "name", "game_id"]
                if c in filtered.columns
            ]

            if all(c in filtered.columns for c in [median_col, lower_90, upper_90]):
                forest_df = (
                    filtered.select(forest_cols)
                    .sort(median_col, descending=True)
                    .head(top_n)
                    .to_pandas()
                )
                forest_df = forest_df.iloc[::-1]  # reverse for bottom-to-top display

                fig = go.Figure()

                # 90% interval lines
                for _, row in forest_df.iterrows():
                    fig.add_shape(
                        type="line",
                        x0=row[lower_90],
                        x1=row[upper_90],
                        y0=row["name"],
                        y1=row["name"],
                        line=dict(color="steelblue", width=1),
                    )

                # 50% interval lines
                if lower_50 in forest_df.columns:
                    for _, row in forest_df.iterrows():
                        fig.add_shape(
                            type="line",
                            x0=row[lower_50],
                            x1=row[upper_50],
                            y0=row["name"],
                            y1=row["name"],
                            line=dict(color="steelblue", width=3),
                        )

                # Median
                fig.add_trace(
                    go.Scatter(
                        x=forest_df[median_col],
                        y=forest_df["name"],
                        mode="markers",
                        marker=dict(color="steelblue", size=6),
                        name="Median",
                        hovertemplate="<b>%{y}</b><br>Median: %{x:.3f}<extra></extra>",
                    )
                )

                # Actual (filter out invalid values for display)
                if actual_col in forest_df.columns:
                    valid_actuals = forest_df.copy()
                    if forest_outcome in ("complexity", "rating", "users_rated"):
                        valid_actuals = valid_actuals[valid_actuals[actual_col] != 0]
                    elif forest_outcome == "geek_rating" and "users_rated_actual" in forest_df.columns:
                        valid_actuals = valid_actuals[forest_df["users_rated_actual"] != 0]

                    if len(valid_actuals) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=valid_actuals[actual_col],
                                y=valid_actuals["name"],
                                mode="markers",
                                marker=dict(color="red", size=6, symbol="circle-open"),
                                name="Actual",
                                hovertemplate="<b>%{y}</b><br>Actual: %{x:.3f}<extra></extra>",
                            )
                        )

                label = forest_outcome.replace("_", " ").title()
                fig.update_layout(
                    title=f"Top {top_n} Games \u2014 {label}",
                    xaxis_title=label,
                    height=max(500, top_n * 22),
                    showlegend=True,
                    margin=dict(l=250),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Predictions table ---
        st.subheader("Predictions")
        st.dataframe(filtered.to_pandas(), use_container_width=True)

# ============================
# Tab 3: Game Explorer
# ============================
with tab_explorer:

    @st.fragment
    def game_explorer():
        st.header("Game Explorer")
        st.caption("Simulate posterior distributions for specific games via the scoring service.")

        run_options = {r["name"]: str(r["path"]) for r in runs}
        explorer_run_name = st.selectbox(
            "Run to source games from",
            options=list(run_options.keys()),
            key="explorer_source_run",
        )
        catalog = load_game_catalog_from_run(run_options[explorer_run_name])
        if catalog.empty:
            st.warning("No predictions found for the selected run.")
            return

        max_year = int(catalog["test_year"].max())
        latest = catalog[catalog["test_year"] == max_year].sort_values(
            "geek_rating_point", ascending=False, na_position="last"
        )

        source = st.radio(
            "Game source",
            [f"Latest test year ({max_year})", "Search any game in run"],
            horizontal=True,
            key="explorer_game_source",
        )

        if source.startswith("Latest test year"):
            options = latest
        else:
            options = catalog.sort_values("name")

        def _label(row: pd.Series) -> str:
            year = int(row["test_year"]) if pd.notna(row["test_year"]) else "?"
            return f"{row['name']} ({year}) — {int(row['game_id'])}"

        label_to_id = {_label(row): int(row["game_id"]) for _, row in options.iterrows()}

        explorer_cols = st.columns([2, 1])
        with explorer_cols[0]:
            selected_label = st.selectbox(
                "Game",
                options=list(label_to_id.keys()),
                key="explorer_game_select",
            )
        with explorer_cols[1]:
            n_samples = st.slider(
                "Samples", min_value=100, max_value=500, value=500, step=100,
                key="explorer_n_samples",
            )

        service_url = st.text_input(
            "Scoring service URL",
            value="http://localhost:8087",
            key="explorer_service_url",
        )

        game_ids = [label_to_id[selected_label]] if selected_label else []

        if game_ids:
            gid = game_ids[0]
            try:
                with st.spinner(f"Simulating game {gid} ({n_samples} samples)..."):
                    sim_response = _cached_simulate(gid, service_url, n_samples)
                st.session_state["explorer_sim_results"] = sim_response["games"]
                st.session_state["explorer_sim_error"] = None
            except Exception as e:
                st.session_state["explorer_sim_results"] = None
                st.session_state["explorer_sim_error"] = str(e)

            try:
                with st.spinner(f"Explaining game {gid}..."):
                    explain_response = _cached_explain(gid, service_url)
                st.session_state["explorer_explain_results"] = [explain_response]
            except Exception as e:
                st.session_state["explorer_explain_results"] = []
                st.error(f"Error explaining game {gid}: {e}")

        # Render cached results
        if st.session_state.get("explorer_sim_error"):
            st.error(f"Error calling scoring service: {st.session_state['explorer_sim_error']}")
        if st.session_state.get("explorer_sim_results"):
            for game in st.session_state["explorer_sim_results"]:
                fig = plot_posterior_distributions(game)
                st.plotly_chart(fig, use_container_width=True)
                # Show explanation for this game if available
                explain_results = st.session_state.get("explorer_explain_results", [])
                matching = [r for r in explain_results if r["game_id"] == game["game_id"]]
                if matching:
                    fig = plot_explanation(
                        explanations=matching[0]["explanations"],
                        game_name=matching[0].get("game_name", str(game["game_id"])),
                        game_id=matching[0]["game_id"],
                    )
                    st.plotly_chart(fig, use_container_width=True)

    game_explorer()

# Footer
render_footer()
