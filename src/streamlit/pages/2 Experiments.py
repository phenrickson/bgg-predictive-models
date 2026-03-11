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


# ============================
# Helper functions
# ============================


def _render_regression_scatter(preds: pl.DataFrame, outcome: str, exp_name: str):
    """Render predicted vs actual scatter plot for regression."""
    select_cols = ["prediction", "actual"] + [
        c for c in ["name", "game_id"] if c in preds.columns
    ]
    pdf = preds.select(select_cols).to_pandas()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pdf["prediction"],
            y=pdf["actual"],
            mode="markers",
            marker=dict(size=4, opacity=0.5),
            text=pdf.get("name"),
            hovertemplate="<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )

    min_val = min(pdf["prediction"].min(), pdf["actual"].min())
    max_val = max(pdf["prediction"].max(), pdf["actual"].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash"),
            showlegend=False,
        )
    )

    if HAS_STATSMODELS and len(pdf) > 10:
        try:
            sorted_idx = np.argsort(pdf["prediction"].values)
            smoothed = lowess_mod.lowess(
                pdf["actual"].values[sorted_idx],
                pdf["prediction"].values[sorted_idx],
                frac=2 / 3,
                it=5,
            )
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

    corr = pdf["prediction"].corr(pdf["actual"])
    label = outcome.replace("_", " ").title()
    fig.update_layout(
        title=f"{label} — {exp_name} (r={corr:.3f})",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_classification_metrics(preds: pl.DataFrame):
    """Render classification metrics and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    pdf = preds.to_pandas()
    cols = st.columns(4)
    cols[0].metric(
        "Accuracy", f"{accuracy_score(pdf['actual'], pdf['prediction']):.2%}"
    )
    cols[1].metric(
        "Precision", f"{precision_score(pdf['actual'], pdf['prediction']):.2%}"
    )
    cols[2].metric(
        "Recall", f"{recall_score(pdf['actual'], pdf['prediction']):.2%}"
    )
    cols[3].metric("F1", f"{f1_score(pdf['actual'], pdf['prediction']):.2%}")

    cm = confusion_matrix(pdf["actual"], pdf["prediction"])
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Negative", "Positive"],
        y=["Negative", "Positive"],
        title="Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)


FEATURE_CATEGORIES = {
    "All": None,
    "Designer": "designer_",
    "Publisher": "publisher_",
    "Artist": "artist_",
    "Mechanic": "mechanic_",
    "Category": "category_",
    "Family": "family_",
    "Embedding": "emb_",
    "Other": "__other__",
}


def _render_coefficients_by_year(exp_coeffs: list, key_prefix: str, outcome: str = ""):
    """Render dot plot of coefficients colored by year.

    Args:
        exp_coeffs: list of (label, coeffs_df) tuples where label is e.g. "2022"
        key_prefix: unique key prefix for streamlit widgets
        outcome: outcome name for display in chart title
    """
    import pandas as pd

    ctrl_cols = st.columns(2)
    with ctrl_cols[0]:
        top_n = st.slider(
            "Top N features", 10, 100, 40, 5, key=f"coeff_n_{key_prefix}"
        )
    with ctrl_cols[1]:
        category = st.selectbox(
            "Feature category",
            list(FEATURE_CATEGORIES.keys()),
            key=f"coeff_cat_{key_prefix}",
        )

    # Combine all coefficients with their year label
    frames = []
    for label, coeffs in exp_coeffs:
        pdf = coeffs.to_pandas()
        if "coefficient" not in pdf.columns:
            continue
        pdf["year"] = str(label)
        frames.append(pdf[["feature", "coefficient", "year"]])

    if not frames:
        st.info("No coefficient data available.")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Filter by category and clean up display names
    prefix = FEATURE_CATEGORIES[category]
    if prefix == "__other__":
        known_prefixes = [p for p in FEATURE_CATEGORIES.values() if p and p != "__other__"]
        combined = combined[~combined["feature"].apply(
            lambda f: any(f.startswith(p) for p in known_prefixes)
        )]
    elif prefix is not None:
        combined = combined[combined["feature"].str.startswith(prefix)]

    # Clean feature names: strip prefix, replace underscores, title case
    def _clean_feature(name):
        if prefix and prefix != "__other__":
            name = name[len(prefix):]
        return name.replace("_", " ").strip().title()

    combined["display_name"] = combined["feature"].apply(_clean_feature)

    # Drop zero coefficients (ARD shrinks many to exactly zero)
    combined = combined[combined["coefficient"] != 0]

    if combined.empty:
        st.info(f"No coefficients found for category: {category}")
        return

    # Pick top N features by max absolute coefficient across all years
    feature_max = (
        combined.groupby("display_name")["coefficient"]
        .apply(lambda x: x.abs().max())
        .nlargest(top_n)
        .index
    )
    plot_df = combined[combined["display_name"].isin(feature_max)]

    # Sort features by mean coefficient: largest positive at top
    feature_order = (
        plot_df.groupby("display_name")["coefficient"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

    fig = go.Figure()
    years = sorted(plot_df["year"].unique())
    # Use viridis color scale sampled evenly across years
    from plotly.colors import sample_colorscale
    # Truncate viridis to avoid dark navy and bright yellow extremes
    scale_positions = [
        0.15 + 0.70 * i / max(len(years) - 1, 1) for i in range(len(years))
    ]
    year_colors = sample_colorscale("Viridis", scale_positions)
    for year, color in zip(years, year_colors):
        year_df = plot_df[plot_df["year"] == year]
        year_df = year_df.set_index("display_name").reindex(feature_order).reset_index()
        fig.add_trace(go.Scatter(
            x=year_df["coefficient"],
            y=year_df["display_name"],
            mode="markers",
            marker=dict(size=8, color=color),
            name=str(year),
            hovertemplate="<b>%{y}</b><br>Coefficient: %{x:.4f}<br>Year: %{fullData.name}<extra></extra>",
            hoverlabel=dict(bgcolor=color, font_color="white"),
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    outcome_title = outcome.replace("_", " ").title() if outcome else ""
    title_parts = [f"Top {top_n} Coefficients"]
    if outcome_title:
        title_parts.append(outcome_title)
    title_parts.append(category)
    fig.update_layout(
        title=" — ".join(title_parts),
        height=max(400, len(feature_order) * 22),
        yaxis=dict(title="", type="category", categoryorder="array", categoryarray=feature_order),
        xaxis_title="Coefficient",
        legend_title="Year",
        hoverlabel=dict(font_size=13),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_coefficients_single(coeffs: pl.DataFrame, key_prefix: str):
    """Render coefficient bar chart for a single experiment."""
    top_n = st.slider(
        "Top N features", 10, 100, 40, 5, key=f"coeff_n_{key_prefix}"
    )

    pdf = coeffs.to_pandas()
    if "coefficient" in pdf.columns:
        pdf["abs_val"] = pdf["coefficient"].abs()
        plot_df = pdf.nlargest(top_n, "abs_val").sort_values("coefficient")
        fig = px.bar(
            plot_df,
            y="feature",
            x="coefficient",
            orientation="h",
            color="coefficient",
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            title=f"Top {top_n} Coefficients",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=max(400, top_n * 20),
            yaxis_title="",
            xaxis_title="Effect",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Unknown coefficient format:", pdf.columns.tolist())


# ============================
# Data loading
# ============================


@st.cache_data(ttl=60)
def get_experiments():
    return discover_experiments()


experiments = get_experiments()

if not experiments:
    st.warning(
        "No experiments found in models/experiments/. Run training first."
    )
    st.stop()

# Split into eval and finalized
eval_exps = [e for e in experiments if e["is_eval"]]
finalized_exps = [e for e in experiments if e["is_finalized"]]

# --- Tabs ---
tab_eval, tab_finalized, tab_metadata = st.tabs(["Experiments", "Models", "Metadata"])

# ============================
# Tab 1: Experiments
# ============================
with tab_eval:
    st.header("Experiments")

    if not eval_exps:
        st.info("No eval experiments found.")
    else:
        eval_outcomes = sorted(set(e["outcome"] for e in eval_exps))
        filter_cols = st.columns(2)
        with filter_cols[0]:
            selected_outcome = st.selectbox(
                "Outcome", eval_outcomes, key="eval_outcome"
            )
        with filter_cols[1]:
            selected_dataset = st.selectbox(
                "Dataset", ["test", "tune", "train"], key="eval_dataset"
            )

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
                # Only include scalar metrics (skip nested dicts like confusion_matrix)
                for k, v in m.items():
                    if isinstance(v, (int, float, str, bool)):
                        # MAPE is not meaningful for count-based outcomes
                        if k == "mape" and selected_outcome == "users_rated":
                            continue
                        row[k] = v
                rows.append(row)

        if rows:
            metrics_df = pl.DataFrame(rows)
            st.dataframe(metrics_df.to_pandas(), use_container_width=True)

            # --- Metrics Over Time ---
            exclude_cols = {"experiment", "version", "test_year"}
            # MAPE is not meaningful for count-based outcomes like users_rated
            if selected_outcome == "users_rated":
                exclude_cols.add("mape")
            metric_cols = [
                c for c in metrics_df.columns
                if c not in exclude_cols
                and metrics_df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
            ]
            if metric_cols and "test_year" in metrics_df.columns:
                sorted_df = metrics_df.sort("test_year").to_pandas()
                melted = sorted_df.melt(
                    id_vars=["test_year", "experiment", "version"],
                    value_vars=metric_cols,
                    var_name="metric",
                    value_name="value",
                )
                fig = px.line(
                    melted,
                    x="test_year",
                    y="value",
                    color="version",
                    facet_col="metric",
                    facet_col_wrap=3,
                    facet_col_spacing=0.08,
                    facet_row_spacing=0.12,
                    markers=True,
                    title="Metrics Over Time",
                    custom_data=["experiment", "version", "metric"],
                )
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[2]}</b>: %{y:.4f}<br>"
                        "Year: %{x}<br>"
                        "Model: %{customdata[0]} (%{customdata[1]})"
                        "<extra></extra>"
                    )
                )
                fig.update_yaxes(matches=None, rangemode="tozero", showticklabels=True)
                fig.update_xaxes(matches=None)
                fig.update_annotations(font_size=12)
                n_rows = (len(metric_cols) + 2) // 3
                fig.update_layout(
                    xaxis_title="Test Year",
                    hovermode="closest",
                    height=350 * n_rows,
                    margin=dict(t=60, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"No {selected_dataset} metrics found for {selected_outcome} eval experiments."
            )

        # --- Predictions Scatter ---
        st.subheader("Predictions")
        pred_cols = st.columns(2)
        with pred_cols[0]:
            exp_names = sorted(set(e["name"] for e in outcome_exps))
            selected_exp_name = st.selectbox(
                "Experiment", exp_names, key="eval_exp_select"
            )
        matching_versions = sorted(
            [e["version"] for e in outcome_exps if e["name"] == selected_exp_name]
        )
        with pred_cols[1]:
            selected_version = st.selectbox(
                "Version", matching_versions, index=len(matching_versions) - 1, key="eval_version_select"
            )
        selected_exp = next(
            e for e in outcome_exps if e["name"] == selected_exp_name and e["version"] == selected_version
        )

        @st.cache_data
        def load_preds_cached(path_str, dataset):
            from pathlib import Path

            return load_predictions(Path(path_str), dataset)

        preds = load_preds_cached(str(selected_exp["path"]), selected_dataset)

        if preds is not None and "prediction" in preds.columns and "actual" in preds.columns:
            model_task = selected_exp["metadata"].get("metadata", {}).get(
                "model_task", "regression"
            )
            if model_task == "regression":
                _render_regression_scatter(preds, selected_outcome, selected_exp_name)
            elif model_task == "classification":
                _render_classification_metrics(preds)

            # Sort by predicted probability for classification, prediction for regression
            sort_col = "predicted_proba_class_1" if "predicted_proba_class_1" in preds.columns else "prediction"
            sorted_preds = preds.sort(sort_col, descending=True)
            st.caption(f"Showing {len(sorted_preds):,} predictions")
            st.dataframe(sorted_preds.to_pandas(), use_container_width=True)
        elif preds is not None:
            st.info("Predictions loaded but missing prediction column.")
        else:
            st.info(f"No {selected_dataset} predictions found.")

        # --- Feature Importance ---
        st.subheader("Coefficients")

        @st.cache_data
        def load_coeff_cached(path_str):
            from pathlib import Path
            return load_coefficients(Path(path_str))

        # Use only the latest version per experiment for coefficient plots
        latest_exps = {}
        for exp in outcome_exps:
            test_year = exp["metadata"].get("metadata", {}).get("test_through", exp["name"])
            key = str(test_year)
            if key not in latest_exps or exp["version"] > latest_exps[key]["version"]:
                latest_exps[key] = exp

        exp_coeffs = []
        for exp in latest_exps.values():
            coeffs = load_coeff_cached(str(exp["path"]))
            if coeffs is not None:
                test_year = exp["metadata"].get("metadata", {}).get("test_through", exp["name"])
                exp_coeffs.append((test_year, coeffs))

        if exp_coeffs:
            _render_coefficients_by_year(exp_coeffs, f"eval_{selected_outcome}", outcome=selected_outcome)
        else:
            st.info("No coefficient data found for these experiments.")

# ============================
# Tab 2: Models
# ============================
with tab_finalized:
    st.header("Models")

    if not finalized_exps:
        st.info("No finalized models found.")
    else:
        final_outcomes = sorted(set(e["outcome"] for e in finalized_exps))
        selected_final_outcome = st.selectbox(
            "Outcome", final_outcomes, key="final_outcome"
        )

        outcome_finals = [
            e for e in finalized_exps if e["outcome"] == selected_final_outcome
        ]

        for exp in outcome_finals:
            st.subheader(f"{exp['name']} ({exp['version']})")

            # Metrics
            m = load_metrics(exp["path"], "test")
            if m:
                model_task = exp["metadata"].get("metadata", {}).get(
                    "model_task", "regression"
                )
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

            # Model info summary
            meta = exp["metadata"].get("metadata", {})
            info_cols = st.columns(3)
            info_cols[0].metric("Features", meta.get("feature_count", "?"))
            info_cols[1].metric(
                "Train Size", meta.get("data_sizes", {}).get("train", "?")
            )
            info_cols[2].metric("Algorithm", meta.get("algorithm", "?"))

            # Coefficients
            @st.cache_data
            def load_final_coeff(path_str):
                from pathlib import Path

                return load_coefficients(Path(path_str))

            coeffs = load_final_coeff(str(exp["path"]))
            if coeffs is not None:
                _render_coefficients_single(coeffs, f"final_{exp['name']}_{exp['version']}")

            # Metadata
            with st.expander("Metadata"):
                st.json(exp["metadata"])
                if exp["model_info"]:
                    st.json(exp["model_info"])

# ============================
# Tab 3: Metadata
# ============================
with tab_metadata:
    st.header("Metadata")

    all_exps = eval_exps + finalized_exps
    if not all_exps:
        st.info("No experiments found.")
    else:
        for exp in all_exps:
            with st.expander(exp["name"]):
                st.json(exp["metadata"])
                if exp["model_info"]:
                    st.json(exp["model_info"])

render_footer()
