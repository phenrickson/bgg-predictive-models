"""
Streamlit page for exploring model coefficient rankings.

Visualize top designers, mechanics, publishers, etc. by their estimated
effect on each outcome, with uncertainty from Bayesian model coefficients.
"""

import sys
import os
import logging

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.streamlit.components.experiment_loader import (
    discover_experiments,
    load_coefficients,
)
from src.utils.coefficient_rankings import (
    FEATURE_CATEGORIES,
    extract_category_coefficients,
    rank_coefficients_across_experiments,
    get_category_summary,
)
from src.streamlit.components.footer import render_footer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Rankings | BGG Models Dashboard", layout="wide")
st.title("Rankings")
st.caption("Estimated effects of designers, mechanics, publishers, and more on board game outcomes")


# ============================
# Data loading
# ============================


@st.cache_data(ttl=300)
def get_experiments():
    return discover_experiments()


@st.cache_data(ttl=300)
def load_coeff_cached(path_str):
    from pathlib import Path
    return load_coefficients(Path(path_str))


experiments = get_experiments()

if not experiments:
    st.warning("No experiments found. Run training first.")
    st.stop()

# Get finalized experiments (the production models)
finalized_exps = [e for e in experiments if e["is_finalized"]]
eval_exps = [e for e in experiments if e["is_eval"]]

if not finalized_exps and not eval_exps:
    st.warning("No experiments with coefficients found.")
    st.stop()

# ============================
# Controls
# ============================

all_outcomes = sorted(set(
    e["outcome"] for e in (finalized_exps + eval_exps)
))

ctrl_row1 = st.columns(3)
with ctrl_row1[0]:
    selected_outcome = st.selectbox("Outcome", all_outcomes, key="rank_outcome")
with ctrl_row1[1]:
    selected_category = st.selectbox(
        "Feature Category",
        list(FEATURE_CATEGORIES.keys()),
        key="rank_category",
    )
with ctrl_row1[2]:
    # Source: finalized model or eval experiments
    source_options = []
    outcome_finalized = [e for e in finalized_exps if e["outcome"] == selected_outcome]
    outcome_evals = [e for e in eval_exps if e["outcome"] == selected_outcome]

    if outcome_finalized:
        source_options.append("Finalized Model")
    if outcome_evals:
        source_options.append("Eval Experiments (across years)")

    if not source_options:
        st.warning(f"No experiments found for {selected_outcome}")
        st.stop()

    selected_source = st.selectbox("Source", source_options, key="rank_source")

ctrl_row2 = st.columns(3)
with ctrl_row2[0]:
    top_n = st.select_slider("Top N", options=list(range(10, 510, 10)), value=50, key="rank_top_n")
with ctrl_row2[1]:
    show_positive_only = st.checkbox("Positive effects only", value=False, key="rank_pos_only")
with ctrl_row2[2]:
    show_negative_only = st.checkbox("Negative effects only", value=False, key="rank_neg_only")


# ============================
# Load coefficients
# ============================


if selected_source == "Finalized Model":
    # Use the first (usually only) finalized experiment for this outcome
    exp = outcome_finalized[0]
    coeffs = load_coeff_cached(str(exp["path"]))

    if coeffs is None:
        st.warning(f"No coefficients found for {exp['name']}")
        st.stop()

    # Extract category coefficients
    ranked = extract_category_coefficients(coeffs.to_pandas(), selected_category)

    if ranked.empty:
        st.info(f"No {selected_category} coefficients found (all may have been shrunk to zero by ARD).")
        st.stop()

    # Apply filters
    if show_positive_only:
        ranked = ranked[ranked["coefficient"] > 0]
    if show_negative_only:
        ranked = ranked[ranked["coefficient"] < 0]

    ranked = ranked.head(top_n)

    # Summary
    st.markdown(f"**{exp['name']}** — {len(ranked)} {selected_category.lower()}s shown")

    # Category summary
    with st.expander("Category overview"):
        summary = get_category_summary(coeffs.to_pandas())
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ---- Dot plot with error bars ----
    has_std = "std" in ranked.columns and ranked["std"].notna().any()

    fig = go.Figure()

    if has_std:
        # 95% CI error bars
        fig.add_trace(go.Scatter(
            x=ranked["coefficient"],
            y=ranked["name"],
            mode="markers",
            marker=dict(size=8, color=ranked["coefficient"], colorscale="RdBu", cmid=0),
            error_x=dict(
                type="data",
                array=1.96 * ranked["std"],
                arrayminus=1.96 * ranked["std"],
                color="rgba(100,100,100,0.4)",
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Effect: %{x:.4f}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
    else:
        fig.add_trace(go.Scatter(
            x=ranked["coefficient"],
            y=ranked["name"],
            mode="markers",
            marker=dict(size=8, color=ranked["coefficient"], colorscale="RdBu", cmid=0),
            hovertemplate="<b>%{y}</b><br>Effect: %{x:.4f}<extra></extra>",
            showlegend=False,
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    outcome_label = selected_outcome.replace("_", " ").title()
    fig.update_layout(
        title=f"Top {selected_category}s — Effect on {outcome_label}",
        xaxis_title="Coefficient (effect size)",
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=ranked["name"].tolist()[::-1],
        ),
        height=700,
        margin=dict(l=200),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Data table ----
    display_cols = ["name", "coefficient"]
    if "std" in ranked.columns:
        display_cols.append("std")
    for col in ["lower_95", "upper_95"]:
        if col in ranked.columns:
            display_cols.append(col)

    st.dataframe(
        ranked[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


elif selected_source == "Eval Experiments (across years)":
    # Load coefficients for all eval experiments for this outcome
    exp_coefficients = {}
    for exp in outcome_evals:
        coeffs = load_coeff_cached(str(exp["path"]))
        if coeffs is not None:
            # Use test year as experiment label
            test_year = exp["metadata"].get("metadata", {}).get("test_through", exp["name"])
            exp_coefficients[str(test_year)] = coeffs.to_pandas()

    if not exp_coefficients:
        st.warning("No coefficient data found for eval experiments.")
        st.stop()

    st.markdown(f"**{len(exp_coefficients)} experiments** — years: {', '.join(sorted(exp_coefficients.keys()))}")

    # Ranked summary across experiments
    ranked = rank_coefficients_across_experiments(
        exp_coefficients, selected_category, top_n=top_n
    )

    if ranked.empty:
        st.info(f"No {selected_category} coefficients found across experiments.")
        st.stop()

    # Apply filters
    if show_positive_only:
        ranked = ranked[ranked["mean_coefficient"] > 0]
    if show_negative_only:
        ranked = ranked[ranked["mean_coefficient"] < 0]

    # ---- Dot plot: mean with min/max range ----
    fig = go.Figure()

    # Range bar (min to max across years)
    fig.add_trace(go.Scatter(
        x=ranked["mean_coefficient"],
        y=ranked["name"],
        mode="markers",
        marker=dict(size=8, color=ranked["mean_coefficient"], colorscale="RdBu", cmid=0),
        error_x=dict(
            type="data",
            array=ranked["max_coefficient"] - ranked["mean_coefficient"],
            arrayminus=ranked["mean_coefficient"] - ranked["min_coefficient"],
            color="rgba(100,100,100,0.3)",
        ),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean: %{x:.4f}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    ))

    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    outcome_label = selected_outcome.replace("_", " ").title()
    fig.update_layout(
        title=f"Top {selected_category}s — Mean Effect on {outcome_label} (across eval years)",
        xaxis_title="Mean coefficient (range: min–max across years)",
        yaxis=dict(
            title="",
            categoryorder="array",
            categoryarray=ranked["name"].tolist()[::-1],
        ),
        height=700,
        margin=dict(l=200),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---- Per-year dot plot ----
    with st.expander("Per-year coefficients", expanded=False):
        # Build per-year data for the top entities
        top_names = ranked["name"].tolist()
        frames = []
        for year, coeff_df in sorted(exp_coefficients.items()):
            extracted = extract_category_coefficients(coeff_df, selected_category)
            if not extracted.empty:
                year_top = extracted[extracted["name"].isin(top_names)]
                year_top = year_top.copy()
                year_top["year"] = year
                frames.append(year_top[["name", "coefficient", "year"]])

        if frames:
            per_year = pd.concat(frames, ignore_index=True)

            from plotly.colors import sample_colorscale

            years = sorted(per_year["year"].unique())
            positions = [0.15 + 0.70 * i / max(len(years) - 1, 1) for i in range(len(years))]
            year_colors = sample_colorscale("Viridis", positions)

            fig2 = go.Figure()
            for year, color in zip(years, year_colors):
                yr_df = per_year[per_year["year"] == year]
                fig2.add_trace(go.Scatter(
                    x=yr_df["coefficient"],
                    y=yr_df["name"],
                    mode="markers",
                    marker=dict(size=7, color=color),
                    name=str(year),
                    hovertemplate="<b>%{y}</b><br>%{x:.4f}<br>Year: %{fullData.name}<extra></extra>",
                ))

            fig2.add_vline(x=0, line_dash="dash", line_color="gray")
            fig2.update_layout(
                title=f"{selected_category} Coefficients by Year",
                xaxis_title="Coefficient",
                yaxis=dict(
                    title="",
                    categoryorder="array",
                    categoryarray=top_names[::-1],
                ),
                height=700,
                legend_title="Year",
                margin=dict(l=200),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ---- Data table ----
    st.dataframe(
        ranked.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


render_footer()
