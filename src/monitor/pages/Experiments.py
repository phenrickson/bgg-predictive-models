"""
Streamlit dashboard for exploring model experiments.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import plotly.express as px
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

# Import dashboard modules
from src.monitor.experiment_dashboard import (
    load_experiments,
    create_metrics_overview,
    create_performance_by_run_visualization,
    create_parameters_overview,
    create_feature_importance_plot,
    create_category_feature_importance_plots,
    get_experiment_details,
    display_predictions as display_experiment_predictions,
)

st.set_page_config(page_title="Experiments | BGG Models Dashboard", layout="wide")
st.title("Experiment Tracking")

# Model Type Selection
model_types = [
    d.name
    for d in Path("models/experiments").iterdir()
    if d.is_dir() and d.name not in ["predictions"]
]
selected_model_type = st.sidebar.selectbox("Select Model Type", model_types)

# Load experiments
experiments = load_experiments(selected_model_type)

# Check if any experiments were loaded
if not experiments:
    st.error("No experiments found. Please ensure experiments have been tracked.")
    st.stop()

# Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Metrics Overview",
        "Predictions",
        "Parameters",
        "Feature Importance",
        "Experiment Details",
        "Experiment Metadata",
    ]
)

with tab1:
    st.header("Metrics Overview")
    selected_dataset = st.selectbox("Select Dataset", ["train", "tune", "test"])
    metrics_df = create_metrics_overview(experiments, selected_dataset)
    st.subheader("Metrics Table")
    st.dataframe(metrics_df)

    st.subheader("Performance by Model Run")
    if len(metrics_df) > 0:
        available_metrics = [
            col for col in metrics_df.columns if col not in ["Experiment", "Timestamp"]
        ]
        selected_metrics = st.multiselect(
            "Select Metrics to Visualize",
            available_metrics,
            default=available_metrics[: min(3, len(available_metrics))],
        )
        if selected_metrics:
            performance_fig = create_performance_by_run_visualization(
                experiments, selected_dataset, selected_metrics
            )
            if performance_fig:
                st.plotly_chart(performance_fig, use_container_width=True)

with tab2:
    st.header("Model Predictions")
    selected_experiment = st.selectbox(
        "Select Experiment",
        [exp["full_name"] for exp in experiments],
        key="predictions_experiment",
    )
    exp_name, exp_version = selected_experiment.split("/")
    exp_version = int(exp_version.replace("v", ""))
    selected_dataset = st.selectbox(
        "Select Dataset", ["tune", "test"], key="predictions_dataset"
    )
    # Find selected experiment
    selected_exp = next(
        (exp for exp in experiments if exp["full_name"] == selected_experiment),
        None,
    )
    if selected_exp:
        predictions_df = display_experiment_predictions(
            selected_exp, selected_dataset, "regression", selected_model_type
        )
    else:
        st.error("Could not find the selected experiment")
        st.stop()
    if predictions_df is not None:
        st.dataframe(predictions_df)

with tab3:
    st.header("Model Parameters")
    params_df = create_parameters_overview(experiments)
    st.dataframe(params_df)

with tab4:
    st.header("Feature Importance")
    selected_experiment = st.selectbox(
        "Select Experiment",
        [exp["full_name"] for exp in experiments],
        key="feature_importance_experiment",
    )
    exp_name, exp_version = selected_experiment.split("/")
    exp_version = int(exp_version.replace("v", ""))
    selected_exp = next(
        (exp for exp in experiments if exp["full_name"] == selected_experiment),
        None,
    )
    if selected_exp:
        top_n = st.slider(
            "Top N Features (Overall)",
            min_value=5,
            max_value=250,
            value=40,
            step=5,
        )
        feature_fig = create_feature_importance_plot(
            selected_exp, model_type=selected_model_type, top_n=top_n
        )
        if feature_fig:
            st.plotly_chart(feature_fig)

        st.subheader("Feature Importance by Category")
        top_n_per_category = st.slider(
            "Top N Features (Per Category)",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
        )
        category_plots = create_category_feature_importance_plots(
            selected_exp,
            model_type=selected_model_type,
            top_n_per_category=top_n_per_category,
        )
        if category_plots:
            category_names = list(category_plots.keys())
            if len(category_names) > 0:
                category_tabs = st.tabs(category_names)
                for i, (category_name, fig) in enumerate(category_plots.items()):
                    with category_tabs[i]:
                        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Experiment Details")
    selected_experiment = st.selectbox(
        "Select Experiment",
        [exp["full_name"] for exp in experiments],
        key="details_experiment",
    )
    exp_name, exp_version = selected_experiment.split("/")
    exp_version = int(exp_version.replace("v", ""))
    try:
        details = get_experiment_details(selected_model_type, exp_name, exp_version)
        st.json(details)
    except Exception as e:
        st.error(f"Could not load experiment details: {e}")

with tab6:
    st.header("Experiment Metadata")
    for exp in experiments:
        with st.expander(f"{exp['full_name']} Metadata"):
            st.json(exp)
