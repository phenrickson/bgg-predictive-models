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

# Import efficient experiment loader
from src.utils.experiment_loader import get_experiment_loader

# Import dashboard modules for visualization functions
from src.monitor.experiment_dashboard import (
    create_metrics_overview,
    create_performance_by_run_visualization,
    create_parameters_overview,
    create_feature_importance_plot,
    create_category_feature_importance_plots,
    display_predictions as display_experiment_predictions,
)

st.set_page_config(page_title="Experiments | BGG Models Dashboard", layout="wide")
st.title("Experiment Tracking")


# Initialize experiment loader
@st.cache_resource
def get_loader():
    """Get cached experiment loader instance."""
    return get_experiment_loader()


loader = get_loader()

# Model Type Selection
try:
    with st.spinner("Loading model types..."):
        model_types = loader.list_model_types()

    if not model_types:
        st.sidebar.warning("No model types found in GCS experiments bucket.")
        selected_model_type = None
    else:
        selected_model_type = st.sidebar.selectbox("Select Model Type", model_types)
except Exception as e:
    st.sidebar.error(f"Error loading model types: {e}")
    selected_model_type = None

# Cache management
st.sidebar.divider()
st.sidebar.header("🔄 Cache Management")
if st.sidebar.button("🗑️ Clear Cache", help="Clear cached experiment data"):
    loader.clear_cache()
    st.cache_data.clear()
    st.sidebar.success("✅ Cache cleared!")
    st.sidebar.info("Refresh the page to reload data")

# Only proceed if we have a valid model type selected
if selected_model_type is None:
    st.info("Please select a model type from the sidebar to view experiments.")
    st.stop()


# Load experiments using efficient loader
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_experiments_cached(model_type):
    """Load experiments with caching."""
    return loader.list_experiments(model_type)


with st.spinner(f"Loading experiments for {selected_model_type}..."):
    experiments = load_experiments_cached(selected_model_type)

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
    exp_name = selected_experiment
    selected_dataset = st.selectbox(
        "Select Dataset", ["tune", "test"], key="predictions_dataset"
    )

    try:
        with st.spinner(f"Loading {selected_dataset} predictions..."):
            predictions_df = loader.load_predictions(
                selected_model_type, exp_name, selected_dataset
            )

        if predictions_df is not None:
            st.subheader(f"Predictions for {selected_dataset} dataset")
            st.dataframe(predictions_df)

            # Basic statistics
            if len(predictions_df) > 0:
                st.subheader("Dataset Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(predictions_df))
                with col2:
                    if "prediction" in predictions_df.columns:
                        st.metric(
                            "Avg Prediction",
                            f"{predictions_df['prediction'].mean():.3f}",
                        )
                with col3:
                    if "actual" in predictions_df.columns:
                        st.metric(
                            "Avg Actual", f"{predictions_df['actual'].mean():.3f}"
                        )
        else:
            st.warning(f"No predictions found for {selected_dataset} dataset")
            st.info("This could mean:")
            st.info("1. Predictions were not saved during training")
            st.info("2. The selected dataset might be empty")
            st.info("3. Experiment tracking did not log predictions")
    except Exception as e:
        st.error(f"Error loading predictions: {e}")

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
    exp_name = selected_experiment

    try:
        with st.spinner("Loading feature importance..."):
            feature_importance_data = loader.load_feature_importance(
                selected_model_type, exp_name
            )

        if feature_importance_data:
            st.subheader("Feature Importance Data")
            st.json(feature_importance_data)

            # You can add visualization here if the data structure supports it
            # For now, just display the raw data
            st.info(
                "Feature importance visualization will be added based on the data structure"
            )
        else:
            st.warning("No feature importance data found for this experiment")
            st.info("This could mean:")
            st.info("1. Feature importance was not calculated during training")
            st.info("2. The feature importance file was not saved")
            st.info(
                "3. The experiment used a model type that doesn't support feature importance"
            )
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")

with tab5:
    st.header("Experiment Details")
    selected_experiment = st.selectbox(
        "Select Experiment",
        [exp["full_name"] for exp in experiments],
        key="details_experiment",
    )
    exp_name = selected_experiment
    try:
        with st.spinner("Loading experiment details..."):
            details = loader.load_experiment_details(selected_model_type, exp_name)
        st.json(details)
    except Exception as e:
        st.error(f"Could not load experiment details: {e}")

with tab6:
    st.header("Experiment Metadata")
    for exp in experiments:
        with st.expander(f"{exp['full_name']} Metadata"):
            st.json(exp)
