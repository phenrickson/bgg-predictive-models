"""
Streamlit dashboard for exploring model experiments.

This module provides a comprehensive interface for viewing and analyzing machine learning
experiments, including metrics, predictions, parameters, and feature importance visualizations.

Key Features:
- Interactive experiment selection and filtering
- Real-time metrics visualization and comparison
- Prediction analysis with regression and classification support
- Feature importance plots with category-specific breakdowns
- Experiment metadata and parameter exploration
- Efficient caching and data loading

The dashboard is organized into tabs:
1. Metrics Overview: Compare performance metrics across experiments
2. Predictions: Analyze model predictions with interactive plots
3. Parameters: View model hyperparameters and configurations
4. Feature Importance: Explore feature contributions and importance
5. Experiment Details: Raw experiment data and configurations
6. Experiment Metadata: Complete experiment metadata

Dependencies:
- streamlit: Web application framework
- plotly: Interactive plotting library
- pandas/polars: Data manipulation
- sklearn: Machine learning metrics
- statsmodels: Statistical analysis (optional, for LOESS smoothing)

Usage:
    Run with: streamlit run src/streamlit/pages/Experiments.py
"""

# Standard library imports
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Third-party imports
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# Optional import for LOESS smoothing
try:
    import statsmodels.nonparametric.smoothers_lowess as lowess

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

# Local application imports
from src.utils.experiment_loader import get_experiment_loader
from src.monitor.experiment_dashboard import (
    create_metrics_overview,
    create_performance_by_run_visualization,
    create_parameters_overview,
    create_feature_importance_plot,
    create_category_feature_importance_plots,
    display_predictions as display_experiment_predictions,
)
from src.models.experiments import ExperimentTracker
from src.streamlit.components.footer import render_footer

# Constants
CACHE_TTL = 300  # 5 minutes
TOP_N_FEATURES_DEFAULT = 40
TOP_N_FEATURES_PER_CATEGORY_DEFAULT = 25
MIN_USERS_RATED_FILTER_DEFAULT = 5


# Utility Functions
def validate_experiment_data(experiment: Dict[str, Any]) -> bool:
    """Validate experiment data structure."""
    required_keys = ["full_name", "experiment_name", "metrics"]
    return all(key in experiment for key in required_keys)


def create_hover_text(row: pd.Series) -> str:
    """Create hover text for prediction plots."""
    hover_info = f"Predicted: {row['prediction']:.4f}<br>Actual: {row['actual']:.4f}"
    if "game_id" in row.index:
        hover_info += f"<br>Game ID: {row['game_id']}"
    if "name" in row.index:
        hover_info += f"<br>Name: {row['name']}"
    return hover_info


def abbreviate_feature(feature: str, n: int = 40) -> str:
    """Abbreviate feature names for display."""
    if len(str(feature)) > n:
        return str(feature)[: n - 3] + "..."
    return str(feature)


def determine_model_type(
    experiment: Dict[str, Any], predictions_df: pd.DataFrame
) -> str:
    """Determine model type from experiment metadata or predictions."""
    try:
        # Try to get model type from metadata
        model_type = experiment.get("model_info", {}).get("model_type", "regression")
        if not model_type:
            # Fallback to checking if it's a classification task
            if "threshold" in predictions_df.columns:
                model_type = "classification"
            else:
                model_type = "regression"
        return model_type
    except Exception as e:
        logger.warning(f"Error determining model type: {e}")
        return "regression"


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

# Debug information
logger.debug(f"Loaded {len(experiments)} experiments for {selected_model_type}")
if experiments:
    logger.debug(f"First experiment structure: {list(experiments[0].keys())}")
    if "metrics" in experiments[0]:
        logger.debug(f"Metrics structure: {list(experiments[0]['metrics'].keys())}")

# Note: Debug information removed for cleaner UI
# Debug information is still logged to console for development purposes

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

    # Check if experiments exist
    if not experiments:
        st.error("No experiments available. Please track some experiments first.")
        st.stop()

    # Validate experiments list
    if len(experiments) == 0:
        st.warning("No experiments found. Have you run any model training?")
        st.info("To see predictions:")
        st.info("1. Train a model using the experiment tracking system")
        st.info("2. Ensure predictions are logged during training")
        st.stop()

    try:
        # Select specific experiment
        selected_experiment = st.selectbox(
            "Select Experiment",
            [exp["full_name"] for exp in experiments],
            key="predictions_experiment",
        )

        # Find the selected experiment
        selected_exp = next(
            (exp for exp in experiments if exp["full_name"] == selected_experiment),
            None,
        )

        if selected_exp:
            # Load all predictions for the selected experiment once
            @st.cache_data
            def load_all_predictions_cached(model_type, exp_name):
                return loader.load_all_predictions(model_type, exp_name)

            try:
                all_predictions = load_all_predictions_cached(
                    selected_model_type, selected_exp["experiment_name"]
                )

                # Dataset selector - now just filters the loaded data
                available_datasets = list(all_predictions.keys())
                if not available_datasets:
                    st.warning("No prediction datasets found for this experiment")
                    st.stop()

                selected_dataset = st.selectbox(
                    "Select Dataset", available_datasets, key="predictions_dataset"
                )

                predictions_df = all_predictions.get(selected_dataset)

                if predictions_df is not None:
                    # Convert pandas DataFrame to polars for compatibility
                    display_df = pl.from_pandas(predictions_df)

                    # Determine model type from experiment metadata
                    model_type = determine_model_type(selected_exp, predictions_df)

                    # Special handling for geek_rating model type
                    if selected_model_type == "geek_rating":
                        # Check if 'actual' column exists
                        if "actual" not in display_df.columns:
                            st.error("No 'actual' column found in the DataFrame")
                            st.warning(f"Available columns: {display_df.columns}")
                        else:
                            # Replace non-numeric values with 5.5
                            try:
                                display_df = display_df.with_columns(
                                    [
                                        pl.when(
                                            ~pl.col("actual")
                                            .cast(pl.Float64)
                                            .is_finite()
                                        )
                                        .then(pl.lit(5.5))
                                        .otherwise(pl.col("actual"))
                                        .alias("actual")
                                    ]
                                )
                                st.success(
                                    "Set non-numeric 'actual' values to 5.5 for geek_rating model"
                                )
                            except Exception as e:
                                st.error(f"Error replacing null values: {e}")

                    # Add filters if users_rated column exists
                    if "users_rated" in display_df.columns:
                        min_users_rated_filter = st.slider(
                            "Minimum Number of Users Rated (Percentile)",
                            min_value=0,
                            max_value=100,
                            value=5,
                            step=5,
                        )

                        # Calculate the actual minimum users_rated based on the percentage
                        actual_min_users_rated = int(
                            display_df["users_rated"].quantile(
                                min_users_rated_filter / 100
                            )
                        )

                        # Apply users_rated filter
                        display_df = display_df.filter(
                            pl.col("users_rated") >= actual_min_users_rated
                        )

                        # Check if year_published column exists and add year filter
                        if "year_published" in display_df.columns:
                            # Get min and max years
                            min_year = display_df["year_published"].min()
                            max_year = display_df["year_published"].max()

                            # Only show year filter if there's a range of years
                            if min_year != max_year:
                                year_filter = st.slider(
                                    "Year Published Range",
                                    min_value=int(min_year),
                                    max_value=int(max_year),
                                    value=(int(min_year), int(max_year)),
                                )

                                # Apply year filter
                                display_df = display_df.filter(
                                    (pl.col("year_published") >= year_filter[0])
                                    & (pl.col("year_published") <= year_filter[1])
                                )

                    # Add filter for geek_rating model type to exclude games with actual = 5.5
                    if selected_model_type == "geek_rating":
                        filter_5_5 = st.checkbox(
                            "Filter out games where actual rating = 5.5", value=False
                        )
                        if filter_5_5:
                            display_df = display_df.filter(pl.col("actual") != 5.5)
                            st.info("Filtered out games where actual rating = 5.5")

                    # Display basic info about the predictions
                    st.write(
                        f"**Shape:** {display_df.shape[0]} rows, {display_df.shape[1]} columns"
                    )

                    # Performance metrics and visualization based on model type
                    if (
                        model_type == "regression"
                        and "prediction" in display_df.columns
                        and "actual" in display_df.columns
                    ):
                        # Convert to pandas for sklearn metrics
                        df_pandas = display_df.to_pandas()

                        # Calculate metrics
                        mse = mean_squared_error(
                            df_pandas["actual"], df_pandas["prediction"]
                        )
                        mae = mean_absolute_error(
                            df_pandas["actual"], df_pandas["prediction"]
                        )
                        r2 = r2_score(df_pandas["actual"], df_pandas["prediction"])
                        mape = mean_absolute_percentage_error(
                            df_pandas["actual"], df_pandas["prediction"]
                        )

                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                        with col2:
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                        with col3:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col4:
                            st.metric("MAPE", f"{mape:.2f}%")

                        # Prepare hover text with game_id and name if available
                        hover_text = [
                            create_hover_text(row) for _, row in df_pandas.iterrows()
                        ]

                        # Scatter plot of predicted vs actual with hover information
                        fig = go.Figure()

                        # Add scatter points
                        fig.add_trace(
                            go.Scatter(
                                x=df_pandas["prediction"],
                                y=df_pandas["actual"],
                                mode="markers",
                                marker=dict(
                                    size=5,
                                    color="blue",
                                    opacity=0.7,
                                    line=dict(width=0.5, color="darkblue"),
                                ),
                                text=hover_text,
                                hoverinfo="text",
                                hovertemplate="%{text}<extra></extra>",
                                showlegend=False,
                            )
                        )

                        # Add perfect prediction line
                        fig.add_trace(
                            go.Scatter(
                                x=[
                                    df_pandas["prediction"].min(),
                                    df_pandas["prediction"].max(),
                                ],
                                y=[
                                    df_pandas["prediction"].min(),
                                    df_pandas["prediction"].max(),
                                ],
                                mode="lines",
                                name="Perfect Prediction",
                                line=dict(color="red", dash="dash"),
                                hoverinfo="none",
                            )
                        )

                        # Add LOESS trend line if available
                        if HAS_STATSMODELS:
                            try:
                                # Sort data for LOESS smoothing
                                sorted_indices = np.argsort(df_pandas["prediction"])
                                x_sorted = df_pandas["prediction"].iloc[sorted_indices]
                                y_sorted = df_pandas["actual"].iloc[sorted_indices]

                                # Apply LOESS smoothing
                                loess_smoothed = lowess.lowess(
                                    y_sorted, x_sorted, frac=2 / 3, it=5
                                )

                                # Add LOESS line
                                fig.add_trace(
                                    go.Scatter(
                                        x=loess_smoothed[:, 0],
                                        y=loess_smoothed[:, 1],
                                        mode="lines",
                                        name="LOESS Trend",
                                        line=dict(color="green", width=2, dash="dot"),
                                    )
                                )
                            except Exception as e:
                                logger.warning(f"Error adding LOESS trend line: {e}")
                        else:
                            st.info("Install statsmodels for LOESS trend line")

                        # Update layout
                        fig.update_layout(
                            title="Predicted vs Actual Values",
                            xaxis_title="Predicted",
                            yaxis_title="Actual",
                            hovermode="closest",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    elif (
                        model_type == "classification"
                        and "prediction" in display_df.columns
                        and "actual" in display_df.columns
                    ):
                        # Convert to pandas for sklearn metrics
                        df_pandas = display_df.to_pandas()

                        # Calculate metrics
                        accuracy = accuracy_score(
                            df_pandas["actual"], df_pandas["prediction"]
                        )
                        precision = precision_score(
                            df_pandas["actual"], df_pandas["prediction"]
                        )
                        recall = recall_score(
                            df_pandas["actual"], df_pandas["prediction"]
                        )
                        f1 = f1_score(df_pandas["actual"], df_pandas["prediction"])
                        roc_auc = roc_auc_score(
                            df_pandas["actual"], df_pandas["prediction"]
                        )

                        # Display metrics
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Accuracy", f"{accuracy:.2%}")
                        with col2:
                            st.metric("Precision", f"{precision:.2%}")
                        with col3:
                            st.metric("Recall", f"{recall:.2%}")
                        with col4:
                            st.metric("F1 Score", f"{f1:.2%}")
                        with col5:
                            st.metric("ROC AUC", f"{roc_auc:.2%}")

                        # Confusion Matrix Visualization
                        cm = confusion_matrix(
                            df_pandas["actual"], df_pandas["prediction"]
                        )
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=["Negative", "Positive"],
                            y=["Negative", "Positive"],
                            title="Confusion Matrix",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show the dataframe
                    st.dataframe(display_df.to_pandas())

                else:
                    st.warning(
                        f"No predictions available for {selected_dataset} dataset"
                    )
                    st.info("Possible reasons:")
                    st.info("1. Predictions were not saved during training")
                    st.info("2. The selected dataset might be empty")
                    st.info("3. Experiment tracking did not log predictions")

            except Exception as e:
                st.error(f"Error loading predictions: {e}")
                st.info("Troubleshooting tips:")
                st.info("1. Verify the experiment was tracked correctly")
                st.info("2. Check that prediction files exist in GCS")
                st.info("3. Ensure the experiment structure is valid")
                logger.exception("Predictions loading error")
        else:
            st.error("Could not find the selected experiment")

    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        st.info("Troubleshooting tips:")
        st.info("1. Verify the experiment was tracked correctly")
        st.info("2. Check that predictions were logged during training")
        st.info("3. Ensure the experiment file is not corrupted")
        logger.exception("Predictions loading error")

with tab3:
    st.header("Model Parameters")
    params_df = create_parameters_overview(experiments)
    st.dataframe(params_df)

with tab4:
    st.header("Feature Importance")

    # Load all feature importance data for the selected model type (cached)
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_all_feature_importance_cached(model_type):
        """Load all feature importance data with caching."""
        return loader.load_all_feature_importance(model_type)

    with st.spinner(f"Loading feature importance data for {selected_model_type}..."):
        all_feature_importance = load_all_feature_importance_cached(selected_model_type)

    # Check if any feature importance data was loaded
    if not all_feature_importance:
        st.warning("No feature importance data available for this model type")
        st.info("This could mean:")
        st.info("1. Feature importance was not calculated during training")
        st.info("2. The feature importance files were not saved")
        st.info(
            "3. The experiments used model types that don't support feature importance"
        )
    else:
        # Select specific experiment from available ones
        available_experiments = list(all_feature_importance.keys())
        experiment_display_names = [
            exp["full_name"]
            for exp in experiments
            if exp["experiment_name"] in available_experiments
        ]

        if not experiment_display_names:
            st.warning("No experiments with feature importance data found")
        else:
            selected_experiment = st.selectbox(
                "Select Experiment",
                experiment_display_names,
                key="feature_importance_experiment",
            )

            # Find the selected experiment
            selected_exp = next(
                (exp for exp in experiments if exp["full_name"] == selected_experiment),
                None,
            )

            # Feature importance configuration for overall plot
            top_n = st.slider(
                "Top N Features (Overall)", min_value=5, max_value=250, value=40, step=5
            )

            if (
                selected_exp
                and selected_exp["experiment_name"] in all_feature_importance
            ):
                # Get feature importance data from the cached batch load
                feature_importance_df = all_feature_importance[
                    selected_exp["experiment_name"]
                ]

                if feature_importance_df is not None:
                    # Overall feature importance plot
                    st.subheader("Overall Feature Importance")

                    # Determine the importance column name
                    importance_col = None
                    if "feature_importance" in feature_importance_df.columns:
                        importance_col = "feature_importance"
                    elif "coefficient" in feature_importance_df.columns:
                        importance_col = "coefficient"
                    elif "abs_feature_importance" in feature_importance_df.columns:
                        importance_col = "abs_feature_importance"
                    elif "abs_coefficient" in feature_importance_df.columns:
                        importance_col = "abs_coefficient"

                    if importance_col:
                        # Sort by absolute importance and take top N
                        if importance_col in ["coefficient"]:
                            # For coefficients, sort by absolute value
                            sorted_df = feature_importance_df.copy()
                            sorted_df["abs_importance"] = sorted_df[
                                importance_col
                            ].abs()
                            sorted_df = sorted_df.nlargest(top_n, "abs_importance")
                        else:
                            # For feature importance, sort directly
                            sorted_df = feature_importance_df.nlargest(
                                top_n, importance_col
                            )

                        # Create horizontal bar plot
                        sorted_df["abbreviated_feature"] = sorted_df["feature"].apply(
                            abbreviate_feature
                        )

                        # Create the plot
                        if importance_col == "coefficient":
                            # For coefficients, use signed values with color scale
                            fig = px.bar(
                                sorted_df.sort_values(importance_col),
                                y="abbreviated_feature",
                                x=importance_col,
                                orientation="h",
                                color=importance_col,
                                color_continuous_scale="RdBu",
                                color_continuous_midpoint=0,
                                title=f"Top {top_n} Features - {selected_exp['experiment_name']} (Coefficients)",
                            )
                            # Add zero line
                            fig.add_vline(
                                x=0, line_width=2, line_dash="dash", line_color="gray"
                            )
                        else:
                            # For feature importance, use positive values
                            fig = px.bar(
                                sorted_df.sort_values(importance_col, ascending=True),
                                y="abbreviated_feature",
                                x=importance_col,
                                orientation="h",
                                color=importance_col,
                                color_continuous_scale="Viridis",
                                title=f"Top {top_n} Features - {selected_exp['experiment_name']} (Feature Importance)",
                            )

                        # Update layout
                        fig.update_layout(
                            height=max(400, len(sorted_df) * 20),
                            width=800,
                            yaxis_title="Feature",
                            xaxis_title=(
                                "Importance"
                                if importance_col != "coefficient"
                                else "Effect"
                            ),
                            title_x=0.5,
                            coloraxis_colorbar=dict(title=""),
                        )

                        # Add hover text with full feature names
                        fig.update_traces(
                            hovertemplate="<b>%{y}</b><br>Full Name: %{text}<br>"
                            + (
                                f"{importance_col.replace('_', ' ').title()}: %{{x:.4f}}<extra></extra>"
                            ),
                            text=sorted_df["feature"],
                            texttemplate="",
                            textposition="none",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Category-specific feature importance plots
                        st.subheader("Feature Importance by Category")

                        # Configuration for category plots
                        top_n_per_category = st.slider(
                            "Top N Features (Per Category)",
                            min_value=5,
                            max_value=100,
                            value=25,
                            step=5,
                        )

                        # Define categories to look for
                        categories = {
                            "Publisher": "publisher_",
                            "Designer": "designer_",
                            "Artist": "artist_",
                            "Mechanic": "mechanic_",
                            "Category": "category_",
                            "Family": "family_",
                        }

                        category_plots = {}

                        for category_name, prefix in categories.items():
                            # Filter features for this category
                            category_features = feature_importance_df[
                                feature_importance_df["feature"].str.startswith(prefix)
                            ]

                            if len(category_features) == 0:
                                continue

                            # Sort and select top N features
                            if importance_col == "coefficient":
                                category_features["abs_importance"] = category_features[
                                    importance_col
                                ].abs()
                                category_sorted = category_features.nlargest(
                                    top_n_per_category, "abs_importance"
                                )
                                category_sorted = category_sorted.sort_values(
                                    importance_col
                                )
                            else:
                                category_sorted = category_features.nlargest(
                                    top_n_per_category, importance_col
                                )
                                category_sorted = category_sorted.sort_values(
                                    importance_col, ascending=True
                                )

                            # Abbreviate feature names (remove prefix)
                            category_sorted["abbreviated_feature"] = category_sorted[
                                "feature"
                            ].apply(
                                lambda x: abbreviate_feature(
                                    str(x).replace(prefix, ""), 30
                                )
                            )

                            # Create plot for this category
                            if importance_col == "coefficient":
                                cat_fig = px.bar(
                                    category_sorted,
                                    y="abbreviated_feature",
                                    x=importance_col,
                                    orientation="h",
                                    color=importance_col,
                                    color_continuous_scale="RdBu",
                                    color_continuous_midpoint=0,
                                    title=f"Top {min(top_n_per_category, len(category_sorted))} {category_name} Features",
                                )
                                cat_fig.add_vline(
                                    x=0,
                                    line_width=2,
                                    line_dash="dash",
                                    line_color="gray",
                                )
                            else:
                                cat_fig = px.bar(
                                    category_sorted,
                                    y="abbreviated_feature",
                                    x=importance_col,
                                    orientation="h",
                                    color=importance_col,
                                    color_continuous_scale="Viridis",
                                    title=f"Top {min(top_n_per_category, len(category_sorted))} {category_name} Features",
                                )

                            cat_fig.update_layout(
                                height=max(400, len(category_sorted) * 25),
                                width=800,
                                yaxis_title="Feature",
                                xaxis_title=(
                                    "Importance"
                                    if importance_col != "coefficient"
                                    else "Effect"
                                ),
                                title_x=0.5,
                                coloraxis_colorbar=dict(title=""),
                            )

                            cat_fig.update_traces(
                                hovertemplate="<b>%{y}</b><br>Full Name: %{text}<br>"
                                + (
                                    f"{importance_col.replace('_', ' ').title()}: %{{x:.4f}}<extra></extra>"
                                ),
                                text=category_sorted["feature"],
                                texttemplate="",
                                textposition="none",
                            )

                            category_plots[category_name] = cat_fig

                        if category_plots:
                            # Create tabs for each category
                            category_names = list(category_plots.keys())
                            if len(category_names) > 0:
                                category_tabs = st.tabs(category_names)

                                for i, (category_name, fig) in enumerate(
                                    category_plots.items()
                                ):
                                    with category_tabs[i]:
                                        st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(
                                    "No category-specific features found in this experiment."
                                )
                        else:
                            st.info(
                                "No category-specific features found in this experiment."
                            )

                    else:
                        st.warning(
                            "Could not determine importance column in feature importance data"
                        )
                        st.write(
                            "Available columns:", list(feature_importance_df.columns)
                        )

                else:
                    st.warning(
                        "No feature importance data available for this experiment"
                    )
                    st.info("This could mean:")
                    st.info("1. Feature importance was not calculated during training")
                    st.info("2. The feature importance file was not saved")
                    st.info(
                        "3. The experiment used a model type that doesn't support feature importance"
                    )

            else:
                st.warning("No feature importance data available for this experiment")
                st.info("This could mean:")
                st.info("1. Feature importance was not calculated during training")
                st.info("2. The feature importance file was not saved")
                st.info(
                    "3. The experiment used a model type that doesn't support feature importance"
                )

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
        with st.expander(f"{exp['full_name']}"):
            st.json(exp)

# Add footer with BGG logo
render_footer()
