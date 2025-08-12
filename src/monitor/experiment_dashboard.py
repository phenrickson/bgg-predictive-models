"""Streamlit-based Experiment Tracking Dashboard."""

import streamlit as st
import polars as pl
import plotly.express as px
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from src.models.experiments import ExperimentTracker  # noqa: E402


def format_timestamp(timestamp: str) -> str:
    """
    Format timestamp to a more readable format.

    Args:
        timestamp: ISO 8601 formatted timestamp

    Returns:
        Formatted timestamp string
    """
    try:
        # Parse the ISO 8601 timestamp
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Convert to local timezone
        local_dt = dt.astimezone()

        # Format as a more readable string
        return local_dt.strftime("%Y-%m-%d %I:%M %p %Z")
    except (ValueError, TypeError) as e:
        # Fallback to original timestamp if parsing fails
        st.warning(f"Could not parse timestamp: {timestamp}. Error: {e}")
        return timestamp


def load_experiments(model_type: str) -> List[Dict[str, Any]]:
    """
    Load experiments for a specific model type using ExperimentTracker.

    Args:
        model_type: Type of model to load experiments for

    Returns:
        List of experiment details
    """
    try:
        tracker = ExperimentTracker(model_type)
        experiments = tracker.list_experiments()

        # Minimal diagnostic logging
        st.sidebar.info(f"Loaded {len(experiments)} experiments for {model_type}")

        # Enrich experiments with additional details
        enriched_experiments = []
        for exp in experiments:
            try:
                # Try to load additional experiment details
                full_exp = tracker.load_experiment(exp["name"], exp["version"])

                # Combine listed and loaded experiment details
                enriched_exp = {
                    **exp,
                    "metrics": {},
                    "parameters": {},
                    "model_info": {},
                }

                # Load metrics
                for dataset in ["train", "tune", "test"]:
                    try:
                        enriched_exp["metrics"][dataset] = full_exp.get_metrics(dataset)
                    except ValueError:
                        pass

                # Load parameters and model info
                try:
                    enriched_exp["parameters"] = full_exp.get_parameters()
                    enriched_exp["model_info"] = full_exp.get_model_info()
                except ValueError:
                    pass

                enriched_experiments.append(enriched_exp)

            except Exception as e:
                st.sidebar.warning(
                    f"Could not fully load experiment {exp['name']}: {e}"
                )
                # Fallback to original experiment details
                enriched_experiments.append(exp)

        return enriched_experiments

    except Exception as e:
        st.sidebar.error(f"Error loading experiments: {e}")
        return []


def get_experiment_details(model_type: str, experiment_name: str, version: int):
    """
    Retrieve detailed information for a specific experiment.

    Args:
        model_type: Type of model
        experiment_name: Name of the experiment
        version: Version of the experiment

    Returns:
        Detailed experiment information
    """
    tracker = ExperimentTracker(model_type)
    experiment = tracker.load_experiment(experiment_name, version)

    # Collect comprehensive experiment details
    details = {
        "name": experiment.name,
        "version": version,
        "description": experiment.description,
        "timestamp": experiment.timestamp,
        "metadata": experiment.metadata,
        "metrics": {},
        "parameters": {},
        "model_info": {},
    }

    # Load metrics
    for dataset in ["train", "tune", "test"]:
        try:
            details["metrics"][dataset] = experiment.get_metrics(dataset)
        except ValueError:
            st.warning(f"No metrics found for {dataset} dataset")

    # Load parameters
    try:
        details["parameters"] = experiment.get_parameters()
    except ValueError:
        st.warning("No parameters found for experiment")

    # Load model info
    try:
        details["model_info"] = experiment.get_model_info()
    except ValueError:
        st.warning("No model info found for experiment")

    return details


def create_metrics_overview(experiments: List[Dict[str, Any]], selected_dataset: str):
    """
    Create a comprehensive metrics overview for a specific dataset.

    Args:
        experiments: List of experiments
        selected_dataset: Dataset to analyze (train/tune/test)

    Returns:
        Polars DataFrame with metrics
    """
    # Collect all unique metrics
    all_metrics = set()
    for exp in experiments:
        metrics = exp.get("metrics", {}).get(selected_dataset, {})
        all_metrics.update(metrics.keys())

    # Prepare metrics data
    metrics_data = []
    for exp in experiments:
        row = {
            "Experiment": exp["full_name"],
            "Timestamp": exp.get("timestamp", "Unknown"),
        }

        # Add metrics for the selected dataset
        dataset_metrics = exp.get("metrics", {}).get(selected_dataset, {})
        for metric in all_metrics:
            row[metric] = dataset_metrics.get(metric, "N/A")

        metrics_data.append(row)

    return pl.DataFrame(metrics_data)


def create_parameters_overview(experiments: List[Dict[str, Any]]):
    """
    Create a comprehensive parameters overview.

    Args:
        experiments: List of experiments

    Returns:
        Polars DataFrame with parameters
    """
    # Prepare parameters data
    params_data = []
    for exp in experiments:
        row = {
            "Experiment": exp["full_name"],
            "Timestamp": exp.get("timestamp", "Unknown"),
        }

        # Add parameters
        parameters = exp.get("parameters", {})
        for key, value in parameters.items():
            row[key] = str(value)

        # Add model info parameters
        model_info = exp.get("model_info", {})
        for key in ["n_features", "intercept", "threshold"]:
            if key in model_info:
                row[f"model_info.{key}"] = str(model_info[key])

        params_data.append(row)

    return pl.DataFrame(params_data)


def create_feature_importance_plot(
    experiment: Dict[str, Any],
    model_type: str,
    top_n: int = 20,
):
    """
    Create feature importance visualization for a single experiment.

    Args:
        experiment: Specific experiment details
        model_type: Type of model for experiment
        top_n: Number of top features to display (default: 20)

    Returns:
        Plotly figure or None
    """
    # Construct coefficient file path with more robust method
    exp_name = experiment.get("name", experiment.get("full_name", "unknown"))
    exp_version = experiment.get("version", 1)

    # Construct the specific paths for both coefficient and feature importance files
    coef_paths = [
        Path(
            f"models/experiments/{model_type}/{exp_name}/v{exp_version}/coefficients.csv"
        ),
        Path(f"models/experiments/{exp_name}/v{exp_version}/coefficients.csv"),
        Path(f"models/experiments/{exp_name}/coefficients.csv"),
        Path(f"models/experiments/{exp_name}/v{exp_version}/metadata/coefficients.csv"),
    ]

    feature_importance_paths = [
        Path(
            f"models/experiments/{model_type}/{exp_name}/v{exp_version}/feature_importance.csv"
        ),
        Path(f"models/experiments/{exp_name}/v{exp_version}/feature_importance.csv"),
        Path(f"models/experiments/{exp_name}/feature_importance.csv"),
        Path(
            f"models/experiments/{exp_name}/v{exp_version}/metadata/feature_importance.csv"
        ),
    ]

    # Find the first existing path
    coef_path = next((path for path in coef_paths if path.exists()), None)
    feature_importance_path = next(
        (path for path in feature_importance_paths if path.exists()), None
    )

    # Abbreviate feature names
    def abbreviate_feature(feature, n=40):
        # Truncate to maximum n characters
        if len(feature) > n:
            return feature[: n - 3] + "..."
        return feature

    try:
        # Prefer coefficient file if it exists
        if coef_path:
            coef_df = pl.read_csv(coef_path)

            # Sort and select top N features by absolute coefficient value
            df_sorted = (
                coef_df.with_columns(
                    [
                        pl.col("coefficient").abs().alias("abs_coefficient"),
                        pl.Series(
                            name="abbreviated_feature",
                            values=[
                                abbreviate_feature(f)
                                for f in coef_df.get_column("feature")
                            ],
                        ),
                    ]
                )
                .sort("abs_coefficient", descending=True)
                .head(top_n)
                .sort(
                    "coefficient"
                )  # Sort by signed coefficient for correct y-axis order
            )

            # Create horizontal bar plot
            fig = px.bar(
                df_sorted.to_pandas(),
                y="abbreviated_feature",
                x="coefficient",
                orientation="h",
                color="coefficient",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )

            # Enhanced layout
            fig.update_layout(
                height=800,
                width=1000,
                title=f"Top {top_n} Features {exp_name} (Coefficients)",
                yaxis_title="Feature",
                xaxis_title="Effect",
                title_x=0.5,
            )

            # Remove color axis label
            fig.update_layout(coloraxis_colorbar=dict(title=""))

            # Ensure zero line is visible
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")

            # Add hover text with full feature names and remove bar text
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Full Name: %{text}<br>Coefficient: %{x:.4f}<extra></extra>",
                text=df_sorted.get_column("feature"),
                texttemplate="",
                textposition="none",
            )

            return fig

        # If no coefficient file, try feature importance file
        elif feature_importance_path:
            feature_df = pl.read_csv(feature_importance_path)

            # Diagnostic logging of columns
            # Flexible column name detection
            feature_col = None
            importance_col = None

            # Common column name variations
            feature_candidates = [
                "feature",
                "features",
                "feature_name",
                "name",
                "column",
            ]
            importance_candidates = [
                "importance",
                "feature_importance",
                "importance_score",
                "score",
                "value",
            ]

            # Find first matching column names
            for candidate in feature_candidates:
                if candidate in feature_df.columns:
                    feature_col = candidate
                    break

            for candidate in importance_candidates:
                if candidate in feature_df.columns:
                    importance_col = candidate
                    break

            # Validate column detection
            if not feature_col or not importance_col:
                st.warning(
                    f"Could not find feature and importance columns in {feature_importance_path}"
                )
                st.warning(f"Available columns: {feature_df.columns}")
                return None

            # Sort and select top N features by importance
            df_sorted = (
                feature_df.with_columns(
                    [
                        pl.Series(
                            name="abbreviated_feature",
                            values=[
                                abbreviate_feature(str(f))
                                for f in feature_df.get_column(feature_col)
                            ],
                        )
                    ]
                )
                .sort(importance_col, descending=True)
                .head(top_n)
                .sort(
                    importance_col, descending=False
                )  # Reverse order to have most important at top
            )

            # Create horizontal bar plot for feature importance
            fig = px.bar(
                df_sorted.to_pandas(),
                y="abbreviated_feature",
                x=importance_col,
                orientation="h",
                color=importance_col,
                color_continuous_scale="Viridis",  # Different color scale to distinguish from coefficient plot
            )

            # Enhanced layout
            fig.update_layout(
                height=800,
                width=1000,
                title=f"Top {top_n} Features {exp_name} (Feature Importance)",
                yaxis_title="Feature",
                xaxis_title="Importance",
                title_x=0.5,
            )

            # Remove color axis label
            fig.update_layout(coloraxis_colorbar=dict(title=""))

            # Add hover text with full feature names
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Full Name: %{text}<br>Importance: %{x:.4f}<extra></extra>",
                text=df_sorted.get_column(feature_col),
                texttemplate="",
                textposition="none",
            )

            return fig

        else:
            st.warning(
                f"No coefficient or feature importance file found for experiment {exp_name}"
            )
            return None

    except Exception as e:
        st.warning(f"Error processing feature importance for {exp_name}: {e}")
        return None


def display_predictions(
    experiment, dataset: str, model_type: str, selected_model_type: str
):
    """
    Display predictions for a specific dataset with comprehensive analysis.

    Args:
        experiment: Experiment object
        dataset: Dataset to display predictions for ('tune' or 'test')
        model_type: Type of model ('regression' or 'classification')
        selected_model_type: Model type selected in the sidebar

    Returns:
        Polars DataFrame with predictions
    """
    # Check if predictions exist
    try:
        predictions_df = experiment.get_predictions(dataset)
    except Exception as e:
        st.warning(f"Could not retrieve predictions: {e}")
        st.info("Possible reasons:")
        st.info("1. No predictions were saved for this experiment")
        st.info("2. Predictions file might be corrupted")
        st.info("3. Experiment tracking might not have logged predictions")
        return None

    # Validate predictions DataFrame
    if predictions_df is None or len(predictions_df) == 0:
        st.warning(f"No predictions found for {dataset} dataset")
        st.info("This could mean:")
        st.info("1. The model did not generate predictions")
        st.info("2. Predictions were not saved during experiment tracking")
        return None

    # Special handling for geek_rating model type
    if selected_model_type == "geek_rating":
        # # Diagnostic logging
        # st.warning(f"Geek Rating Model: Initial DataFrame Null Check")
        # st.warning(f"DataFrame columns: {predictions_df.columns}")

        # Check if 'actual' column exists
        if "actual" not in predictions_df.columns:
            st.error("No 'actual' column found in the DataFrame")
            st.warning(f"Available columns: {predictions_df.columns}")
            st.warning(f"DataFrame schema: {predictions_df.schema}")
            return None

        # Detailed null check
        null_mask = predictions_df["actual"].is_null()
        # st.warning(f"Null values in 'actual' column: {null_mask.sum()}")

        # If there are null values, show some examples
        if null_mask.sum() > 0:
            null_rows = predictions_df.filter(null_mask)
            st.warning("Sample of rows with null 'actual' values:")
            st.dataframe(null_rows)

        try:
            # Replace only non-numeric values with 5.5
            predictions_df = predictions_df.with_columns(
                [
                    pl.when(~pl.col("actual").cast(pl.Float64).is_finite())
                    .then(pl.lit(5.5))
                    .otherwise(pl.col("actual"))
                    .alias("actual")
                ]
            )
            st.success("Set non-numeric 'actual' values to 5.5 for geek_rating model")
        except Exception as e:
            st.error(f"Error replacing null values: {e}")
            return None

    # Select relevant columns for display
    display_columns = [
        col
        for col in predictions_df.columns
        if col not in ["prediction", "actual", "threshold"]
    ]

    # Prepare DataFrame for display
    display_df = predictions_df.select(["prediction", "actual"] + display_columns)

    # Performance metrics and visualization based on model type
    try:
        # Check if users_rated column exists and add filter if it does
        if "users_rated" in display_df.columns:
            # Add users_rated filter above the plot
            # st.header("Filter Predictions")
            min_users_rated_filter = st.slider(
                "Minimum Number of Users Rated",
                min_value=0,
                max_value=100,
                value=5,
                step=5,
            )

            # Calculate the actual minimum users_rated based on the percentage
            actual_min_users_rated = int(
                display_df["users_rated"].quantile(min_users_rated_filter / 100)
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

                # Add year filter
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

            # # Display filter info
            # st.info(
            #     f"Showing games with at least {actual_min_users_rated} users rated (percentile {min_users_rated_filter}%)"
            # )

        # Add filter for geek_rating model type to exclude games with actual = 5.5
        if selected_model_type == "geek_rating":
            filter_5_5 = st.checkbox(
                "Filter out games where actual rating = 5.5", value=False
            )
            if filter_5_5:
                display_df = display_df.filter(pl.col("actual") != 5.5)
                st.info("Filtered out games where actual rating = 5.5")

        if model_type == "regression":
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
                mean_absolute_percentage_error,
            )
            import plotly.express as px
            import plotly.graph_objs as go

            # Calculate metrics
            mse = mean_squared_error(display_df["actual"], display_df["prediction"])
            mae = mean_absolute_error(display_df["actual"], display_df["prediction"])
            r2 = r2_score(display_df["actual"], display_df["prediction"])
            mape = mean_absolute_percentage_error(
                display_df["actual"], display_df["prediction"]
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

            # Convert to pandas for easier manipulation
            df_pandas = display_df.to_pandas()

            # Prepare hover text with game_id and name if available
            hover_text = []
            for index, row in df_pandas.iterrows():
                hover_info = (
                    f"Predicted: {row['prediction']:.4f}<br>Actual: {row['actual']:.4f}"
                )

                # Add game_id and name if they exist in the columns
                if "game_id" in df_pandas.columns:
                    hover_info += f"<br>Game ID: {row['game_id']}"
                if "name" in df_pandas.columns:
                    hover_info += f"<br>Name: {row['name']}"

                hover_text.append(hover_info)

            # Scatter plot of predicted vs actual with hover information
            fig = go.Figure()

            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=df_pandas["prediction"],
                    y=df_pandas["actual"],
                    mode="markers",
                    marker=dict(
                        size=5,  # Reduced marker size
                        color="blue",  # Consistent color
                        opacity=0.7,  # Slight transparency
                        line=dict(width=0.5, color="darkblue"),  # Subtle border
                    ),
                    text=hover_text,
                    hoverinfo="text",
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,  # Remove from legend
                )
            )

            # Add perfect prediction line
            fig.add_trace(
                go.Scatter(
                    x=[df_pandas["prediction"].min(), df_pandas["prediction"].max()],
                    y=[df_pandas["prediction"].min(), df_pandas["prediction"].max()],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(color="red", dash="dash"),
                    hoverinfo="none",
                )
            )

            # Add LOESS trend line using statsmodels
            import numpy as np
            import statsmodels.nonparametric.smoothers_lowess as lowess

            # Sort data for LOESS smoothing
            sorted_indices = np.argsort(df_pandas["prediction"])
            x_sorted = df_pandas["prediction"].iloc[sorted_indices]
            y_sorted = df_pandas["actual"].iloc[sorted_indices]

            # Apply LOESS smoothing
            loess_smoothed = lowess.lowess(y_sorted, x_sorted, frac=2 / 3, it=5)

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

            # Update layout
            fig.update_layout(
                title="Predicted vs Actual Values",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                hovermode="closest",
            )

            st.plotly_chart(fig)

        elif model_type == "classification":
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
                roc_auc_score,
            )
            import plotly.express as px
            import numpy as np

            # Calculate metrics
            accuracy = accuracy_score(display_df["actual"], display_df["prediction"])
            precision = precision_score(display_df["actual"], display_df["prediction"])
            recall = recall_score(display_df["actual"], display_df["prediction"])
            f1 = f1_score(display_df["actual"], display_df["prediction"])
            roc_auc = roc_auc_score(display_df["actual"], display_df["prediction"])

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
            cm = confusion_matrix(display_df["actual"], display_df["prediction"])
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Negative", "Positive"],
                y=["Negative", "Positive"],
                title="Confusion Matrix",
            )
            st.plotly_chart(fig)

        return display_df

    except Exception as e:
        st.error(f"Error processing predictions: {e}")
        st.info("Unable to generate visualizations or metrics")
        return display_df


def main():
    """Streamlit app main function."""
    st.title("Experiment Tracking Dashboard")

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
        return

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

        # Dataset selector
        selected_dataset = st.selectbox("Select Dataset", ["train", "tune", "test"])

        # Metrics overview
        metrics_df = create_metrics_overview(experiments, selected_dataset)
        st.dataframe(metrics_df)

    with tab2:
        st.header("Model Predictions")

        # Check if experiments exist
        if not experiments:
            st.error("No experiments available. Please track some experiments first.")
            return

        # Validate experiments list
        if len(experiments) == 0:
            st.warning("No experiments found. Have you run any model training?")
            st.info("To see predictions:")
            st.info("1. Train a model using the experiment tracking system")
            st.info("2. Ensure predictions are logged during training")
            return

        try:
            # Select specific experiment
            selected_experiment = st.selectbox(
                "Select Experiment",
                [exp["full_name"] for exp in experiments],
                key="predictions_experiment",
            )

            # Parse experiment name and version
            exp_name, exp_version = selected_experiment.split("/")
            exp_version = int(exp_version.replace("v", ""))

            # Load experiment
            tracker = ExperimentTracker(selected_model_type)
            experiment = tracker.load_experiment(exp_name, exp_version)

            # Determine model type from experiment metadata
            try:
                # Retrieve full metadata
                full_metadata = experiment.metadata

                # Try to get model type from metadata
                model_type = full_metadata.get(
                    "model_type",
                    experiment.get_model_info().get("model_type", "regression"),
                )
            except Exception as e:
                st.warning(f"Error determining model type: {e}")
                st.warning("Defaulting to regression model type")
                model_type = "regression"

            # Dataset selector
            selected_dataset = st.selectbox(
                "Select Dataset", ["tune", "test"], key="predictions_dataset"
            )

            # Validate dataset selection
            if selected_dataset not in ["tune", "test"]:
                st.error(f"Invalid dataset: {selected_dataset}")
                return

            # # Special handling for geek_rating model type
            # if selected_model_type == "geek_rating":
            #     st.warning("Geek Rating Model: Setting 'actual' to 5.5")

            # Display predictions
            predictions_df = display_predictions(
                experiment, selected_dataset, model_type, selected_model_type
            )

            if predictions_df is not None:
                st.dataframe(predictions_df)
            else:
                st.warning(f"No predictions available for {selected_dataset} dataset")
                st.info("Possible reasons:")
                st.info("1. Predictions were not saved during training")
                st.info("2. The selected dataset might be empty")
                st.info("3. Experiment tracking did not log predictions")

        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            st.info("Troubleshooting tips:")
            st.info("1. Verify the experiment was tracked correctly")
            st.info("2. Check that predictions were logged during training")
            st.info("3. Ensure the experiment file is not corrupted")

    with tab3:
        st.header("Model Parameters")
        params_df = create_parameters_overview(experiments)
        st.dataframe(params_df)

    with tab4:
        st.header("Feature Importance")

        # Select specific experiment
        selected_experiment = st.selectbox(
            "Select Experiment",
            [exp["full_name"] for exp in experiments],
            key="feature_importance_experiment",
        )

        # Parse experiment name and version
        exp_name, exp_version = selected_experiment.split("/")
        exp_version = int(exp_version.replace("v", ""))

        # Find the selected experiment
        selected_exp = next(
            (exp for exp in experiments if exp["full_name"] == selected_experiment),
            None,
        )

        # Feature importance configuration
        top_n = st.slider("Top N Features", min_value=5, max_value=50, value=30, step=5)

        if selected_exp:
            feature_fig = create_feature_importance_plot(
                selected_exp,
                model_type=selected_model_type,
                top_n=top_n,
            )

            if feature_fig:
                st.plotly_chart(feature_fig)
            else:
                st.write("No coefficient data available for this experiment")
        else:
            st.error("Could not find the selected experiment")

    with tab5:
        st.header("Experiment Details")

        # Select specific experiment
        selected_experiment = st.selectbox(
            "Select Experiment",
            [exp["full_name"] for exp in experiments],
            key="details_experiment",
        )

        # Parse experiment name and version
        exp_name, exp_version = selected_experiment.split("/")
        exp_version = int(exp_version.replace("v", ""))

        # Load and display experiment details
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


if __name__ == "__main__":
    main()
