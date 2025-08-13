"""
Streamlit dashboard for exploring geek rating predictions.
"""

import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path
import os
import sys
from typing import Optional
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.models.experiments import ExperimentTracker


def load_predictions(
    experiment_name: str, version: Optional[int] = None
) -> pl.DataFrame:
    """
    Load predictions for a specific geek rating experiment.

    Args:
        experiment_name: Name of the experiment to load predictions for
        version: Optional version of the experiment

    Returns:
        Polars DataFrame with experiment predictions
    """
    # First, try to load from experiment tracker
    tracker = ExperimentTracker(model_type="geek_rating")

    # Debug: List all available experiments
    experiments = tracker.list_experiments()
    st.sidebar.subheader("Available Experiments:")
    for exp in experiments:
        st.sidebar.text(f"{exp['name']} (v{exp['version']})")

    try:
        # If version is not specified, get the latest version
        if version is None:
            versions = [
                exp["version"] for exp in experiments if exp["name"] == experiment_name
            ]
            version = max(versions) if versions else None

        if version is not None:
            experiment = tracker.load_experiment(experiment_name, version)
            predictions_path = experiment.exp_dir / "predictions.parquet"

            if predictions_path.exists():
                return pl.read_parquet(str(predictions_path))
    except Exception as e:
        st.warning(f"Could not load experiment predictions: {e}")

    # Fallback to prediction directory paths
    prediction_paths = [
        # Check in models/experiments directory first
        Path(
            f"models/experiments/geek_rating/{experiment_name}/v{version}/test_predictions.parquet"
        ),
        Path(
            f"models/experiments/geek_rating/{experiment_name}/test_predictions.parquet"
        ),
    ]

    # Try each potential path
    for predictions_path in prediction_paths:
        st.sidebar.text(f"Checking path: {predictions_path}")
        if predictions_path.exists():
            try:
                return pl.read_parquet(str(predictions_path))
            except Exception as e:
                st.error(f"Error reading {predictions_path}: {e}")

    st.error(f"No predictions found for experiment: {experiment_name}")
    return None


def list_geek_rating_experiments() -> list:
    """
    List all geek rating experiments.

    Returns:
        List of experiment names and their versions
    """
    tracker = ExperimentTracker(model_type="geek_rating")
    experiments = tracker.list_experiments()

    # Create a list of experiment names with versions
    experiment_list = [f"{exp['name']} (v{exp['version']})" for exp in experiments]

    # Also list Parquet files in the predictions directory
    predictions_dir = Path("data/predictions/geek_rating")
    if predictions_dir.exists():
        parquet_files = [f.stem for f in predictions_dir.glob("*.parquet")]

        # Combine and deduplicate experiment names
        experiment_list.extend(parquet_files)
        experiment_list = list(
            dict.fromkeys(experiment_list)
        )  # Remove duplicates while preserving order

    return sorted(experiment_list)


def main():
    st.set_page_config(page_title="Geek Rating Predictions Explorer", layout="wide")

    st.title("Geek Rating Predictions Explorer")

    # Sidebar for experiment selection
    st.sidebar.header("Experiment Selection")

    # List available experiments
    experiments = list_geek_rating_experiments()
    selected_experiment_full = st.sidebar.selectbox(
        "Select Experiment to Explore", experiments
    )

    # Parse experiment name and version
    if " (v" in selected_experiment_full:
        selected_experiment, version_str = selected_experiment_full.split(" (v")
        version = int(version_str[:-1])  # Remove closing parenthesis
    else:
        selected_experiment = selected_experiment_full
        version = None

    # Load predictions for selected experiment
    predictions = load_predictions(selected_experiment)

    if predictions is None:
        st.error("Could not load predictions. Please select another experiment.")
        return

    # Convert to pandas for easier manipulation
    df = predictions.to_pandas()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["Predictions Table", "Geek Rating Distribution", "Analysis"]
    )

    with tab1:
        # Prepare year buckets
        df["year_bucket"] = df["year_published"].apply(
            lambda x: "Other" if x < 2010 else str(x)
        )

        # rename
        df["predicted_geek_rating"] = df["prediction"]
        df["actual_geek_rating"] = df["actual"]

        # Year selection
        unique_years = sorted(
            df["year_bucket"].unique(), key=lambda x: (x == "Other", x)
        )

        # Find the year closest to the current year, excluding "Other"
        current_year = datetime.now().year
        numeric_years = [int(year) for year in unique_years if year != "Other"]
        closest_year = min(numeric_years, key=lambda x: abs(x - current_year))

        default_year_index = unique_years.index(str(closest_year))
        selected_year = st.selectbox(
            "Select Publication Year", unique_years, index=default_year_index
        )

        # Filter dataframe
        filtered_df = df[df["year_bucket"] == selected_year]

        # Sort by geek_rating in descending order
        filtered_df = filtered_df.sort_values("predicted_geek_rating", ascending=False)

        # Basic statistics
        st.header("Prediction Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Games", len(filtered_df))

        with col2:
            st.metric(
                "Average Geek Rating",
                f"{filtered_df['predicted_geek_rating'].mean():.2f}",
            )

        with col3:
            st.metric(
                "Median Geek Rating",
                f"{filtered_df['predicted_geek_rating'].median():.2f}",
            )

        # st.header("Predictions Table")

        # Columns to display
        display_columns = [
            "year_published",
            "game_id",
            "name",
            "actual_geek_rating",
            "predicted_geek_rating",
            "predicted_hurdle_prob",
            "predicted_complexity",
            "predicted_rating",
            "predicted_users_rated",
        ]

        # Ensure all columns exist
        display_columns = [col for col in display_columns if col in filtered_df.columns]

        # Create a copy of the dataframe for display
        display_df = filtered_df.copy()

        # Add BoardGameGeek link column
        display_df["bgg_link"] = display_df.apply(
            lambda row: f"https://boardgamegeek.com/boardgame/{row['game_id']}", axis=1
        )

        # Update display columns to include the new link column
        display_columns.append("bgg_link")

        # Display interactive table
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            hide_index=True,
            height=800,  # Increase default height to show more rows
            column_config={
                "bgg_link": st.column_config.LinkColumn(
                    "BoardGameGeek", display_text="BGG"
                )
            },
        )

    with tab2:
        st.header("Geek Rating Distribution")

        # Histogram of geek ratings for selected year
        fig = px.histogram(
            filtered_df,
            x="predicted_geek_rating",
            title=f"Distribution of Geek Ratings for {selected_year}",
            labels={"geek_rating": "Geek Rating"},
            marginal="box",  # Add box plot
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Analysis")

        # Select x-axis variable
        x_axis_options = [
            "predicted_complexity",
            "predicted_hurdle_prob",
            "predicted_users_rated",
            "predicted_rating",
        ]
        x_axis = st.selectbox("Select X-Axis Variable", x_axis_options, index=0)

        # Scatter plot with dynamic x-axis
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y="predicted_geek_rating",
            color="predicted_hurdle_prob",
            title=f"{x_axis.replace('_', ' ').title()} vs Geek Rating for {selected_year}",
            labels={
                x_axis: x_axis.replace("_", " ").title(),
                "predicted_geek_rating": "Geek Rating",
                "predicted_complexity": "Complexity",
                "predicted_rating": "Rating",
                "predicted_users_rated": "Users Rated",
                "predicted_hurdle_prob": "Likelihood of Rating",
            },
            hover_data=["name", "year_published"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # Experiment Metadata in Sidebar
    st.sidebar.header("Experiment Metadata")

    # Load experiment metadata
    tracker = ExperimentTracker(model_type="geek_rating")
    try:
        experiment = tracker.load_experiment(selected_experiment)
        metadata = experiment.metadata

        # Display key metadata
        if "model_experiments" in metadata:
            st.sidebar.subheader("Model Experiments")
            for model, model_exp in metadata["model_experiments"].items():
                st.sidebar.text(f"{model.capitalize()}: {model_exp}")

        if "prediction_parameters" in metadata:
            st.sidebar.subheader("Prediction Parameters")
            for param, value in metadata["prediction_parameters"].items():
                st.sidebar.text(f"{param.replace('_', ' ').title()}: {value}")

    except Exception as e:
        st.sidebar.warning(f"Could not load experiment metadata: {e}")


if __name__ == "__main__":
    main()
