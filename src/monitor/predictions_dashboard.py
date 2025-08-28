"""
Streamlit dashboard for exploring board game predictions.

This dashboard loads the latest predictions from BigQuery and provides interactive
visualizations for exploring predicted geek ratings, complexity, ratings, and other metrics.
"""

import sys
import os
from dotenv import load_dotenv
import streamlit as st
import polars as pl
import plotly.express as px
from datetime import datetime
import pandas as pd

load_dotenv()

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from src.data.bigquery_uploader import BigQueryUploader  # noqa: E402


def load_latest_predictions() -> pd.DataFrame:
    """
    Load the most recent predictions from BigQuery.

    Returns:
        DataFrame with latest predictions
    """
    try:
        uploader = BigQueryUploader(environment="dev")

        # Get the most recent job
        jobs = uploader.get_prediction_summary()
        if len(jobs) == 0:
            st.error("No prediction jobs found in BigQuery.")
            return pd.DataFrame()

        # Get the latest job
        latest_job = jobs.iloc[0]  # Already sorted by latest_prediction DESC

        # Load predictions for the latest job
        df = uploader.query_predictions(job_id=latest_job["job_id"])

        return df
    except Exception as e:
        st.error(f"Error loading predictions from BigQuery: {e}")
        return pd.DataFrame()


def main():
    st.set_page_config(page_title="Predictions Explorer", layout="wide")

    st.title("Board Game Predictions")

    # Load latest predictions
    with st.spinner("Loading latest predictions from BigQuery..."):
        df = load_latest_predictions()

    if df.empty:
        st.error("No predictions data available.")
        return

    # Basic info
    st.sidebar.header("Data Info")
    st.sidebar.text(f"Total Predictions: {len(df):,}")
    if "score_ts" in df.columns:
        latest_update = pd.to_datetime(df["score_ts"]).max()
        st.sidebar.text(f"Last Updated: {latest_update.strftime('%Y-%m-%d %H:%M')}")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(
        ["Predictions Table", "Geek Rating Distribution", "Analysis"]
    )

    with tab1:
        # Prepare year buckets
        df["year_bucket"] = df["year_published"].apply(
            lambda x: "Other" if x < 2010 else str(x)
        )

        # rename columns if they exist
        if "predicted_geek_rating" not in df.columns:
            if "geek_rating" in df.columns:
                df["predicted_geek_rating"] = df["geek_rating"]
            else:
                st.error("Could not find predicted geek rating column")
                return

        if "actual_geek_rating" not in df.columns and "actual" in df.columns:
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


if __name__ == "__main__":
    main()
