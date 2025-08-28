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
import hashlib

load_dotenv()

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
from src.data.bigquery_uploader import BigQueryUploader  # noqa: E402


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_prediction_jobs() -> pd.DataFrame:
    """
    Get prediction jobs summary from BigQuery with caching.

    Returns:
        DataFrame with job information
    """
    try:
        uploader = BigQueryUploader(environment="dev")
        jobs = uploader.get_prediction_summary()
        return jobs
    except Exception as e:
        st.error(f"Error loading prediction jobs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_predictions_for_job(job_id: str) -> pd.DataFrame:
    """
    Load predictions for a specific job with caching.

    Args:
        job_id: The job ID to load predictions for

    Returns:
        DataFrame with predictions
    """
    try:
        uploader = BigQueryUploader(environment="dev")
        df = uploader.query_predictions(job_id=job_id)
        return df
    except Exception as e:
        st.error(f"Error loading predictions for job {job_id}: {e}")
        return pd.DataFrame()


def load_latest_predictions() -> pd.DataFrame:
    """
    Load the most recent predictions from BigQuery with caching.

    Returns:
        DataFrame with latest predictions
    """
    # Get jobs (cached)
    jobs = get_prediction_jobs()

    if len(jobs) == 0:
        st.error("No prediction jobs found in BigQuery.")
        return pd.DataFrame()

    # Get the latest job
    latest_job = jobs.iloc[0]  # Already sorted by latest_prediction DESC

    # Load predictions for the latest job (cached)
    df = load_predictions_for_job(latest_job["job_id"])

    return df


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

    # Cache controls
    st.sidebar.header("Cache Controls")
    if st.sidebar.button("🔄 Refresh Data", help="Clear cache and reload data"):
        st.cache_data.clear()
        st.rerun()

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Predictions Table", "Geek Rating Distribution", "Analysis", "BigQuery Jobs"]
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

    with tab4:
        st.header("BigQuery Jobs")

        # Load jobs data (cached)
        jobs_df = get_prediction_jobs()

        if jobs_df.empty:
            st.warning("No BigQuery jobs found.")
        else:
            st.subheader("Prediction Job History")

            # Format the jobs dataframe for display
            display_jobs = jobs_df.copy()

            # Convert timestamps to readable format
            if "latest_prediction" in display_jobs.columns:
                display_jobs["latest_prediction"] = pd.to_datetime(
                    display_jobs["latest_prediction"]
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

            if "earliest_prediction" in display_jobs.columns:
                display_jobs["earliest_prediction"] = pd.to_datetime(
                    display_jobs["earliest_prediction"]
                ).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Round numeric columns for better display
            numeric_columns = ["avg_predicted_rating"]
            for col in numeric_columns:
                if col in display_jobs.columns:
                    display_jobs[col] = display_jobs[col].round(3)

            # Display the jobs table
            st.dataframe(
                display_jobs,
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    "job_id": st.column_config.TextColumn("Job ID", width="medium"),
                    "num_predictions": st.column_config.NumberColumn(
                        "# Predictions", format="%d"
                    ),
                    "latest_prediction": st.column_config.TextColumn(
                        "Latest Prediction"
                    ),
                    "earliest_prediction": st.column_config.TextColumn(
                        "Earliest Prediction"
                    ),
                    "min_year": st.column_config.NumberColumn("Min Year", format="%d"),
                    "max_year": st.column_config.NumberColumn("Max Year", format="%d"),
                    "avg_predicted_rating": st.column_config.NumberColumn(
                        "Avg Rating", format="%.3f"
                    ),
                    "hurdle_experiment": st.column_config.TextColumn("Hurdle Model"),
                    "complexity_experiment": st.column_config.TextColumn(
                        "Complexity Model"
                    ),
                    "rating_experiment": st.column_config.TextColumn("Rating Model"),
                    "users_rated_experiment": st.column_config.TextColumn(
                        "Users Rated Model"
                    ),
                },
            )

            # Job statistics
            st.subheader("Job Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Jobs", len(jobs_df))

            with col2:
                total_predictions = jobs_df["num_predictions"].sum()
                st.metric("Total Predictions", f"{total_predictions:,}")

            with col3:
                if "latest_prediction" in jobs_df.columns:
                    latest_job_time = pd.to_datetime(jobs_df["latest_prediction"]).max()
                    st.metric("Latest Job", latest_job_time.strftime("%Y-%m-%d"))

            with col4:
                if "avg_predicted_rating" in jobs_df.columns:
                    overall_avg = jobs_df["avg_predicted_rating"].mean()
                    st.metric("Overall Avg Rating", f"{overall_avg:.3f}")


if __name__ == "__main__":
    main()
