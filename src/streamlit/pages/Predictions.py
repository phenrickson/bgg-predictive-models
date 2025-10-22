"""
Streamlit dashboard for exploring board game predictions.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.monitor.predictions_dashboard import (
    get_prediction_jobs,
    load_predictions_for_job,
    load_predictions_for_selected_job,
)

st.set_page_config(page_title="Predictions | BGG Models Dashboard", layout="wide")
st.title("Board Game Predictions")

# Load available jobs
with st.spinner("Loading available prediction jobs..."):
    jobs_df = get_prediction_jobs()

if jobs_df.empty:
    st.error("No prediction jobs found in BigQuery.")
    st.stop()

# Job selection in sidebar
st.sidebar.header("Job Selection")
job_options = []
for _, job in jobs_df.iterrows():
    latest_pred = pd.to_datetime(job["latest_prediction"]).strftime("%Y-%m-%d %H:%M")
    option_text = f"{latest_pred} ({job['num_predictions']} predictions)"
    job_options.append((job["job_id"], option_text))

selected_job_id = st.sidebar.selectbox(
    "Select Prediction Job",
    options=[j[0] for j in job_options],
    format_func=lambda x: next(j[1] for j in job_options if j[0] == x),
    index=0,
)

# Show selected job details
selected_job = jobs_df[jobs_df["job_id"] == selected_job_id].iloc[0]
st.sidebar.markdown("**Selected Job Details:**")
st.sidebar.markdown(
    f"""
    - **Job ID**: `{selected_job_id[:8]}...`
    - **Predictions**: {selected_job["num_predictions"]:,}
    - **Date**: {pd.to_datetime(selected_job["latest_prediction"]).strftime("%Y-%m-%d %H:%M")}
    - **Hurdle Model**: `{selected_job["hurdle_experiment"]}`
    - **Complexity Model**: `{selected_job["complexity_experiment"]}`
    - **Rating Model**: `{selected_job["rating_experiment"]}`
    - **Users Rated Model**: `{selected_job["users_rated_experiment"]}`
    """
)

# Load predictions for selected job
with st.spinner("Loading predictions for selected job..."):
    df = load_predictions_for_selected_job(selected_job_id)

if df.empty:
    st.error("No predictions data available for the selected job.")
    st.stop()

# Cache controls
st.sidebar.header("Cache Controls")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(
    ["Predictions Table", "Geek Rating Distribution", "Analysis", "BigQuery Jobs"]
)

with tab1:
    # Year bucketing
    df["year_bucket"] = df["year_published"].apply(
        lambda x: "Other" if x < 2010 else str(x)
    )

    # Column renaming
    if "predicted_geek_rating" not in df.columns and "geek_rating" in df.columns:
        df["predicted_geek_rating"] = df["geek_rating"]
    if "actual_geek_rating" not in df.columns and "actual" in df.columns:
        df["actual_geek_rating"] = df["actual"]

    # Year selection
    unique_years = sorted(df["year_bucket"].unique(), key=lambda x: (x == "Other", x))
    current_year = datetime.now().year
    numeric_years = [int(year) for year in unique_years if year != "Other"]
    closest_year = min(numeric_years, key=lambda x: abs(x - current_year))
    default_year_index = unique_years.index(str(closest_year))
    selected_year = st.selectbox(
        "Select Publication Year", unique_years, index=default_year_index
    )

    # Filter and sort data
    filtered_df = df[df["year_bucket"] == selected_year].sort_values(
        "predicted_geek_rating", ascending=False
    )

    # Display statistics
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

    # Prepare display dataframe
    display_df = filtered_df.copy()
    display_df["bgg_link"] = display_df.apply(
        lambda row: f"https://boardgamegeek.com/boardgame/{row['game_id']}", axis=1
    )

    # Define display columns
    display_columns = [
        "year_published",
        "game_id",
        "bgg_link",
        "name",
        "actual_geek_rating",
        "predicted_geek_rating",
        "predicted_hurdle_prob",
        "predicted_complexity",
        "predicted_rating",
        "predicted_users_rated",
    ]
    display_columns = [col for col in display_columns if col in display_df.columns]

    # Display table
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True,
        height=800,
        column_config={
            "bgg_link": st.column_config.LinkColumn("BoardGameGeek", display_text="BGG")
        },
    )

with tab2:
    st.header("Geek Rating Distribution")
    fig = px.histogram(
        filtered_df,
        x="predicted_geek_rating",
        title=f"Distribution of Geek Ratings for {selected_year}",
        labels={"geek_rating": "Geek Rating"},
        marginal="box",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Analysis")
    x_axis_options = [
        "predicted_complexity",
        "predicted_hurdle_prob",
        "predicted_users_rated",
        "predicted_rating",
    ]
    x_axis = st.selectbox("Select X-Axis Variable", x_axis_options, index=0)
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
    if jobs_df.empty:
        st.warning("No BigQuery jobs found.")
    else:
        st.subheader("Prediction Job History")
        display_jobs = jobs_df.copy()

        # Format timestamps
        if "latest_prediction" in display_jobs.columns:
            display_jobs["latest_prediction"] = pd.to_datetime(
                display_jobs["latest_prediction"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "earliest_prediction" in display_jobs.columns:
            display_jobs["earliest_prediction"] = pd.to_datetime(
                display_jobs["earliest_prediction"]
            ).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Round numeric columns
        numeric_columns = ["avg_predicted_rating"]
        for col in numeric_columns:
            if col in display_jobs.columns:
                display_jobs[col] = display_jobs[col].round(3)

        # Display jobs table
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
                "latest_prediction": st.column_config.TextColumn("Latest Prediction"),
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
