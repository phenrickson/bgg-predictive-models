"""
Multi-page Streamlit dashboard combining experiment tracking and predictions exploration.
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
from src.monitor.predictions_dashboard import (
    get_prediction_jobs,
    load_predictions_for_job,
    load_predictions_for_selected_job,
)


def main():
    """Main function for the multi-page dashboard."""
    st.set_page_config(page_title="BGG Models Dashboard", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "", ["Experiment Tracking", "Predictions"], label_visibility="collapsed"
    )

    if page == "Experiment Tracking":
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
            st.error(
                "No experiments found. Please ensure experiments have been tracked."
            )
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
            selected_dataset = st.selectbox("Select Dataset", ["train", "tune", "test"])
            metrics_df = create_metrics_overview(experiments, selected_dataset)
            st.subheader("Metrics Table")
            st.dataframe(metrics_df)

            st.subheader("Performance by Model Run")
            if len(metrics_df) > 0:
                available_metrics = [
                    col
                    for col in metrics_df.columns
                    if col not in ["Experiment", "Timestamp"]
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
            predictions_df = display_experiment_predictions(
                experiments[0], selected_dataset, "regression", selected_model_type
            )
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
                        for i, (category_name, fig) in enumerate(
                            category_plots.items()
                        ):
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
                details = get_experiment_details(
                    selected_model_type, exp_name, exp_version
                )
                st.json(details)
            except Exception as e:
                st.error(f"Could not load experiment details: {e}")

        with tab6:
            st.header("Experiment Metadata")
            for exp in experiments:
                with st.expander(f"{exp['full_name']} Metadata"):
                    st.json(exp)

    if page == "Predictions":
        st.title("Board Game Predictions")

        # Load available jobs
        with st.spinner("Loading available prediction jobs..."):
            jobs_df = get_prediction_jobs()

        if jobs_df.empty:
            st.error("No prediction jobs found in BigQuery.")
            return

        # Job selection in sidebar
        st.sidebar.header("Job Selection")
        job_options = []
        for _, job in jobs_df.iterrows():
            latest_pred = pd.to_datetime(job["latest_prediction"]).strftime(
                "%Y-%m-%d %H:%M"
            )
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
            return

        # Cache controls
        st.sidebar.header("Cache Controls")
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Predictions Table",
                "Geek Rating Distribution",
                "Analysis",
                "BigQuery Jobs",
            ]
        )

        with tab1:
            # Year bucketing
            df["year_bucket"] = df["year_published"].apply(
                lambda x: "Other" if x < 2010 else str(x)
            )

            # Column renaming
            if (
                "predicted_geek_rating" not in df.columns
                and "geek_rating" in df.columns
            ):
                df["predicted_geek_rating"] = df["geek_rating"]
            if "actual_geek_rating" not in df.columns and "actual" in df.columns:
                df["actual_geek_rating"] = df["actual"]

            # Year selection
            unique_years = sorted(
                df["year_bucket"].unique(), key=lambda x: (x == "Other", x)
            )
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
                lambda row: f"https://boardgamegeek.com/boardgame/{row['game_id']}",
                axis=1,
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
            display_columns = [
                col for col in display_columns if col in display_df.columns
            ]

            # Display table
            st.dataframe(
                display_df[display_columns],
                use_container_width=True,
                hide_index=True,
                height=800,
                column_config={
                    "bgg_link": st.column_config.LinkColumn(
                        "BoardGameGeek", display_text="BGG"
                    )
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
                        "latest_prediction": st.column_config.TextColumn(
                            "Latest Prediction"
                        ),
                        "earliest_prediction": st.column_config.TextColumn(
                            "Earliest Prediction"
                        ),
                        "min_year": st.column_config.NumberColumn(
                            "Min Year", format="%d"
                        ),
                        "max_year": st.column_config.NumberColumn(
                            "Max Year", format="%d"
                        ),
                        "avg_predicted_rating": st.column_config.NumberColumn(
                            "Avg Rating", format="%.3f"
                        ),
                        "hurdle_experiment": st.column_config.TextColumn(
                            "Hurdle Model"
                        ),
                        "complexity_experiment": st.column_config.TextColumn(
                            "Complexity Model"
                        ),
                        "rating_experiment": st.column_config.TextColumn(
                            "Rating Model"
                        ),
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
                        latest_job_time = pd.to_datetime(
                            jobs_df["latest_prediction"]
                        ).max()
                        st.metric("Latest Job", latest_job_time.strftime("%Y-%m-%d"))
                with col4:
                    if "avg_predicted_rating" in jobs_df.columns:
                        overall_avg = jobs_df["avg_predicted_rating"].mean()
                        st.metric("Overall Avg Rating", f"{overall_avg:.3f}")


if __name__ == "__main__":
    main()
