"""
Streamlit dashboard for exploring board game predictions.

This dashboard loads predictions from data/predictions/game_predictions.parquet
and provides interactive visualizations for exploring predicted geek ratings,
complexity, ratings, and other metrics across different publication years.
"""

import streamlit as st
import polars as pl
import plotly.express as px
from pathlib import Path
from datetime import datetime


def load_predictions() -> pl.DataFrame:
    """
    Load predictions from the game predictions parquet file.

    Returns:
        Polars DataFrame with predictions
    """
    predictions_path = Path("data/predictions/game_predictions.parquet")

    if not predictions_path.exists():
        st.error(
            "Predictions file not found at: data/predictions/game_predictions.parquet"
        )
        return None

    try:
        return pl.read_parquet(str(predictions_path))
    except Exception as e:
        st.error(f"Error reading predictions file: {e}")
        return None


def check_predictions_file() -> bool:
    """
    Check if the predictions file exists.

    Returns:
        bool: True if file exists, False otherwise
    """
    predictions_path = Path("data/predictions/game_predictions.parquet")
    return predictions_path.exists()


def main():
    st.set_page_config(page_title="Geek Rating Predictions Explorer", layout="wide")

    st.title("Geek Rating Predictions Explorer")

    # Check if predictions file exists
    if not check_predictions_file():
        st.error(
            "No predictions file found at data/predictions/game_predictions.parquet"
        )
        return

    # Load predictions
    predictions = load_predictions()

    if predictions is None:
        st.error("Could not load predictions. Please select another experiment.")
        return

    # Convert to pandas for easier manipulation
    df = predictions.to_pandas()

    # Debug: Print available columns
    st.sidebar.subheader("Available Columns")
    st.sidebar.write(df.columns.tolist())

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

    # Add last updated time in sidebar
    st.sidebar.header("Data Info")
    predictions_path = Path("data/predictions/game_predictions.parquet")
    if predictions_path.exists():
        last_modified = datetime.fromtimestamp(predictions_path.stat().st_mtime)
        st.sidebar.text(f"Last Updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
