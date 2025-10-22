"""
Reverse Engineering the Geek Rating

This script analyzes the Boardgame Geek rating calculation by examining
the relationship between average rating and Bayesian average rating.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_games_data():
    """Load games data from BigQuery."""
    config = load_config()

    # Custom query to get necessary columns
    query = f"""
    SELECT 
        game_id, 
        name, 
        year_published, 
        average_weight as complexity,
        average_rating as rating,
        bayes_average as geek_rating,
        users_rated
    FROM `{config.project_id}.{config.dataset}.games_active`
    WHERE year_published IS NOT NULL and bayes_average != 0
    """

    # Execute query and convert to polars DataFrame
    logger.info(f"Executing query: {query}")
    client = config.get_client()
    pandas_df = client.query(query).to_dataframe()

    logger.info(f"Retrieved {len(pandas_df)} rows")
    return pl.from_pandas(pandas_df)


def plot_average_vs_bayesaverage(games):
    """Create scatter plot comparing average and Bayesian average."""
    plt.figure(figsize=(10, 6))
    plt.scatter(games["rating"], games["geek_rating"], alpha=0.25)
    plt.title("Average Rating vs Bayesian Average")
    plt.xlabel("Average Rating")
    plt.ylabel("Bayesian Average")
    plt.tight_layout()
    plt.savefig("figures/examine_geek_rating/average_vs_bayesaverage.png")
    plt.close()


def plot_average_vs_log_usersrated(games):
    """Create scatter plot of average vs log(users rated)."""
    plt.figure(figsize=(10, 6))
    plt.scatter(games["rating"], np.log(games["users_rated"]), alpha=0.5)
    plt.title("Average Rating vs Log(Users Rated)")
    plt.xlabel("Average Rating")
    plt.ylabel("Log(Users Rated)")
    plt.tight_layout()
    plt.savefig("figures/examine_geek_rating/average_vs_log_usersrated.png")
    plt.close()


def calculate_bayesian_average(games, dummy_ratings, dummy_average):
    """Calculate estimated Bayesian average."""
    est_bayesaverage = (
        (games["rating"] * games["users_rated"]) + (dummy_ratings * dummy_average)
    ) / (games["users_rated"] + dummy_ratings)
    return est_bayesaverage


def parameter_search(games):
    """Perform parameter grid search to estimate Bayesian average calculation."""
    # Create parameter grid
    ratings = np.arange(500, 2100, 100)
    averages = np.arange(5.0, 6.1, 0.1)

    # Prepare results storage
    metrics_results = []

    # Grid search
    for dummy_ratings in ratings:
        for dummy_average in averages:
            # Calculate estimated Bayesian average
            est_bayesaverage = calculate_bayesian_average(
                games, dummy_ratings, dummy_average
            )

            # Calculate metrics
            rmse = root_mean_squared_error(games["geek_rating"], est_bayesaverage)
            mae = mean_absolute_error(games["geek_rating"], est_bayesaverage)
            r2 = r2_score(games["geek_rating"], est_bayesaverage)

            # Store results
            metrics_results.append(
                {
                    "dummy_ratings": dummy_ratings,
                    "dummy_average": dummy_average,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )

    # Convert to DataFrame
    metrics_df = pl.DataFrame(metrics_results)

    return metrics_df


def plot_metrics(metrics_df):
    """Plot metrics across different parameter combinations."""
    # RMSE Plot
    plt.figure(figsize=(12, 8))
    for avg in metrics_df["dummy_average"].unique():
        subset = metrics_df.filter(pl.col("dummy_average") == avg)
        plt.plot(subset["dummy_ratings"], subset["rmse"], label=f"Avg: {avg:.1f}")

    plt.title("RMSE across Different Parameters")
    plt.xlabel("Dummy Ratings")
    plt.ylabel("RMSE")
    plt.legend(title="Dummy Average")
    plt.tight_layout()
    plt.savefig("figures/examine_geek_rating/rmse_across_parameters.png")
    plt.close()

    # MAE Plot
    plt.figure(figsize=(12, 8))
    for avg in metrics_df["dummy_average"].unique():
        subset = metrics_df.filter(pl.col("dummy_average") == avg)
        plt.plot(subset["dummy_ratings"], subset["mae"], label=f"Avg: {avg:.1f}")

    plt.title("MAE across Different Parameters")
    plt.xlabel("Dummy Ratings")
    plt.ylabel("MAE")
    plt.legend(title="Dummy Average")
    plt.tight_layout()
    plt.savefig("figures/examine_geek_rating/mae_across_parameters.png")
    plt.close()

    # R² Plot
    plt.figure(figsize=(12, 8))
    for avg in metrics_df["dummy_average"].unique():
        subset = metrics_df.filter(pl.col("dummy_average") == avg)
        plt.plot(subset["dummy_ratings"], subset["r2"], label=f"Avg: {avg:.1f}")

    plt.title("R² across Different Parameters")
    plt.xlabel("Dummy Ratings")
    plt.ylabel("R²")
    plt.legend(title="Dummy Average")
    plt.tight_layout()
    plt.savefig("figures/examine_geek_rating/r2_across_parameters.png")
    plt.close()


def main():
    """Main execution function."""
    # Create output directory
    import os

    os.makedirs("figures/examine_geek_rating", exist_ok=True)

    # Load games data
    games = load_games_data()

    # Create visualizations
    plot_average_vs_bayesaverage(games)
    plot_average_vs_log_usersrated(games)

    # Perform parameter search
    metrics_df = parameter_search(games)

    # Plot metrics
    plot_metrics(metrics_df)

    # Find best parameters (lowest RMSE)
    best_params = metrics_df.sort(by="rmse").head(1)
    print("Best Parameters:")
    print(best_params)


if __name__ == "__main__":
    main()
