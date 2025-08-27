"""
Create distribution plots for complexity, rating, and log(users_rated)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

# Project imports
from src.models.training import load_data


def plot_distributions(df):
    """
    Create distribution plots for key variables

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing game data
    """
    # Replace 0s with NaN
    for col in ["complexity", "rating", "users_rated"]:
        df = df.with_columns(
            pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
        )

    # Set up the plot
    plt.figure(figsize=(15, 10))

    # Complexity distribution
    plt.subplot(2, 2, 1)
    sns.histplot(
        df.select(pl.col("complexity").drop_nulls()).to_numpy().flatten(), kde=True
    )
    plt.title("Complexity Distribution")
    plt.xlabel("Complexity")
    plt.ylabel("Frequency")

    # Rating distribution
    plt.subplot(2, 2, 2)
    sns.histplot(
        df.select(pl.col("rating").drop_nulls()).to_numpy().flatten(), kde=True
    )
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")

    # Untransformed users_rated distribution
    plt.subplot(2, 2, 3)
    sns.histplot(
        df.select(pl.col("users_rated").drop_nulls()).to_numpy().flatten(), kde=True
    )
    plt.title("Users Rated Distribution")
    plt.xlabel("Users Rated")
    plt.ylabel("Frequency")

    # Log(users_rated) distribution
    plt.subplot(2, 2, 4)
    log_users_rated = np.log1p(
        df.select(pl.col("users_rated").drop_nulls()).to_numpy().flatten()
    )
    sns.histplot(log_users_rated, kde=True)
    plt.title("Log(Users Rated) Distribution")
    plt.xlabel("Log(Users Rated + 1)")
    plt.ylabel("Frequency")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("figures/outcome_distributions.png")
    plt.close()


def main():
    """
    Load data and create distribution plots
    """

    # Load data with a reasonable minimum ratings threshold
    df = load_data(min_ratings=0, end_train_year=2023)

    # Create distribution plots
    plot_distributions(df)

    print("Distribution plots saved to figures/variable_distributions.png")


if __name__ == "__main__":
    main()
