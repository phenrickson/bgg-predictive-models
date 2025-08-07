import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_year_transformations(
    transformed_df: pl.DataFrame, output_path: str = "figures/year_transformations.png"
):
    """
    Create a multi-panel plot showing different year feature distributions.

    Args:
        transformed_df (pl.DataFrame): Transformed DataFrame with year features
        output_path (str, optional): Path to save the plot. Defaults to 'figures/year_transformations.png'
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    plt.figure(figsize=(20, 15))

    # 1. Original Year Distribution
    plt.subplot(2, 3, 1)
    plt.hist(transformed_df["year_published"].to_numpy(), bins=50, edgecolor="black")
    plt.title("Original Year Distribution")
    plt.xlabel("Year Published")
    plt.ylabel("Number of Games")

    # 2. Centered Year Distribution
    plt.subplot(2, 3, 2)
    plt.hist(
        transformed_df["year_published_centered"].to_numpy(), bins=50, edgecolor="black"
    )
    plt.title("Centered Year Distribution")
    plt.xlabel("Years from 2000")
    plt.ylabel("Number of Games")

    # 3. Normalized Year Distribution
    plt.subplot(2, 3, 3)
    plt.hist(
        transformed_df["year_published_normalized"].to_numpy(),
        bins=50,
        edgecolor="black",
    )
    plt.title("Normalized Year Distribution")
    plt.xlabel("Normalized Year")
    plt.ylabel("Number of Games")

    # 4. Log Distance Distribution
    plt.subplot(2, 3, 4)
    plt.hist(
        transformed_df["year_published_log_distance"].to_numpy(),
        bins=50,
        edgecolor="black",
    )
    plt.title("Log Distance from 2000")
    plt.xlabel("Log Transformed Distance")
    plt.ylabel("Number of Games")

    # 5. Quadratic Year Feature
    plt.subplot(2, 3, 5)
    plt.hist(
        transformed_df["year_published_quadratic"].to_numpy(),
        bins=50,
        edgecolor="black",
    )
    plt.title("Quadratic Year Feature")
    plt.xlabel("Quadratic Transformed Year")
    plt.ylabel("Number of Games")

    # 6. Cubic Year Feature
    plt.subplot(2, 3, 6)
    plt.hist(
        transformed_df["year_published_cubic"].to_numpy(), bins=50, edgecolor="black"
    )
    plt.title("Cubic Year Feature")
    plt.xlabel("Cubic Transformed Year")
    plt.ylabel("Number of Games")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_year_era_distribution(
    transformed_df: pl.DataFrame, output_path: str = "figures/year_era_distribution.png"
):
    """
    Create a bar plot showing the distribution of games across publication eras.

    Args:
        transformed_df (pl.DataFrame): Transformed DataFrame with year features
        output_path (str, optional): Path to save the plot. Defaults to 'figures/year_era_distribution.png'
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Calculate era distribution
    era_counts = transformed_df["year_published_era"].value_counts()
    era_counts_sorted = era_counts.sort(by="count", descending=True)

    plt.figure(figsize=(12, 6))
    plt.bar(
        era_counts_sorted["year_published_era"].to_numpy(),
        era_counts_sorted["count"].to_numpy(),
    )
    plt.title("Distribution of Games by Publication Era")
    plt.xlabel("Publication Era")
    plt.ylabel("Number of Games")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return era_counts


def generate_year_visualizations(
    raw_features: pl.DataFrame, year_transformer=None, output_dir: str = "figures"
):
    """
    Generate and save year-related visualizations.

    Args:
        raw_features (pl.DataFrame): Raw features DataFrame
        year_transformer (YearTransformer, optional): Year transformer. If None, will create a default one.
        output_dir (str, optional): Directory to save visualizations

    Returns:
        pl.DataFrame: Era counts
    """
    from src.features.preprocessor import YearTransformer

    # Create default transformer if not provided
    if year_transformer is None:
        year_transformer = YearTransformer(reference_year=2000, normalization_factor=25)

    # Transform the data
    transformed_df = year_transformer.transform(raw_features)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot transformations
    plot_year_transformations(
        transformed_df, os.path.join(output_dir, "year_transformations.png")
    )

    # Plot era distribution
    era_counts = plot_year_era_distribution(
        transformed_df, os.path.join(output_dir, "year_era_distribution.png")
    )

    return era_counts


if __name__ == "__main__":
    # Example usage when script is run directly
    import polars as pl

    # Load raw features
    raw_features = pl.read_parquet("data/raw/games_features_raw.parquet")

    # Generate visualizations
    era_counts = generate_year_visualizations(raw_features)

    # Print era distribution
    print("Year Era Distribution:")
    print(era_counts)
