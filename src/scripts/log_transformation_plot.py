import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.preprocessor import create_bgg_preprocessor


def plot_log_transformation(feature="min_age"):
    """
    Create a plot comparing the original and preprocessor-transformed distribution of a feature.

    Parameters:
    -----------
    feature : str, optional (default='min_age')
        The feature to plot transformation for

    Returns:
    --------
    matplotlib.figure.Figure
        The created plot
    """
    # Load the raw data
    df = pd.read_parquet("data/raw/game_features.parquet")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original distribution
    sns.histplot(df[feature], kde=True, ax=ax1)
    ax1.set_title(f"Original {feature} Distribution")
    ax1.set_xlabel(feature)
    ax1.set_ylabel("Frequency")

    # Compute skewness for original distribution
    original_skew = df[feature].skew()

    # Create preprocessor with linear model configuration
    preprocessor = create_bgg_preprocessor(model_type="linear")

    # Fit and transform the data
    transformed_df = preprocessor.fit_transform(df)

    # Distribution after preprocessing (log-transformed)
    preprocessed_data = transformed_df[feature]
    sns.histplot(preprocessed_data, kde=True, ax=ax2)
    preprocessed_skew = preprocessed_data.skew()

    ax2.set_title(f"Preprocessor Log-Transformed {feature} Distribution")
    ax2.set_xlabel(feature)
    ax2.set_ylabel("Frequency")

    # Add skewness information to the plot
    plt.suptitle(
        f"Distribution Comparison for {feature}\n"
        f"Original Skewness: {original_skew:.4f}, "
        f"Preprocessed Skewness: {preprocessed_skew:.4f}",
        fontsize=12,
    )

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"figures/{feature}_preprocessor_transformation.png")

    return fig


if __name__ == "__main__":
    plot_log_transformation()
