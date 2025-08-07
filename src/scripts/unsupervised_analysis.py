import os
import numpy as np
import polars as pl
import plotnine as pn
import plotly.io as pio
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from src.utils.logging import setup_logging
from src.data.config import load_config
from src.data.loader import BGGDataLoader
from src.features.transformers import ColumnTransformerNoPrefix
from src.features.preprocessor import create_bgg_preprocessor
from src.features.unsupervised import *
from src.visualizations.pca_loadings_plot import plot_pca_loadings


# Load training data
def load_data(loader):

    # query dataset
    df_raw = loader.load_training_data(
        end_train_year=2024,  # Use a reasonable training year
        min_ratings=25,  # Ensure games have a minimum number of ratings
    )

    # load predictions to inner join with predicted_complexity
    predictions = pl.read_parquet("data/predictions/game_predictions.parquet")

    # Use estimated complexity as feature and convert to pandas
    df_pandas = df_raw.join(
        predictions.select(pl.col("game_id", "predicted_complexity")),
        on="game_id",
        how="inner",
    ).to_pandas()

    return df_raw, df_pandas


def main():

    logger = setup_logging()

    # Load configuration
    config = load_config()

    # Create data loader
    loader = BGGDataLoader(config)

    # get raw data and data with predictions
    df_raw, df_pandas = load_data(loader)

    # create preprocessor
    preprocessor = create_bgg_preprocessor(
        create_designer_features=False,
        create_publisher_features=False,
        create_artist_features=False,
        create_family_features=False,
        preserve_columns=["year_published", "predicted_complexity"],
    )

    # Preprocess data
    X = preprocessor.fit_transform(df_pandas)

    # Perform PCA
    pca_results = perform_pca(X.drop("year_published_transformed", axis=1))

    # look at loadings
    loadings = pl.from_pandas(pca_results["loadings"])

    # Limit n_components to available components
    n_components = 5
    n_top_features = 15
    n_components = min(n_components, loadings.shape[1])

    top_loadings = get_top_loadings(
        select_top_pcs(loadings, n_components=2), n_top_features=50
    )

    # Function removed
    fig = plot_pca_loadings(top_loadings, cols=2)

    # Save the figure as a PNG
    plot_pca_variance(pca_results, save_path="figures/unsupervised/pca_variance.png")

    # Perform K-Means
    kmeans_results = perform_kmeans(X)
    plot_clustering_metrics(
        kmeans_results,
        metric_type="kmeans",
        save_path="figures/unsupervised/kmeans_metrics.png",
    )
    visualize_clustering(
        X,
        kmeans_results["labels"],
        method="pca",
        save_path="figures/unsupervised/kmeans_clustering_pca.png",
    )

    # Perform Gaussian Mixture Model
    gmm_results = perform_gmm(X)
    plot_clustering_metrics(
        gmm_results, metric_type="gmm", save_path="figures/unsupervised/gmm_metrics.png"
    )
    visualize_clustering(
        X,
        gmm_results["labels"],
        method="pca",
        save_path="figures/unsupervised/gmm_clustering_pca.png",
    )

    # logger.info some results
    logger.info("PCA Results:")
    logger.info(f"Number of components: {len(pca_results['explained_variance_ratio'])}")
    logger.info(
        f"Cumulative explained variance: {pca_results['cumulative_explained_variance'][-1]}"
    )

    logger.info("\nK-Means Results:")
    logger.info(f"Best number of clusters: {kmeans_results['best_n_clusters']}")
    logger.info(
        f"Best silhouette score: {max(kmeans_results['metrics']['silhouette_score'])}"
    )

    logger.info("\nGMM Results:")
    logger.info(f"Best number of components: {gmm_results['best_n_components']}")
    logger.info(
        f"Best silhouette score: {max(gmm_results['metrics']['silhouette_score'])}"
    )


# if __name__ == "__main__":
#     main()
