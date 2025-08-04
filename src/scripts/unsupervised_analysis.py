import os
import pandas as pd
import numpy as np

from src.data.loader import load_data
from src.features.unsupervised import (
    create_unsupervised_pipeline,
    perform_pca,
    perform_kmeans,
    perform_gmm,
    plot_pca_variance,
    plot_clustering_metrics,
    visualize_clustering,
)


def main():
    # Load data
    df = load_data()

    # Create preprocessing pipeline
    pipeline = create_unsupervised_pipeline(
        model_type="linear", description_embedding=True
    )

    # Preprocess data
    X = pipeline.fit_transform(df)

    # Perform PCA
    pca_results = perform_pca(X)
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

    # Print some results
    print("PCA Results:")
    print(f"Number of components: {len(pca_results['explained_variance_ratio'])}")
    print(
        f"Cumulative explained variance: {pca_results['cumulative_explained_variance'][-1]}"
    )

    print("\nK-Means Results:")
    print(f"Best number of clusters: {kmeans_results['best_n_clusters']}")
    print(
        f"Best silhouette score: {max(kmeans_results['metrics']['silhouette_score'])}"
    )

    print("\nGMM Results:")
    print(f"Best number of components: {gmm_results['best_n_components']}")
    print(f"Best silhouette score: {max(gmm_results['metrics']['silhouette_score'])}")


if __name__ == "__main__":
    main()
