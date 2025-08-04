import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

from .preprocessor import create_bgg_preprocessor
from .transformers import DescriptionTransformer


def create_unsupervised_pipeline(
    model_type: str = "linear", description_embedding: bool = True, **kwargs
) -> Pipeline:
    """
    Create a preprocessing pipeline for unsupervised learning.

    Parameters
    ----------
    model_type : str, optional (default='linear')
        Type of preprocessing for the model.
        Options:
        - 'linear': Full preprocessing with scaling
        - 'tree': Minimal preprocessing

    description_embedding : bool, optional (default=True)
        Whether to include description embeddings.

    **kwargs : dict
        Additional arguments passed to BaseBGGTransformer.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A preprocessing pipeline for unsupervised learning.
    """
    # Create base preprocessor pipeline
    pipeline_steps = [
        ("preprocessor", create_bgg_preprocessor(model_type=model_type, **kwargs))
    ]

    # Optionally add description embeddings
    if description_embedding:
        pipeline_steps.append(("description_embed", DescriptionTransformer()))

    pipeline = Pipeline(pipeline_steps)
    pipeline.set_output(transform="pandas")

    return pipeline


def perform_pca(
    X: pd.DataFrame, n_components: int = None, explained_variance_ratio: float = 0.95
) -> dict:
    """
    Perform Principal Component Analysis.

    Parameters
    ----------
    X : pd.DataFrame
        Input data for PCA.

    n_components : int, optional
        Number of components to keep. If None, determined by explained_variance_ratio.

    explained_variance_ratio : float, optional (default=0.95)
        Minimum proportion of variance to preserve.

    Returns
    -------
    dict
        PCA results including transformed data, explained variance, and loadings.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    if n_components is None:
        pca = PCA(n_components=explained_variance_ratio)
    else:
        pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X_scaled)

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=X.columns,
    )

    return {
        "transformed_data": X_pca,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "pca_model": pca,
    }


def perform_kmeans(
    X: pd.DataFrame, n_clusters_range: list = range(2, 11), random_state: int = 42
) -> dict:
    """
    Perform K-Means clustering with evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Input data for clustering.

    n_clusters_range : list, optional
        Range of cluster numbers to evaluate.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        K-Means clustering results including best model, metrics, and visualizations.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute metrics for different cluster numbers
    metrics = {
        "n_clusters": [],
        "inertia": [],
        "silhouette_score": [],
        "davies_bouldin_score": [],
    }

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)

        metrics["n_clusters"].append(n_clusters)
        metrics["inertia"].append(kmeans.inertia_)
        metrics["silhouette_score"].append(silhouette_score(X_scaled, kmeans.labels_))
        metrics["davies_bouldin_score"].append(
            davies_bouldin_score(X_scaled, kmeans.labels_)
        )

    # Find the best number of clusters (lowest Davies-Bouldin score)
    best_n_clusters = metrics["n_clusters"][np.argmin(metrics["davies_bouldin_score"])]

    # Fit the best K-Means model
    best_kmeans = KMeans(
        n_clusters=best_n_clusters, random_state=random_state, n_init=10
    )
    best_kmeans.fit(X_scaled)

    return {
        "best_model": best_kmeans,
        "metrics": metrics,
        "labels": best_kmeans.labels_,
        "best_n_clusters": best_n_clusters,
    }


def perform_gmm(
    X: pd.DataFrame, n_components_range: list = range(2, 11), random_state: int = 42
) -> dict:
    """
    Perform Gaussian Mixture Model clustering with evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Input data for clustering.

    n_components_range : list, optional
        Range of component numbers to evaluate.

    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        GMM clustering results including best model, metrics, and visualizations.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute metrics for different component numbers
    metrics = {"n_components": [], "bic": [], "aic": [], "silhouette_score": []}

    best_gmm = None
    best_bic = np.inf

    for n_components in n_components_range:
        gmm = GaussianMixture(
            n_components=n_components, random_state=random_state, n_init=10
        )
        gmm.fit(X_scaled)

        metrics["n_components"].append(n_components)
        metrics["bic"].append(gmm.bic(X_scaled))
        metrics["aic"].append(gmm.aic(X_scaled))

        # Predict labels for silhouette score
        labels = gmm.predict(X_scaled)
        metrics["silhouette_score"].append(silhouette_score(X_scaled, labels))

        # Track the best model based on BIC
        if gmm.bic(X_scaled) < best_bic:
            best_bic = gmm.bic(X_scaled)
            best_gmm = gmm

    return {
        "best_model": best_gmm,
        "metrics": metrics,
        "labels": best_gmm.predict(X_scaled),
        "best_n_components": best_gmm.n_components_,
    }


def plot_pca_variance(pca_results: dict, save_path: str = None):
    """
    Plot explained variance for PCA results.

    Parameters
    ----------
    pca_results : dict
        Results from perform_pca function.

    save_path : str, optional
        Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(pca_results["explained_variance_ratio"]) + 1),
        pca_results["cumulative_explained_variance"],
        marker="o",
    )
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_clustering_metrics(
    results: dict, metric_type: str = "kmeans", save_path: str = None
):
    """
    Plot clustering metrics for K-Means or GMM.

    Parameters
    ----------
    results : dict
        Results from perform_kmeans or perform_gmm function.

    metric_type : str
        Type of clustering ('kmeans' or 'gmm').

    save_path : str, optional
        Path to save the plot.
    """
    plt.figure(figsize=(12, 4))

    if metric_type == "kmeans":
        plt.subplot(131)
        plt.plot(results["metrics"]["n_clusters"], results["metrics"]["inertia"])
        plt.title("Inertia")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Inertia")

        plt.subplot(132)
        plt.plot(
            results["metrics"]["n_clusters"], results["metrics"]["silhouette_score"]
        )
        plt.title("Silhouette Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")

        plt.subplot(133)
        plt.plot(
            results["metrics"]["n_clusters"], results["metrics"]["davies_bouldin_score"]
        )
        plt.title("Davies-Bouldin Score")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Davies-Bouldin Score")

    elif metric_type == "gmm":
        plt.subplot(131)
        plt.plot(results["metrics"]["n_components"], results["metrics"]["bic"])
        plt.title("BIC")
        plt.xlabel("Number of Components")
        plt.ylabel("BIC")

        plt.subplot(132)
        plt.plot(results["metrics"]["n_components"], results["metrics"]["aic"])
        plt.title("AIC")
        plt.xlabel("Number of Components")
        plt.ylabel("AIC")

        plt.subplot(133)
        plt.plot(
            results["metrics"]["n_components"], results["metrics"]["silhouette_score"]
        )
        plt.title("Silhouette Score")
        plt.xlabel("Number of Components")
        plt.ylabel("Silhouette Score")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def visualize_clustering(
    X: pd.DataFrame, labels: np.ndarray, method: str = "pca", save_path: str = None
):
    """
    Visualize clustering results using dimensionality reduction.

    Parameters
    ----------
    X : pd.DataFrame
        Original input data.

    labels : np.ndarray
        Cluster labels.

    method : str, optional
        Dimensionality reduction method ('pca' or 'umap').

    save_path : str, optional
        Path to save the plot.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X_scaled)
    elif method == "umap":
        try:
            import umap

            reducer = umap.UMAP(n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            reducer = PCA(n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"Clustering Visualization ({method.upper()})")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")

    if save_path:
        plt.savefig(save_path)
    plt.close()
