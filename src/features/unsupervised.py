import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as pn

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

    # Create transformed data DataFrame
    transformed_df = pd.DataFrame(
        X_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=X.index
    )

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=X.columns,
    ).reset_index(names="feature")

    return {
        "transformed_data": transformed_df,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        "loadings": loadings,
        "pca_model": pca,
    }


def select_top_pcs(df: pl.DataFrame, n_components: int = 10) -> pl.DataFrame:
    columns = ["feature"] + [f"PC{i}" for i in range(1, n_components + 1)]
    selected = df.select([col for col in columns if col in df.columns])
    return selected


def get_top_loadings(df: pl.DataFrame, n_top_features: int = 15) -> pl.DataFrame:
    top_loadings = (
        df.unpivot(index="feature", variable_name="component")
        .with_columns(pl.col("value").abs().alias("abs_value"))
        .sort(["component", "abs_value"], descending=True)
        .group_by("component")
        .head(n_top_features)
        .drop("abs_value")
    )
    return top_loadings


def plot_pca_loadings(
    pca_results: dict,
    n_top_features: int = 25,
    n_components: int = 5,
    save_path: str = None,
) -> pn.ggplot:
    """
    Create a loadings plot for top features within each principal component.

    Parameters
    ----------
    pca_results : dict
        Results from perform_pca function.

    n_top_features : int, optional (default=25)
        Number of top features to plot for each principal component.

    n_components : int, optional (default=5)
        Number of principal components to plot.

    save_path : str, optional
        Path to save the plot.

    Returns
    -------
    pn.ggplot
        Plotnine plot of feature loadings.
    """
    # Convert loadings to Polars DataFrame
    loadings_pl = pl.DataFrame(
        pca_results["loadings"].reset_index().rename(columns={"index": "feature"})
    )

    # Limit n_components to available components
    n_components = min(n_components, loadings_pl.width - 1)
    pc_columns = [f"PC{i+1}" for i in range(n_components)]

    # Prepare data for plotting
    plot_data_list = []
    for pc_col in pc_columns:
        # Get top features for this specific principal component
        top_features = (
            loadings_pl.select(["feature", pc_col])
            .with_columns(pl.col(pc_col).abs().alias("abs_loading"))
            .sort("abs_loading", descending=True)
            .head(n_top_features)
        )

        # Add principal component column
        top_features = top_features.with_columns(
            pl.lit(pc_col).alias("Principal Component")
        )

        plot_data_list.append(top_features)

    # Combine data for all components
    plot_data_melted = pl.concat(plot_data_list)

    # Convert to pandas for plotnine
    plot_data_pd = plot_data_melted.to_pandas()

    # Create plot
    plot = (
        pn.ggplot(plot_data_pd, pn.aes(x="feature", y=pc_col, fill=pc_col))
        + pn.geom_bar(stat="identity", position="identity")
        + pn.facet_wrap("~ Principal Component", scales="free_y", ncol=1)
        + pn.coord_flip()
        + pn.theme_minimal()
        + pn.labs(
            title="Top 25 Feature Loadings for Each Principal Component",
            x="Features",
            y="Loading",
        )
        + pn.scale_fill_gradient2(low="blue", mid="white", high="red")
    )

    # Save plot if path is provided
    if save_path:
        plot.save(save_path, dpi=300, bbox_inches="tight")

    return plot


def perform_kmeans(
    X: pd.DataFrame,
    n_clusters_range: list = range(2, 11),
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
    early_stopping_threshold: float = 0.01,
    use_mini_batch: bool = False,
) -> dict:
    """
    Perform K-Means clustering with advanced optimization and logging.

    Parameters
    ----------
    X : pd.DataFrame
        Input data for clustering.

    n_clusters_range : list, optional
        Range of cluster numbers to evaluate.

    random_state : int, optional
        Random seed for reproducibility.

    n_jobs : int, optional (default=-1)
        Number of jobs to run in parallel. -1 means using all processors.

    verbose : bool, optional (default=True)
        Whether to print detailed logging information.

    early_stopping_threshold : float, optional (default=0.01)
        Threshold for early stopping based on improvement in Davies-Bouldin score.

    use_mini_batch : bool, optional (default=False)
        Whether to use MiniBatchKMeans for faster processing on large datasets.

    Returns
    -------
    dict
        K-Means clustering results including best model, metrics, and visualizations.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

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

    best_score = float("inf")
    best_n_clusters = n_clusters_range[0]

    for n_clusters in n_clusters_range:
        logger.info(f"Processing clustering with {n_clusters} clusters...")

        # Choose clustering algorithm based on dataset size and user preference
        if use_mini_batch:
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init="auto",
                batch_size=min(1024, X_scaled.shape[0] // 10),
                max_iter=100,
            )
        else:
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init="auto",
                algorithm="lloyd",
                n_jobs=n_jobs,
            )

        # Fit the model
        clusterer.fit(X_scaled)

        # Compute metrics
        current_silhouette = silhouette_score(X_scaled, clusterer.labels_)
        current_davies = davies_bouldin_score(X_scaled, clusterer.labels_)

        metrics["n_clusters"].append(n_clusters)
        metrics["inertia"].append(clusterer.inertia_)
        metrics["silhouette_score"].append(current_silhouette)
        metrics["davies_bouldin_score"].append(current_davies)

        logger.info(
            f"Clusters: {n_clusters}, Silhouette Score: {current_silhouette:.4f}, Davies-Bouldin Score: {current_davies:.4f}"
        )

        # Early stopping logic
        if current_davies < best_score * (1 - early_stopping_threshold):
            best_score = current_davies
            best_n_clusters = n_clusters
            logger.info(f"Found better clustering with {n_clusters} clusters")
        elif n_clusters > n_clusters_range[0] and current_davies >= best_score:
            logger.info("No significant improvement in clustering. Stopping early.")
            break

    # Fit the best K-Means model
    if use_mini_batch:
        best_kmeans = MiniBatchKMeans(
            n_clusters=best_n_clusters,
            random_state=random_state,
            n_init="auto",
            batch_size=min(1024, X_scaled.shape[0] // 10),
            max_iter=100,
        )
    else:
        best_kmeans = KMeans(
            n_clusters=best_n_clusters,
            random_state=random_state,
            n_init="auto",
            algorithm="lloyd",
            n_jobs=n_jobs,
        )

    best_kmeans.fit(X_scaled)

    logger.info(f"Best number of clusters: {best_n_clusters}")

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
