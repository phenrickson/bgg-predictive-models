import os
import sys
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.data.loader import BGGDataLoader
from src.data.config import load_config
from src.features.preprocessor import create_bgg_preprocessor
from src.features.unsupervised import perform_pca, perform_kmeans


def load_bgg_data(end_train_year=2024, min_ratings=25):
    """Load board game data for unsupervised analysis with caching."""
    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(project_root, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file path
    cache_file = os.path.join(
        cache_dir, f"bgg_unsupervised_data_{end_train_year}_{min_ratings}.parquet"
    )

    # Try to load cached data
    if os.path.exists(cache_file):
        try:
            st.info(f"Loading data from cache: {cache_file}")
            cached_data = pd.read_parquet(cache_file)
            st.info(f"Cached data loaded successfully. Shape: {cached_data.shape}")
            return cached_data
        except Exception as e:
            st.warning(f"Could not load cached data: {e}")

    # If no cached data, load and process
    st.info("No cached data found. Processing data from source...")
    config = load_config()
    loader = BGGDataLoader(config)

    # Load training data
    df_raw = loader.load_training_data(
        end_train_year=end_train_year, min_ratings=min_ratings
    )

    # Load predictions to add complexity feature
    predictions = pl.read_parquet("data/predictions/game_predictions.parquet")

    # Join predictions with raw data
    df_processed = df_raw.join(
        predictions.select(["game_id", "predicted_complexity"]),
        on="game_id",
        how="inner",
    ).to_pandas()

    # Cache the processed data
    try:
        st.info(f"Caching processed data to: {cache_file}")
        df_processed.to_parquet(cache_file)
        st.info("Data cached successfully.")
    except Exception as e:
        st.warning(f"Could not cache data: {e}")

    return df_processed


def create_preprocessor(
    create_designer_features=True,
    create_publisher_features=True,
    create_artist_features=True,
    create_family_features=True,
    create_category_features=True,
    create_mechanic_features=True,
    include_base_numeric=True,
):
    """Create a preprocessor with configurable feature groups."""
    preprocessor = create_bgg_preprocessor(
        create_designer_features=create_designer_features,
        create_publisher_features=create_publisher_features,
        create_artist_features=create_artist_features,
        create_family_features=create_family_features,
        create_category_features=create_category_features,
        create_mechanic_features=create_mechanic_features,
        include_base_numeric=include_base_numeric,
        preserve_columns=["year_published", "predicted_complexity"],
    )
    return preprocessor


def dimension_reduction(X, method="PCA", n_components=2):
    """Perform dimension reduction."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "UMAP":
        from umap import UMAP

        reducer = UMAP(n_components=n_components, random_state=42)

    X_reduced = reducer.fit_transform(X_scaled)
    return X_reduced


def clustering(X, method="KMeans", n_clusters=5, eps=0.5, min_samples=5):
    """Perform clustering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "DBSCAN":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    labels = clusterer.fit_predict(X_scaled)
    return labels


def main():
    st.set_page_config(page_title="BGG Unsupervised Learning Dashboard", layout="wide")
    st.title("Board Game Unsupervised Learning Dashboard")

    # Sidebar for preprocessing options
    st.sidebar.header("Preprocessing Options")

    # Feature group toggles
    feature_groups = [
        "Designer",
        "Publisher",
        "Artist",
        "Family",
        "Category",
        "Mechanic",
        "Base Numeric",
    ]
    feature_flags = {}
    for group in feature_groups:
        feature_flags[group.lower().replace(" ", "_")] = st.sidebar.checkbox(
            f"Include {group} Features", value=True
        )

    # Dimension reduction options
    st.sidebar.header("Dimension Reduction")
    dim_reduction_method = st.sidebar.selectbox(
        "Dimension Reduction Method", ["PCA", "t-SNE", "UMAP"], index=0
    )
    n_components = st.sidebar.slider(
        "Number of Components", min_value=2, max_value=10, value=2
    )

    # Clustering options
    st.sidebar.header("Clustering")
    clustering_method = st.sidebar.selectbox(
        "Clustering Method", ["KMeans", "DBSCAN"], index=0
    )

    if clustering_method == "KMeans":
        n_clusters = st.sidebar.slider(
            "Number of Clusters", min_value=2, max_value=10, value=5
        )
        eps = 0.5
        min_samples = 5
    else:
        eps = st.sidebar.slider(
            "EPS", min_value=0.1, max_value=2.0, value=0.5, step=0.1
        )
        min_samples = st.sidebar.slider(
            "Min Samples", min_value=2, max_value=20, value=5
        )

    # Color by options
    st.sidebar.header("Visualization")
    color_by = st.sidebar.selectbox(
        "Color By", ["Cluster", "Predicted Complexity", "Year Published"], index=0
    )

    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        data = load_bgg_data()
        preprocessor = create_preprocessor(
            create_designer_features=feature_flags["designer"],
            create_publisher_features=feature_flags["publisher"],
            create_artist_features=feature_flags["artist"],
            create_family_features=feature_flags["family"],
            create_category_features=feature_flags["category"],
            create_mechanic_features=feature_flags["mechanic"],
            include_base_numeric=feature_flags["base_numeric"],
        )

        X = preprocessor.fit_transform(data)

    # Dimension reduction
    with st.spinner(f"Performing {dim_reduction_method} dimension reduction..."):
        X_reduced = dimension_reduction(
            X, method=dim_reduction_method, n_components=n_components
        )

    # Clustering
    with st.spinner(f"Performing {clustering_method} clustering..."):
        labels = clustering(
            X,
            method=clustering_method,
            n_clusters=n_clusters if clustering_method == "KMeans" else None,
            eps=eps if clustering_method == "DBSCAN" else None,
            min_samples=min_samples if clustering_method == "DBSCAN" else None,
        )

    # Prepare visualization data
    viz_data = pd.DataFrame(
        X_reduced, columns=[f"Component {i+1}" for i in range(n_components)]
    )
    viz_data["Cluster"] = labels
    viz_data["Predicted Complexity"] = data["predicted_complexity"]
    viz_data["Year Published"] = data["year_published"]

    # Add original data columns to viz_data for hover information
    for col in data.columns:
        if col not in viz_data.columns:
            viz_data[col] = data[col]

    # Plotting
    st.header(f"{dim_reduction_method} Visualization")

    if n_components == 2:
        # 2D Plot
        fig = px.scatter(
            viz_data,
            x="Component 1",
            y="Component 2",
            color=color_by,
            hover_data=viz_data.columns.tolist(),
            title=f"{dim_reduction_method} Visualization",
        )
    else:
        # 3D Plot
        fig = px.scatter_3d(
            viz_data,
            x="Component 1",
            y="Component 2",
            z="Component 3",
            color=color_by,
            hover_data=viz_data.columns.tolist(),
            title=f"{dim_reduction_method} 3D Visualization",
        )

    st.plotly_chart(fig, use_container_width=True)

    # Clustering metrics
    st.header("Clustering Metrics")
    try:
        silhouette = silhouette_score(X, labels)
        st.metric("Silhouette Score", f"{silhouette:.4f}")
    except Exception as e:
        st.warning(f"Could not compute silhouette score: {e}")


if __name__ == "__main__":
    main()
