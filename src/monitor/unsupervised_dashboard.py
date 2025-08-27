import os
import sys
import time
from multiprocessing import Pool, cpu_count
import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure pandas is imported early and available globally
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.data.loader import BGGDataLoader  # noqa: E402
from src.data.config import load_config  # noqa: E402
from src.features.preprocessor import create_bgg_preprocessor  # noqa: E402
from src.features.unsupervised import perform_pca  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


# Set up logging
logger = setup_logging()


def get_cluster_colors(n_clusters):
    """Get a consistent color mapping for clusters."""
    colors = px.colors.qualitative.Vivid
    # Create a mapping of cluster index to color
    color_map = {str(i): colors[i % len(colors)] for i in range(n_clusters)}
    return color_map


def load_bgg_data(end_train_year=2024, min_ratings=25):
    """Load board game data for unsupervised analysis with caching."""
    logger.info("Starting data loading process for unsupervised analysis")
    logger.debug(
        f"Parameters: end_train_year={end_train_year}, min_ratings={min_ratings}"
    )

    # Create cache directory if it doesn't exist
    cache_dir = os.path.join(project_root, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    logger.debug(f"Ensuring cache directory exists: {cache_dir}")

    # Define cache file path
    cache_file = os.path.join(
        cache_dir, f"bgg_unsupervised_data_{end_train_year}_{min_ratings}.parquet"
    )
    logger.debug(f"Cache file path: {cache_file}")

    # Try to load cached data
    if os.path.exists(cache_file):
        try:
            logger.info(f"Attempting to load data from cache: {cache_file}")
            cached_data = pd.read_parquet(cache_file)
            logger.info(f"Successfully loaded cached data. Shape: {cached_data.shape}")
            logger.debug(f"Cached data columns: {list(cached_data.columns)}")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            st.warning(f"Could not load cached data: {e}")

    # If no cached data, load and process
    logger.info("No cached data found. Processing data from source...")
    config = load_config()
    loader = BGGDataLoader(config)

    # Load training data
    logger.info(
        f"Loading training data with end_train_year={end_train_year}, min_ratings={min_ratings}"
    )
    df_raw = loader.load_training_data(
        end_train_year=end_train_year, min_ratings=min_ratings
    )
    logger.info(f"Raw training data shape: {df_raw.shape}")
    logger.debug(f"Raw training data columns: {list(df_raw.columns)}")

    # Load predictions to add complexity feature
    logger.info("Loading game predictions for complexity feature")
    predictions = pl.read_parquet("data/predictions/game_predictions.parquet")
    logger.debug(f"Predictions data shape: {predictions.shape}")

    # Join predictions with raw data
    logger.info("Joining raw data with predictions")
    df_processed = df_raw.join(
        predictions.select(["game_id", "predicted_complexity"]),
        on="game_id",
        how="inner",
    ).to_pandas()
    logger.info(f"Processed data shape after join: {df_processed.shape}")
    logger.debug(f"Processed data columns: {list(df_processed.columns)}")

    # Cache the processed data
    try:
        logger.info(f"Attempting to cache processed data to: {cache_file}")
        df_processed.to_parquet(cache_file)
        logger.info("Data successfully cached.")
    except Exception as e:
        logger.error(f"Failed to cache data: {e}")
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
    logger.info("Creating preprocessor for unsupervised learning")

    # Log the feature group configuration
    feature_config = {
        "Designer Features": create_designer_features,
        "Publisher Features": create_publisher_features,
        "Artist Features": create_artist_features,
        "Family Features": create_family_features,
        "Category Features": create_category_features,
        "Mechanic Features": create_mechanic_features,
        "Base Numeric Features": include_base_numeric,
    }

    # Log enabled feature groups
    enabled_features = [name for name, enabled in feature_config.items() if enabled]
    logger.debug(f"Enabled feature groups: {', '.join(enabled_features)}")

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

    logger.info("Preprocessor created successfully")
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


def perform_kmeans_single(data, k):
    """Run K-Means clustering for a single k value with caching."""
    logger.info(f"Running K-Means with k={k}")
    k_start = time.time()

    # Run clustering
    clusterer = KMeans(n_clusters=k, random_state=42)
    labels = clusterer.fit_predict(data)

    # Calculate clustering metrics
    silhouette = silhouette_score(data, labels)
    calinski = calinski_harabasz_score(data, labels)
    davies = davies_bouldin_score(data, labels)

    # Calculate silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, labels)

    logger.info(
        f"K={k} metrics - Silhouette: {silhouette:.4f}, CH: {calinski:.4f}, DB: {davies:.4f}"
    )

    # Calculate feature importance per cluster
    feature_importance = {}
    for i in range(k):
        cluster_mask = labels == i
        cluster_mean = data[cluster_mask].mean()
        # Z-score of cluster mean relative to overall distribution
        importance = np.abs((cluster_mean - data.mean()) / data.std())
        feature_importance[i] = importance

    # Store results
    result = {
        "labels": labels,
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "centroids": clusterer.cluster_centers_,
        "inertia": clusterer.inertia_,
        "feature_importance": feature_importance,
        "sample_silhouette_values": sample_silhouette_values,
    }

    logger.info(f"Completed k={k} in {time.time() - k_start:.2f} seconds")
    return result


def run_kmeans_wrapper(args):
    """Wrapper function for parallel k-means clustering."""
    data, k = args
    return k, perform_kmeans_single(data, k)


@st.cache_data(show_spinner=False)
def perform_multi_k_means(data, k_values, feature_flags):
    """Perform K-Means clustering for multiple k values in parallel with caching."""
    logger.info(f"Starting K-Means clustering for k values: {k_values}")
    start_time = time.time()

    # Run clustering for each k value in parallel
    n_cores = min(cpu_count(), len(k_values))  # Use at most one core per k value

    # Create argument tuples for each k value
    args = [(data, k) for k in k_values]

    with Pool(n_cores) as pool:
        results_list = pool.map(run_kmeans_wrapper, args)

    # Convert results list to dictionary
    results = {f"k_{k}": result for k, result in results_list}

    total_time = time.time() - start_time
    logger.info(f"Completed all clustering in {total_time:.2f} seconds")
    return results


def perform_gmm_single(
    data, n_components, covariance_type="full", is_pca_transformed=False
):
    """Run Gaussian Mixture Model clustering for a single n_components value."""
    logger.info(f"Running GMM with n_components={n_components}")
    gmm_start = time.time()

    # Convert to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Only clean data if it's not PCA-transformed
    if not is_pca_transformed:
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Detailed logging of input data
    logger.info(f"Input data shape: {data.shape}")
    logger.info(f"Input data dtype: {data.dtype}")
    logger.info(f"Input data min: {data.min()}")
    logger.info(f"Input data max: {data.max()}")
    logger.info(f"Input data mean: {data.mean()}")

    # Run clustering
    clusterer = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        n_init=10,  # Multiple initializations to find best fit
        reg_covar=1e-3,  # Small regularization to prevent singularities
    )

    # Fit the model and get detailed logging
    try:
        clusterer.fit(data)
    except Exception as e:
        logger.error(f"Fitting GMM failed: {e}")
        raise

    # Get cluster assignments and probabilities
    labels = clusterer.predict(data)

    # Detailed logging of predict_proba
    try:
        responsibilities = clusterer.predict_proba(data)
        logger.info(f"Responsibilities shape: {responsibilities.shape}")
        logger.info(f"Responsibilities min: {responsibilities.min()}")
        logger.info(f"Responsibilities max: {responsibilities.max()}")
        logger.info(f"Responsibilities mean: {responsibilities.mean()}")

        # Check row-wise sums
        row_sums = responsibilities.sum(axis=1)
        logger.info(f"Row sums min: {row_sums.min()}")
        logger.info(f"Row sums max: {row_sums.max()}")
        logger.info(f"Row sums mean: {row_sums.mean()}")

        # Find rows where sum is not close to 1
        problematic_rows = np.where(np.abs(row_sums - 1) > 1e-10)[0]
        if len(problematic_rows) > 0:
            logger.warning(
                f"Found {len(problematic_rows)} rows where probabilities do not sum to 1"
            )
            logger.warning(f"Problematic row indices: {problematic_rows}")
            logger.warning(f"Problematic row sums: {row_sums[problematic_rows]}")
    except Exception as e:
        logger.error(f"predict_proba failed: {e}")
        raise

    # Calculate clustering metrics
    silhouette = silhouette_score(data, labels)
    calinski = calinski_harabasz_score(data, labels)
    davies = davies_bouldin_score(data, labels)

    # Calculate silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, labels)

    logger.info(
        f"n_components={n_components} metrics - Silhouette: {silhouette:.4f}, CH: {calinski:.4f}, DB: {davies:.4f}"
    )

    # Calculate feature importance for each component using scaled data
    feature_importance = {}
    for i in range(n_components):
        # Get samples with high responsibility to this component
        resp_weights = responsibilities[:, i]

        # Calculate weighted mean
        weighted_mean = np.average(data, weights=resp_weights, axis=0)

        # Calculate overall mean and std for comparison
        overall_mean = np.mean(data, axis=0)
        overall_std = np.std(data, axis=0)

        # Calculate feature importance as normalized difference between component and overall statistics
        importance = np.abs(weighted_mean - overall_mean) / (overall_std + 1e-10)
        feature_importance[i] = pd.Series(
            importance, index=data.columns if isinstance(data, pd.DataFrame) else None
        )

    # Store results
    result = {
        "labels": labels,
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "davies_bouldin": davies,
        "means": clusterer.means_,
        "covariances": clusterer.covariances_,
        "weights": clusterer.weights_,
        "bic": clusterer.bic(data),
        "aic": clusterer.aic(data),
        "feature_importance": feature_importance,
        "sample_silhouette_values": sample_silhouette_values,
        "responsibilities": responsibilities,
    }

    logger.info(
        f"Completed n_components={n_components} in {time.time() - gmm_start:.2f} seconds"
    )
    return result


def run_gmm_wrapper(args):
    """Wrapper function for parallel GMM clustering."""
    data, n_components, covariance_type, is_pca_transformed = args
    return n_components, perform_gmm_single(
        data, n_components, covariance_type, is_pca_transformed=is_pca_transformed
    )


@st.cache_data(show_spinner=False)
def perform_multi_gmm(
    data,
    n_components_values,
    covariance_type="full",
    feature_flags=None,
    is_pca_transformed=False,
):
    """Perform GMM clustering for multiple n_components values in parallel with caching."""
    logger.info(
        f"Starting GMM clustering for n_components values: {n_components_values}"
    )
    start_time = time.time()

    # Run clustering for each n_components value in parallel
    n_cores = min(cpu_count(), len(n_components_values))

    # Create argument tuples for each n_components value
    args = [(data, n, covariance_type, is_pca_transformed) for n in n_components_values]

    with Pool(n_cores) as pool:
        results_list = pool.map(run_gmm_wrapper, args)

    # Convert results list to dictionary
    results = {f"n_{n}": result for n, result in results_list}

    total_time = time.time() - start_time
    logger.info(f"Completed all GMM clustering in {total_time:.2f} seconds")
    return results


@st.cache_data
def load_and_preprocess_data(
    feature_flags,
    end_train_year=2024,
    min_ratings=25,
):
    """Load and preprocess data with caching."""
    # Load data
    data = load_bgg_data(end_train_year, min_ratings)

    # Create preprocessor
    preprocessor = create_preprocessor(
        create_designer_features=feature_flags["designer"],
        create_publisher_features=feature_flags["publisher"],
        create_artist_features=feature_flags["artist"],
        create_family_features=feature_flags["family"],
        create_category_features=feature_flags["category"],
        create_mechanic_features=feature_flags["mechanic"],
        include_base_numeric=feature_flags["base_numeric"],
    )

    # Preprocess data
    X = preprocessor.fit_transform(data)

    return data, X, preprocessor


def main():
    st.set_page_config(
        page_title="Board Games and Unsupervised Learning", layout="wide"
    )
    st.title("Unsupervised Learning and Board Games")

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = load_bgg_data()

    if "numeric_columns" not in st.session_state:
        st.session_state.numeric_columns = st.session_state.data.select_dtypes(
            include=[np.number]
        ).columns.tolist()

    if "clustering_results" not in st.session_state:
        st.session_state.clustering_results = None

    # Initialize all session state variables
    if "selected_k" not in st.session_state:
        st.session_state.selected_k = None
    if "k_select" not in st.session_state:
        st.session_state.k_select = None
    if "selected_cluster_explore" not in st.session_state:
        st.session_state.selected_cluster_explore = "Cluster 1"
    if "selected_cluster_importance" not in st.session_state:
        st.session_state.selected_cluster_importance = 0
    if "kmeans_x_component" not in st.session_state:
        st.session_state.kmeans_x_component = "PC1"
    if "kmeans_y_component" not in st.session_state:
        st.session_state.kmeans_y_component = "PC2"
    if "kmeans_y_select" not in st.session_state:
        st.session_state.kmeans_y_select = "PC2"
    if "gmm_x_component" not in st.session_state:
        st.session_state.gmm_x_component = "PC1"
    if "gmm_y_component" not in st.session_state:
        st.session_state.gmm_y_component = "PC2"
    if "gmm_results" not in st.session_state:
        st.session_state.gmm_results = None
    if "selected_n" not in st.session_state:
        st.session_state.selected_n = None
    if "n_values_input" not in st.session_state:
        st.session_state.n_values_input = "3, 5, 7, 10"
    if "feature_groups" not in st.session_state:
        st.session_state.feature_groups = [
            "Designer",
            "Publisher",
            "Artist",
            "Family",
            "Category",
            "Mechanic",
            "Base Numeric",
        ]

    if "feature_flags" not in st.session_state:
        st.session_state.feature_flags = {
            group.lower().replace(" ", "_"): True
            for group in st.session_state.feature_groups
        }

    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = create_preprocessor()
        st.session_state.X = st.session_state.preprocessor.fit_transform(
            st.session_state.data
        )

    # Sidebar for preprocessing options
    st.sidebar.header("Preprocessing Options")

    # Multiselect for feature groups
    selected_features = st.sidebar.multiselect(
        "Select Feature Groups",
        st.session_state.feature_groups,
        default=st.session_state.feature_groups[-3:],
        key="selected_features",
    )

    # Update feature flags and reprocess only when Apply is clicked
    if st.sidebar.button("Apply Preprocessing", key="apply_preprocessing"):
        with st.spinner("Updating preprocessing..."):
            # Update feature flags
            st.session_state.feature_flags = {
                group.lower().replace(" ", "_"): group in selected_features
                for group in st.session_state.feature_groups
            }

            # Create new preprocessor and transform data
            st.session_state.preprocessor = create_preprocessor(
                create_designer_features=st.session_state.feature_flags["designer"],
                create_publisher_features=st.session_state.feature_flags["publisher"],
                create_artist_features=st.session_state.feature_flags["artist"],
                create_family_features=st.session_state.feature_flags["family"],
                create_category_features=st.session_state.feature_flags["category"],
                create_mechanic_features=st.session_state.feature_flags["mechanic"],
                include_base_numeric=st.session_state.feature_flags["base_numeric"],
            )
            st.session_state.X = st.session_state.preprocessor.fit_transform(
                st.session_state.data
            )
            st.sidebar.success("Preprocessing updated successfully!")

    # Dimension reduction options
    pca_components = st.sidebar.slider(
        "PCA Variance Explained", min_value=0.0, max_value=0.9, value=0.15, step=0.05
    )
    # n_components = st.sidebar.slider(
    #     "Number of Components", min_value=2, max_value=50, value=50
    # )

    # Tabs for different analyses
    tab_data, tab_pca, tab_kmeans, tab_gmm, tab_neighbor = st.tabs(
        ["Data", "PCA", "K-Means", "GMM", "Find a Neighbor"]
    )

    # Cache PCA results
    @st.cache_data
    def compute_pca(_X, pca_components, feature_flags=None):
        try:
            # Ensure input data is numeric and finite
            X_numeric = _X.select_dtypes(include=[np.number])
            # Drop year_published_transformed column if it exists
            if "year_published_transformed" in X_numeric.columns:
                X_numeric = X_numeric.drop("year_published_transformed", axis=1)
            X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).dropna()

            # Validate input data
            if X_numeric.empty:
                return None, "No valid numeric data for PCA"

            # Perform PCA
            pca_results = perform_pca(X_numeric, n_components=pca_components)

            # Validate PCA results
            if not pca_results or len(pca_results.get("transformed_data", [])) == 0:
                return None, "PCA failed to transform data"

            # Prepare visualization data
            X_reduced = pca_results["transformed_data"]
            viz_data = X_reduced.copy()
            viz_data["Predicted Complexity"] = st.session_state.data[
                "predicted_complexity"
            ]
            viz_data["Year Published"] = st.session_state.data["year_published"]

            # Add specific columns for hover
            for col in ["game_id", "name", "year_published"]:
                if col in st.session_state.data.columns:
                    viz_data[col] = st.session_state.data[col]

            return pca_results, viz_data, None

        except Exception as e:
            return None, None, str(e)

    # Perform PCA
    with st.spinner("Performing PCA..."):
        pca_results, viz_data, error = compute_pca(
            st.session_state.X, pca_components, st.session_state.feature_flags
        )
        if error:
            st.error(f"Error performing PCA: {error}")
            pca_results = None

    # Data Sample Tab
    with tab_data:
        st.header("Data")
        # Allow user to control sample size
        sample_size = st.slider(
            "Sample Size", min_value=50, max_value=25000, step=50, value=10000
        )

        # Select columns to display
        all_columns = st.session_state.data.columns.tolist()
        default_columns = [
            "game_id",
            "name",
            "year_published",
            "geek_rating",
            "users_rated",
            "average_rating",
            "predicted_complexity",
            "min_players",
            "max_players",
            "min_playtime",
            "max_playtime",
            "min_age",
            "description",
            "categories",
            "mechanics",
            "publishers",
            "designers",
            "artists",
            "families",
        ]

        # Ensure default columns exist in the dataframe
        default_columns = [col for col in default_columns if col in all_columns]

        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=default_columns,
            placeholder="Choose columns...",
        )

        # Generate sample data
        sample_data = st.session_state.data.sample(n=sample_size, random_state=42)[
            selected_columns
        ]
        st.dataframe(sample_data, use_container_width=True)

        # Use numeric columns from session state
        numeric_columns = st.session_state.numeric_columns

        # Cache default columns
        if "default_x" not in st.session_state:
            st.session_state.default_x = (
                "predicted_complexity"
                if "predicted_complexity" in numeric_columns
                else numeric_columns[0]
            )
        if "default_y" not in st.session_state:
            st.session_state.default_y = (
                "max_playtime"
                if "max_playtime" in numeric_columns
                else (
                    numeric_columns[1]
                    if len(numeric_columns) > 1
                    else st.session_state.default_x
                )
            )

        # Scatter plot controls
        st.header("Data Visualization")
        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox(
                "X-axis",
                numeric_columns,
                index=numeric_columns.index(st.session_state.default_x),
                key="scatter_x",
            )

        with col2:
            y_column = st.selectbox(
                "Y-axis",
                numeric_columns,
                index=numeric_columns.index(st.session_state.default_y),
                key="scatter_y",
            )

        # Log scale options
        col1, col2 = st.columns(2)
        with col1:
            log_x = st.checkbox("Log Scale X-axis", key="scatter_log_x")
        with col2:
            log_y = st.checkbox("Log Scale Y-axis", key="scatter_log_y", value=True)

        # Create scatter plot function
        def create_scatter_plot(data, x_col, y_col, log_x=False, log_y=False):
            """Create scatter plot for raw data visualization."""
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                opacity=0.4,
                hover_data=["game_id", "name", "year_published"],
                labels={
                    x_col: x_col.replace("_", " ").title(),
                    y_col: y_col.replace("_", " ").title(),
                },
            )

            if log_x:
                fig.update_xaxes(type="log")
            if log_y:
                fig.update_yaxes(type="log")

            return fig

        # Create and display scatter plot
        fig = create_scatter_plot(
            st.session_state.data, x_column, y_column, log_x, log_y
        )

        st.plotly_chart(fig, use_container_width=True)

        # Distribution plots
        st.subheader("Feature Distributions")

        # Select features for distribution plots
        dist_columns = st.multiselect(
            "Select Features for Distribution Analysis",
            numeric_columns,
            default=numeric_columns[7:12],
            key="dist_features",
        )

        if dist_columns:
            # Log scale option for distributions
            log_dist = st.checkbox("Log Scale for Distributions", key="dist_log")

            # Distribution plot creation function
            def create_distribution_plot(data, column, use_log_scale=False):
                fig = px.histogram(
                    data,
                    x=column,
                    title=f"Distribution of {column.replace('_', ' ').title()}",
                    marginal="box",
                )
                if use_log_scale:
                    fig.update_xaxes(type="log")
                return fig

            # Create distribution plots
            for i in range(0, len(dist_columns), 2):
                col1, col2 = st.columns(2)

                # First column
                with col1:
                    if i < len(dist_columns):
                        fig_dist = create_distribution_plot(
                            st.session_state.data, dist_columns[i], log_dist
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                # Second column
                with col2:
                    if i + 1 < len(dist_columns):
                        fig_dist = create_distribution_plot(
                            st.session_state.data, dist_columns[i + 1], log_dist
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

    # PCA Tab
    with tab_pca:
        st.header("Principal Component Analysis (PCA)")

        # Component Selection
        st.subheader("PCA Visualization")

        if pca_results is not None and viz_data is not None:
            # Get available components dynamically
            available_components = [
                col for col in viz_data.columns if col.startswith("PC")
            ]

            # Ensure valid default selections
            if st.session_state.kmeans_x_component not in available_components:
                st.session_state.kmeans_x_component = (
                    available_components[0] if available_components else None
                )

            if st.session_state.kmeans_y_component not in available_components:
                st.session_state.kmeans_y_component = (
                    available_components[1]
                    if len(available_components) > 1
                    else available_components[0]
                    if available_components
                    else None
                )

            # Component selection for scatter plot
            col1, col2 = st.columns(2)
            with col1:
                x_component = (
                    st.selectbox(
                        "X-axis Component",
                        available_components,
                        index=(
                            available_components.index(
                                st.session_state.kmeans_x_component
                            )
                            if st.session_state.kmeans_x_component
                            in available_components
                            else 0
                        ),
                    )
                    if available_components
                    else st.write("Not enough components for visualization")
                )
            with col2:
                y_component = (
                    st.selectbox(
                        "Y-axis Component",
                        available_components,
                        index=(
                            available_components.index(
                                st.session_state.kmeans_y_component
                            )
                            if st.session_state.kmeans_y_component
                            in available_components
                            else (1 if len(available_components) > 1 else 0)
                        ),
                    )
                    if available_components
                    else st.write("Not enough components for visualization")
                )

            def create_pca_scatter(
                data, x_comp, y_comp, color=None, cluster_labels=None
            ):
                """Create scatter plot for PCA visualization with optional clustering."""
                if cluster_labels is not None:
                    n_clusters = len(np.unique(cluster_labels))
                    color_map = get_cluster_colors(n_clusters)
                    # Convert cluster labels to "Cluster X" format
                    cluster_names = [f"Cluster {i + 1}" for i in range(n_clusters)]
                    cluster_labels_str = np.array(
                        [f"Cluster {i + 1}" for i in cluster_labels]
                    )

                    fig = px.scatter(
                        data,
                        x=x_comp,
                        y=y_comp,
                        color=cluster_labels_str,
                        hover_data=["game_id", "name", "year_published"],
                        title=f"PCA Scatter Plot with Clustering: {x_comp} vs {y_comp}",
                        color_discrete_sequence=[
                            color_map[str(i)] for i in range(n_clusters)
                        ],
                        labels={"color": "Cluster"},
                        category_orders={"color": cluster_names},
                    )
                elif color == "None":
                    fig = px.scatter(
                        data,
                        x=x_comp,
                        y=y_comp,
                        hover_data=["game_id", "name", "year_published"],
                        title=f"PCA Scatter Plot: {x_comp} vs {y_comp}",
                    )
                else:
                    fig = px.scatter(
                        data,
                        x=x_comp,
                        y=y_comp,
                        opacity=0.4,
                        color=color,
                        color_continuous_scale="viridis",
                        hover_data=["game_id", "name", "year_published"],
                        title=f"PCA Scatter Plot: {x_comp} vs {y_comp}",
                    )

                fig.update_layout(
                    height=600,
                    xaxis=dict(
                        zeroline=True,
                        zerolinewidth=2,
                        showgrid=False,
                    ),
                    yaxis=dict(
                        zeroline=True,
                        zerolinewidth=2,
                        showgrid=False,
                    ),
                )
                return fig

            # Create and display PCA scatter plot
            fig_scatter_pca = create_pca_scatter(
                viz_data, x_component, y_component, color=None
            )
            st.plotly_chart(fig_scatter_pca, use_container_width=True)

        else:
            st.warning(
                "PCA results are not available. Please check the error message above."
            )

        # Biplot
        if pca_results is not None:
            # Get loadings
            loadings = pca_results["loadings"]

            # Validate component selection
            available_components = loadings.columns.tolist()
            if (
                x_component not in available_components
                or y_component not in available_components
            ):
                st.warning(
                    f"Selected components {x_component} and {y_component} are not available. "
                    f"Available components are: {', '.join(available_components)}"
                )
                top_features = pd.DataFrame()
            else:
                # Calculate absolute loadings for selected components
                loadings_subset = loadings.copy()
                loadings_subset["abs_loading"] = np.sqrt(
                    loadings_subset[x_component] ** 2
                    + loadings_subset[y_component] ** 2
                )

                # Get top 10 features by absolute loading
                top_features = loadings_subset.nlargest(20, "abs_loading")

            # Create biplot with improved visibility
            fig_biplot = go.Figure()

            # Add feature vectors with arrows
            for _, row in top_features.iterrows():
                # Calculate angle for the vector
                angle = np.arctan2(row[y_component], row[x_component])

                # Calculate text position with smart offset to avoid overlap
                # Use angle-based positioning to spread labels better
                offset_scale = 0.07  # Controls how far labels are from their points
                text_x = row[x_component] + offset_scale * np.cos(angle + np.pi / 6)
                text_y = row[y_component] + offset_scale * np.sin(angle + np.pi / 6)

                # Add vector with arrow head
                fig_biplot.add_trace(
                    go.Scatter(
                        x=[0, row[x_component]],
                        y=[0, row[y_component]],
                        mode="lines",
                        line=dict(
                            width=2,
                            color="white",
                        ),
                        name=row["feature"],
                        text=[row["feature"]],  # Add feature name for hover
                        hoverinfo="text+name",
                        hovertemplate=f"Feature: {row['feature']}<br>{x_component}: %{{x:.4f}}<br>{y_component}: %{{y:.4f}}<extra></extra>",
                        hoverlabel=dict(
                            bgcolor="rgba(50, 50, 50, 0.8)", font=dict(size=14)
                        ),
                    )
                )

                # Add feature label with improved visibility
                fig_biplot.add_annotation(
                    x=text_x,
                    y=text_y,
                    text=row["feature"],
                    showarrow=True,
                    arrowhead=0,
                    arrowwidth=1,
                    arrowcolor="lightblue",
                    ax=row[x_component],
                    ay=row[y_component],
                    font=dict(
                        size=10,
                        color="lightblue",
                    ),
                    bgcolor="rgba(255, 255, 255, 0.1)",
                    bordercolor="rgba(255, 255, 255, 0.3)",
                    borderwidth=1,
                    borderpad=4,
                )

            # Update layout with improved styling
            fig_biplot.update_layout(
                title=dict(
                    text=f"Feature Loadings: {x_component} vs {y_component}",
                    font=dict(size=24),
                    y=0.95,
                ),
                xaxis_title=dict(text=x_component, font=dict(size=16)),
                yaxis_title=dict(text=y_component, font=dict(size=16)),
                showlegend=False,
                xaxis=dict(
                    range=[-0.5, 0.5],  # Zoomed in range
                    zeroline=True,
                    zerolinewidth=2,
                    showgrid=False,
                ),
                yaxis=dict(
                    range=[-0.5, 0.5],  # Zoomed in range
                    zeroline=True,
                    zerolinewidth=2,
                    showgrid=False,
                ),
                height=600,
            )

            st.plotly_chart(fig_biplot, use_container_width=True)

            # Loadings Charts
            st.subheader("Feature Loadings")

            # Filter available components to only PC columns
            pc_components = [col for col in loadings.columns if col.startswith("PC")]

            loadings_component = st.selectbox("Component", pc_components, index=0)

            # Create separate charts for each component
            for component in [loadings_component]:
                # Filter loadings to only include PC columns
                pc_columns = [col for col in loadings.columns if col.startswith("PC")]

                # Get absolute loadings for this component
                component_loadings = loadings[pc_columns + ["feature"]].copy()

                # Check if the selected component is in PC columns
                if component not in pc_columns:
                    st.warning(
                        f"Component {component} is not a valid PC column. Skipping absolute loading calculation."
                    )
                    continue

                try:
                    import pandas as pd  # Ensure pandas is imported in this scope

                    component_loadings["AbsLoading"] = np.abs(
                        component_loadings[component].apply(
                            lambda x: float(x) if pd.notnull(x) else 0
                        )
                    )

                    # Get top 25 features by absolute loading
                    top_features = component_loadings.nlargest(25, "AbsLoading")

                    # Sort features by loading value for better visualization
                    top_features = top_features.sort_values(component)
                except Exception as e:
                    st.warning(f"Error processing loadings for {component}: {str(e)}")
                    top_features = pd.DataFrame()  # Provide a default empty DataFrame
                    continue

                # Create horizontal bar chart with conditional colors
                fig_loadings = go.Figure(
                    data=[
                        go.Bar(
                            y=top_features["feature"],  # Features on y-axis
                            x=top_features[component],  # Loadings on x-axis
                            orientation="h",  # Horizontal bars
                            marker_color=[
                                "aliceblue" if x >= 0 else "lightcoral"
                                for x in top_features[component]
                            ],
                            hovertemplate="Feature: %{y}<br>Loading: %{x:.4f}<extra></extra>",
                        )
                    ]
                )

                # Update layout
                fig_loadings.update_layout(
                    title=f"Top 25 Feature Loadings for {component}",
                    xaxis_title="Loading Value",
                    yaxis_title="Features",
                    height=600,  # Increase height to accommodate all features
                    margin=dict(l=200),  # Increase left margin for feature names
                    showlegend=False,
                )

                # Display the plot
                st.plotly_chart(fig_loadings, use_container_width=True)

            # Explained Variance Plot
            st.subheader("Variance Explained")

            # Prepare data for plotting
            components = list(
                range(1, len(pca_results["explained_variance_ratio"]) + 1)
            )
            individual_variance = pca_results["explained_variance_ratio"] * 100
            cumulative_variance = pca_results["cumulative_explained_variance"] * 100

            # Create figure
            fig = go.Figure()

            # Bar plot for individual variance
            fig.add_trace(
                go.Bar(
                    x=components,
                    y=individual_variance,
                    name="Individual Variance",
                    marker_color="blue",
                    opacity=0.6,
                    hovertemplate="Component %{x}<br>Individual Variance: %{y:.2f}%<extra></extra>",
                )
            )

            # Line plot for cumulative variance
            fig.add_trace(
                go.Scatter(
                    x=components,
                    y=cumulative_variance,
                    mode="lines",
                    name="Cumulative Variance",
                    marker_color="darkorchid",
                    hovertemplate="Components: %{x}<br>Cumulative Variance: %{y:.2f}%<extra></extra>",
                )
            )

            # Customize layout
            fig.update_layout(
                title="Variance Explained by Principal Components",
                xaxis_title="Number of Principal Components",
                yaxis_title="Variance Explained (%)",
                legend_title="Variance Type",
                hovermode="x unified",
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

    # K-Means Tab
    with tab_kmeans:
        st.header("K-Means Clustering")

        if pca_results is not None and viz_data is not None:
            # Initialize k values in session state if not present
            if "k_values_input" not in st.session_state:
                st.session_state.k_values_input = "3, 5, 7, 10"

            # Input for k values
            k_values_input = st.text_input(
                "Enter k values to test (comma-separated numbers)",
                value=st.session_state.k_values_input,
                help="Enter numbers separated by commas, e.g. '3, 5, 7, 13'",
                key="k_input",
            )

            # Update session state only when input changes
            if k_values_input != st.session_state.k_values_input:
                st.session_state.k_values_input = k_values_input
                st.session_state.clustering_results = (
                    None  # Clear results to force recomputation
                )

            # Parse and validate k values
            try:
                k_values = [int(k.strip()) for k in k_values_input.split(",")]
                k_values = [k for k in k_values if k >= 2]  # Filter valid k values
                if not k_values:
                    st.error("Please enter valid k values (numbers >= 2)")
                elif len(k_values) > 15:
                    st.warning("Large number of k values may increase computation time")
            except ValueError:
                st.error("Invalid input. Please enter numbers separated by commas.")
                k_values = None

            # Button to run clustering
            if st.button("Run K-Means Clustering"):
                if k_values:
                    logger.info(
                        f"Starting K-Means clustering analysis with k values: {k_values}"
                    )
                    with st.spinner(
                        "Running K-Means clustering for multiple k values..."
                    ):
                        # Get PCA transformed data for clustering
                        clustering_data = viz_data[
                            [col for col in viz_data.columns if col.startswith("PC")]
                        ]

                        # Run clustering with user-specified k values and feature flags
                        st.session_state.clustering_results = perform_multi_k_means(
                            clustering_data,
                            k_values=k_values,
                            feature_flags=st.session_state.feature_flags,
                        )
                        logger.info("K-Means clustering completed")

                        # Initialize with first k value
                        st.session_state.selected_k = f"k_{k_values[0]}"

            # Show clustering results if available
            if st.session_state.clustering_results is not None:
                # Create tabs for different metrics
                metrics_tab1, metrics_tab2 = st.tabs(
                    ["Clustering Metrics", "Elbow Plot"]
                )

                with metrics_tab1:
                    # Get all scores
                    silhouette_scores = {
                        k: results["silhouette"]
                        for k, results in st.session_state.clustering_results.items()
                    }
                    calinski_scores = {
                        k: results["calinski_harabasz"]
                        for k, results in st.session_state.clustering_results.items()
                    }
                    davies_scores = {
                        k: results["davies_bouldin"]
                        for k, results in st.session_state.clustering_results.items()
                    }

                    k_values = [int(k.split("_")[1]) for k in silhouette_scores.keys()]

                    # Create combined metrics plot
                    fig_metrics = go.Figure()

                    # Add traces for each metric
                    fig_metrics.add_trace(
                        go.Scatter(
                            x=k_values,
                            y=list(silhouette_scores.values()),
                            mode="lines+markers",
                            name="Silhouette Score",
                            hovertemplate="k=%{x}<br>Score=%{y:.4f}<extra></extra>",
                        )
                    )

                    # Normalize Calinski-Harabasz scores for plotting
                    ch_scores = list(calinski_scores.values())
                    ch_range = max(ch_scores) - min(ch_scores)
                    ch_norm = [
                        (x - min(ch_scores)) / ch_range if ch_range > 0 else 1.0
                        for x in ch_scores
                    ]
                    fig_metrics.add_trace(
                        go.Scatter(
                            x=k_values,
                            y=ch_norm,
                            mode="lines+markers",
                            name="Calinski-Harabasz (normalized)",
                            hovertemplate="k=%{x}<br>Score=%{y:.4f}<extra></extra>",
                        )
                    )

                    # Davies-Bouldin score (lower is better, so we invert it)
                    db_scores = list(davies_scores.values())
                    db_range = max(db_scores) - min(db_scores)
                    db_norm = [
                        1 - ((x - min(db_scores)) / db_range if db_range > 0 else 0.0)
                        for x in db_scores
                    ]
                    fig_metrics.add_trace(
                        go.Scatter(
                            x=k_values,
                            y=db_norm,
                            mode="lines+markers",
                            name="Davies-Bouldin (inverted, normalized)",
                            hovertemplate="k=%{x}<br>Score=%{y:.4f}<extra></extra>",
                        )
                    )

                    fig_metrics.update_layout(
                        title="Clustering Metrics Comparison",
                        xaxis_title="Number of Clusters (k)",
                        yaxis_title="Score",
                        showlegend=True,
                    )

                    st.plotly_chart(fig_metrics, use_container_width=True)

                    # # Display raw scores in a table
                    # metrics_df = pd.DataFrame(
                    #     {
                    #         "k": k_values,
                    #         "Silhouette Score": list(silhouette_scores.values()),
                    #         "Calinski-Harabasz Score": list(calinski_scores.values()),
                    #         "Davies-Bouldin Score": list(davies_scores.values()),
                    #     }
                    # ).set_index("k")

                    # st.write("Raw Clustering Metrics:")
                    # st.dataframe(metrics_df.round(4), use_container_width=True)

                with metrics_tab2:
                    # Create elbow plot
                    inertias = {
                        k: results["inertia"]
                        for k, results in st.session_state.clustering_results.items()
                    }

                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(
                        go.Scatter(
                            x=k_values,
                            y=list(inertias.values()),
                            mode="lines+markers",
                            name="Inertia",
                            hovertemplate="k=%{x}<br>Inertia=%{y:.0f}<extra></extra>",
                        )
                    )

                    fig_elbow.update_layout(
                        title="Elbow Plot (Within-Cluster Sum of Squares)",
                        xaxis_title="Number of Clusters (k)",
                        yaxis_title="Inertia",
                        showlegend=False,
                    )

                    st.plotly_chart(fig_elbow, use_container_width=True)

                # Dropdown to select k with state management
                k_options = list(st.session_state.clustering_results.keys())

                def on_k_change():
                    st.session_state.selected_k = st.session_state.k_select
                    logger.info(
                        f"Selected clustering k changed to {st.session_state.selected_k}"
                    )

                st.selectbox(
                    "Select number of clusters (k)",
                    k_options,
                    index=k_options.index(st.session_state.selected_k),
                    format_func=lambda x: f"k={x.split('_')[1]} (silhouette={silhouette_scores[x]:.4f})",
                    key="k_select",
                    on_change=on_k_change,
                )

                # Update scatter plot with cluster colors
                st.subheader("Cluster Visualization")

                # Get PCA components for visualization
                col1, col2 = st.columns(2)
                with col1:
                    components = [f"PC{i + 1}" for i in range(viz_data.shape[1] - 3)]

                    # Ensure selected x component is valid
                    if st.session_state.kmeans_x_component not in components:
                        st.session_state.kmeans_x_component = components[0]

                    def on_x_component_change():
                        st.session_state.kmeans_x_component = (
                            st.session_state.kmeans_x_select
                        )

                    st.selectbox(
                        "X-axis Component",
                        components,
                        index=components.index(st.session_state.kmeans_x_component),
                        key="kmeans_x_select",
                        on_change=on_x_component_change,
                    )
                with col2:
                    # Ensure selected y component is valid
                    if st.session_state.kmeans_y_component not in components:
                        st.session_state.kmeans_y_component = (
                            components[1] if len(components) > 1 else components[0]
                        )

                    def on_y_component_change():
                        st.session_state.kmeans_y_component = (
                            st.session_state.kmeans_y_select
                        )

                    st.selectbox(
                        "Y-axis Component",
                        components,
                        index=components.index(st.session_state.kmeans_y_component),
                        key="kmeans_y_select",
                        on_change=on_y_component_change,
                    )

                # Get cluster labels
                cluster_labels = st.session_state.clustering_results[
                    st.session_state.selected_k
                ]["labels"]

                # Create and display clustering scatter plot
                fig_scatter_kmeans = create_pca_scatter(
                    viz_data,
                    st.session_state.kmeans_x_component,
                    st.session_state.kmeans_y_component,
                    cluster_labels=cluster_labels,
                )
                st.plotly_chart(fig_scatter_kmeans, use_container_width=True)

                # New PCA Embeddings Visualization
                st.subheader("PCA Embeddings Visualization")

                # Add sample size slider
                sample_size = st.slider(
                    "Number of Games to Sample",
                    min_value=100,
                    max_value=5000,
                    value=500,
                    step=100,
                    help="Select how many games to display in the embeddings plot",
                )

                # Sample random games
                sample_indices = np.random.choice(
                    len(viz_data), sample_size, replace=False
                )
                sample_data = viz_data.iloc[sample_indices]
                sample_labels = cluster_labels[sample_indices]

                # Prepare data for line plot
                pc_columns = [
                    col for col in sample_data.columns if col.startswith("PC")
                ]

                # Color mapping
                n_clusters = int(st.session_state.selected_k.split("_")[1])
                color_map = get_cluster_colors(n_clusters)

                # Prepare data for color mapping
                cluster_colors = [  # noqa: F841
                    color_map[str(cluster)] for cluster in sample_labels
                ]

                # Create line plot
                fig_embeddings = go.Figure()

                # Create traces for each cluster
                for cluster in range(n_clusters):
                    # Get games for this cluster
                    cluster_mask = sample_labels == cluster
                    cluster_data = sample_data[cluster_mask]

                    # Add traces for games in this cluster
                    for _, game_row in cluster_data.iterrows():
                        fig_embeddings.add_trace(
                            go.Scatter(
                                x=pc_columns,
                                y=game_row[pc_columns],
                                mode="lines",
                                name=f"Cluster {cluster + 1}",
                                line=dict(color=color_map[str(cluster)], width=1),
                                opacity=0.3,
                                hovertemplate=(
                                    "Game: "
                                    + str(
                                        st.session_state.data.loc[game_row.name, "name"]
                                    )
                                    + "<br>Cluster: "
                                    + str(cluster + 1)
                                    + "<br>PC Values: %{y:.4f}<extra></extra>"
                                ),
                                legendgroup=f"Cluster {cluster + 1}",
                                showlegend=False,
                            )
                        )

                    # Add a single legend entry for this cluster
                    if len(cluster_data) > 0:
                        fig_embeddings.add_trace(
                            go.Scatter(
                                x=[None],
                                y=[None],
                                mode="lines",
                                name=f"Cluster {cluster + 1}",
                                line=dict(color=color_map[str(cluster)], width=2),
                                legendgroup=f"Cluster {cluster + 1}",
                                showlegend=True,
                            )
                        )

                # Customize layout
                fig_embeddings.update_layout(
                    title=f"PCA Embeddings for {sample_size} Sampled Games",
                    xaxis_title="Principal Components",
                    yaxis_title="Loading Value",
                    height=600,
                    xaxis=dict(
                        zeroline=True,
                        zerolinewidth=2,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(128, 128, 128, 0.2)",
                    ),
                    yaxis=dict(
                        zeroline=True,
                        zerolinewidth=2,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="rgba(128, 128, 128, 0.2)",
                    ),
                    plot_bgcolor="rgba(0, 0, 0, 0)",
                )

                # Display the plot
                st.plotly_chart(fig_embeddings, use_container_width=True)

                # Show cluster analysis
                (
                    cluster_analysis_tab1,
                    cluster_analysis_tab2,
                    cluster_analysis_tab3,
                    cluster_analysis_tab4,
                ) = st.tabs(
                    [
                        "Cluster Sizes",
                        "Feature Importance",
                        "Silhouette Distribution",
                        "Silhouette Scores",
                    ]
                )

                with cluster_analysis_tab1:
                    # Show cluster sizes
                    # Convert cluster labels to "Cluster X" format
                    cluster_sizes = (
                        pd.Series([f"Cluster {i + 1}" for i in cluster_labels])
                        .value_counts()
                        .sort_index()
                    )
                    n_clusters = len(cluster_sizes)
                    color_map = get_cluster_colors(n_clusters)
                    fig_sizes = px.bar(
                        x=cluster_sizes.index,
                        y=cluster_sizes.values,
                        labels={"x": "Cluster", "y": "Number of Games"},
                        title=f"Distribution of Games Across {st.session_state.selected_k} Clusters",
                        color=cluster_sizes.index,
                        color_discrete_sequence=[
                            color_map[str(i)] for i in range(n_clusters)
                        ],
                        category_orders={
                            "color": [f"Cluster {i + 1}" for i in range(n_clusters)]
                        },
                    )
                    st.plotly_chart(fig_sizes, use_container_width=True)

                with cluster_analysis_tab2:
                    # Show feature importance per cluster
                    feature_importance = st.session_state.clustering_results[
                        st.session_state.selected_k
                    ]["feature_importance"]

                    # Create feature importance visualization with state management
                    def on_importance_cluster_change():
                        st.session_state.selected_cluster_importance = (
                            int(st.session_state.importance_cluster_select.split()[-1])
                            - 1
                        )

                    cluster_names = [f"Cluster {i + 1}" for i in range(n_clusters)]
                    selected_cluster_name = st.selectbox(
                        "Select Cluster for Feature Analysis",
                        cluster_names,
                        key="importance_cluster_select",
                        index=st.session_state.selected_cluster_importance,
                        on_change=on_importance_cluster_change,
                    )
                    selected_cluster = int(selected_cluster_name.split()[-1]) - 1

                    # Get feature importance for selected cluster
                    cluster_importance = feature_importance[selected_cluster]
                    importance_df = pd.DataFrame(
                        {
                            "Feature": cluster_importance.index,
                            "Importance": cluster_importance.values,
                        }
                    ).sort_values("Importance", ascending=True)

                    # Show top 15 most important features
                    n_clusters = int(st.session_state.selected_k.split("_")[1])
                    color_map = get_cluster_colors(n_clusters)
                    cluster_color = color_map[str(selected_cluster)]

                    fig_importance = go.Figure()
                    fig_importance.add_trace(
                        go.Bar(
                            y=importance_df["Feature"].tail(15),
                            x=importance_df["Importance"].tail(15),
                            orientation="h",
                            marker_color=cluster_color,
                        )
                    )

                    fig_importance.update_layout(
                        title=f"Top 15 Most Important Features for {selected_cluster_name}",
                        xaxis_title="Feature Importance (Z-score)",
                        yaxis_title="Feature",
                        height=600,
                    )

                    st.plotly_chart(fig_importance, use_container_width=True)

                with cluster_analysis_tab3:
                    # Get silhouette values for each sample
                    silhouette_values = st.session_state.clustering_results[
                        st.session_state.selected_k
                    ]["sample_silhouette_values"]

                    # Create a DataFrame with game info and silhouette scores
                    silhouette_df = pd.DataFrame(
                        {
                            "game_id": st.session_state.data["game_id"],
                            "name": st.session_state.data["name"],
                            "cluster": [f"Cluster {i + 1}" for i in cluster_labels],
                            "silhouette_score": silhouette_values,
                        }
                    )

                    # Sort by silhouette score within each cluster
                    silhouette_df = silhouette_df.sort_values(
                        ["cluster", "silhouette_score"], ascending=[True, False]
                    )

                    # Create box plot with individual points
                    n_clusters = int(st.session_state.selected_k.split("_")[1])
                    color_map = get_cluster_colors(n_clusters)

                    fig_silhouette = go.Figure()

                    # Add box plots for each cluster
                    for i in range(n_clusters):
                        cluster_name = f"Cluster {i + 1}"
                        cluster_data = silhouette_df[
                            silhouette_df["cluster"] == cluster_name
                        ]

                        # Add box plot
                        fig_silhouette.add_trace(
                            go.Box(
                                y=cluster_data["silhouette_score"],
                                name=cluster_name,
                                boxpoints="all",  # Show all points
                                jitter=0.3,  # Add jitter to points for better visibility
                                pointpos=-1.8,  # Position points to the left of box
                                marker=dict(color=color_map[str(i)], size=4),
                                hovertemplate=(
                                    "Game: %{customdata[0]}<br>"
                                    + "Score: %{y:.4f}<br>"
                                    + "<extra></extra>"
                                ),
                                customdata=cluster_data[["name"]].values,
                            )
                        )

                    # Add horizontal line for average silhouette score
                    avg_score = silhouette_scores[st.session_state.selected_k]
                    fig_silhouette.add_shape(
                        type="line",
                        x0=0,
                        x1=1,
                        y0=avg_score,
                        y1=avg_score,
                        line=dict(
                            color="red",
                            dash="dash",
                        ),
                        xref="paper",
                        yref="y",
                    )
                    fig_silhouette.add_annotation(
                        text=f"Average silhouette score: {avg_score:.4f}",
                        x=1,
                        y=avg_score,
                        xref="paper",
                        yref="y",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        xshift=10,
                    )

                    fig_silhouette.update_layout(
                        title="Distribution of Silhouette Scores by Cluster",
                        yaxis_title="Silhouette Score (-1 to 1, higher is better)",
                        height=600,
                        showlegend=True,
                        hovermode="closest",
                    )

                    st.plotly_chart(fig_silhouette, use_container_width=True)

                    # Add table view of silhouette scores
                    st.subheader("Individual Game Silhouette Scores")

                    # Allow filtering by cluster
                    selected_cluster = st.selectbox(
                        "Filter by Cluster",
                        ["All Clusters"]
                        + [f"Cluster {i + 1}" for i in range(n_clusters)],
                    )

                    # Filter data based on selection
                    if selected_cluster == "All Clusters":
                        filtered_df = silhouette_df
                    else:
                        filtered_df = silhouette_df[
                            silhouette_df["cluster"] == selected_cluster
                        ]

                    # Display the data
                    st.dataframe(
                        filtered_df[
                            ["name", "game_id", "cluster", "silhouette_score"]
                        ].sort_values("silhouette_score", ascending=False),
                        use_container_width=True,
                        height=400,
                    )

                    # # Add tooltip explanation
                    # st.help(
                    #     """
                    # Interpreting the Silhouette Distribution:
                    # - Each cluster's distribution is shown in a different color
                    # - Higher silhouette scores (closer to 1) indicate better clustering
                    # - The dashed red line shows the average silhouette score
                    # - Well-defined clusters have distributions skewed towards higher scores
                    # - Overlapping distributions may indicate similar clusters
                    # - Scores below 0 suggest potential misclassification
                    # """
                    # )

                with cluster_analysis_tab4:
                    # Add summary statistics
                    st.subheader("Silhouette Score Statistics by Cluster")
                    stats = []
                    for i in range(int(st.session_state.selected_k.split("_")[1])):
                        cluster_silhouette = silhouette_values[cluster_labels == i]
                        stats.append(
                            {
                                "Cluster": f"Cluster {i + 1}",
                                "Mean": np.mean(cluster_silhouette),
                                "Median": np.median(cluster_silhouette),
                                "Std": np.std(cluster_silhouette),
                                "Min": np.min(cluster_silhouette),
                                "Max": np.max(cluster_silhouette),
                                "Size": len(cluster_silhouette),
                            }
                        )

                    stats_df = pd.DataFrame(stats).round(4)
                    st.dataframe(stats_df, use_container_width=True)

                # Add cluster exploration section
                st.subheader("Explore Clusters")

                # Create session state for cluster selection if it doesn't exist
                if "selected_cluster_explore" not in st.session_state:
                    st.session_state.selected_cluster_explore = "Cluster 1"

                # Select cluster to explore with state management
                def on_explore_cluster_change():
                    st.session_state.selected_cluster_explore = (
                        st.session_state.cluster_explorer_select
                    )

                # Ensure n_clusters is defined
                n_clusters = len(np.unique(cluster_labels))
                cluster_names = [f"Cluster {i + 1}" for i in range(n_clusters)]

                # Validate the current selected cluster
                if st.session_state.selected_cluster_explore not in cluster_names:
                    st.session_state.selected_cluster_explore = "Cluster 1"

                selected_cluster_name = st.selectbox(
                    "Select Cluster to Explore",
                    cluster_names,
                    key="cluster_explorer_select",
                    index=cluster_names.index(
                        st.session_state.selected_cluster_explore
                    ),
                    on_change=on_explore_cluster_change,
                )

                # Get games in selected cluster
                selected_cluster_idx = int(selected_cluster_name.split()[-1]) - 1
                cluster_mask = cluster_labels == selected_cluster_idx
                cluster_games = st.session_state.data[cluster_mask].copy()

                # Select columns to display
                all_columns = st.session_state.data.columns.tolist()
                default_columns = [
                    "game_id",
                    "name",
                    "year_published",
                    "geek_rating",
                    "users_rated",
                    "average_rating",
                    "predicted_complexity",
                    "min_players",
                    "max_players",
                    "min_playtime",
                    "max_playtime",
                    "min_age",
                    "description",
                    "categories",
                    "mechanics",
                ]
                default_columns = [col for col in default_columns if col in all_columns]

                # Display games in selected cluster with matching color
                if not cluster_games.empty and selected_columns:
                    # Get color for this cluster from color mapping
                    n_clusters = int(st.session_state.selected_k.split("_")[1])
                    color_map = get_cluster_colors(n_clusters)

                    # Safely extract cluster index from selected_cluster_explore
                    cluster_idx = (
                        int(st.session_state.selected_cluster_explore.split()[-1]) - 1
                    )
                    cluster_color = color_map[str(cluster_idx)]
                    cluster_name = f"Cluster {cluster_idx + 1}"

                    # Apply cluster color styling only to game_id and name columns
                    styled_df = cluster_games[selected_columns]

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=400,
                    )
                    st.markdown(
                        f'<p style="color: {cluster_color}">Showing {len(cluster_games)} games in {cluster_name}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(
                        "No games found in selected cluster or no columns selected."
                    )
        else:
            st.warning(
                "PCA results are not available. Please compute PCA first in the PCA tab."
            )

    # GMM Tab
    with tab_gmm:
        st.header("Gaussian Mixture Model Clustering")

        if pca_results is not None and viz_data is not None:
            # Initialize n_components values in session state if not present
            if "n_values_input" not in st.session_state:
                st.session_state.n_values_input = "3, 5, 7, 10"

            # Input for n_components values
            n_values_input = st.text_input(
                "Enter n_components values to test (comma-separated numbers)",
                value=st.session_state.n_values_input,
                help="Enter numbers separated by commas, e.g. '3, 5, 7, 13'",
                key="n_input",
            )

            # Covariance type selection
            covariance_type = st.selectbox(
                "Covariance Type",
                ["full", "tied", "diag", "spherical"],
                help=(
                    "full: each component has its own general covariance matrix\n"
                    "tied: all components share the same general covariance matrix\n"
                    "diag: each component has its own diagonal covariance matrix\n"
                    "spherical: each component has its own single variance"
                ),
            )

            # Update session state only when input changes
            if n_values_input != st.session_state.n_values_input:
                st.session_state.n_values_input = n_values_input
                st.session_state.gmm_results = (
                    None  # Clear results to force recomputation
                )

            # Parse and validate n_components values
            try:
                n_values = [int(n.strip()) for n in n_values_input.split(",")]
                n_values = [n for n in n_values if n >= 2]  # Filter valid values
                if not n_values:
                    st.error("Please enter valid n_components values (numbers >= 2)")
                elif len(n_values) > 15:
                    st.warning(
                        "Large number of components may increase computation time"
                    )
            except ValueError:
                st.error("Invalid input. Please enter numbers separated by commas.")
                n_values = None

            # Button to run GMM
            if st.button("Run GMM Clustering"):
                if n_values:
                    logger.info(
                        f"Starting GMM clustering analysis with n_components values: {n_values}"
                    )
                    with st.spinner(
                        "Running GMM clustering for multiple n_components values..."
                    ):
                        # Get PCA transformed data for clustering
                        clustering_data = viz_data[
                            [col for col in viz_data.columns if col.startswith("PC")]
                        ]

                        # Run GMM with user-specified values
                        st.session_state.gmm_results = perform_multi_gmm(
                            clustering_data,
                            n_values,
                            covariance_type=covariance_type,
                            feature_flags=st.session_state.feature_flags,
                            is_pca_transformed=True,  # Data is already PCA-transformed
                        )
                        logger.info("GMM clustering completed")

                        # Initialize with first n_components value
                        st.session_state.selected_n = f"n_{n_values[0]}"

            # Show GMM results if available
            if st.session_state.gmm_results is not None:
                # Create tabs for different metrics
                metrics_tab1, metrics_tab2 = st.tabs(
                    ["Clustering Metrics", "Information Criteria"]
                )

                with metrics_tab1:
                    # Get all scores
                    silhouette_scores = {
                        k: results["silhouette"]
                        for k, results in st.session_state.gmm_results.items()
                    }
                    calinski_scores = {
                        k: results["calinski_harabasz"]
                        for k, results in st.session_state.gmm_results.items()
                    }
                    davies_scores = {
                        k: results["davies_bouldin"]
                        for k, results in st.session_state.gmm_results.items()
                    }

                    n_values = [int(n.split("_")[1]) for n in silhouette_scores.keys()]

                    # Create combined metrics plot
                    fig_metrics = go.Figure()

                    # Add traces for each metric
                    fig_metrics.add_trace(
                        go.Scatter(
                            x=n_values,
                            y=list(silhouette_scores.values()),
                            mode="lines+markers",
                            name="Silhouette Score",
                            hovertemplate="n=%{x}<br>Score=%{y:.4f}<extra></extra>",
                        )
                    )

                    # Normalize Calinski-Harabasz scores for plotting
                    ch_scores = list(calinski_scores.values())
                    ch_range = max(ch_scores) - min(ch_scores)
                    ch_norm = [
                        (x - min(ch_scores)) / ch_range if ch_range > 0 else 1.0
                        for x in ch_scores
                    ]
                    fig_metrics.add_trace(
                        go.Scatter(
                            x=n_values,
                            y=ch_norm,
                            mode="lines+markers",
                            name="Calinski-Harabasz (normalized)",
                            hovertemplate="n=%{x}<br>Score=%{y:.4f}<extra></extra>",
                        )
                    )

                    # Davies-Bouldin score (lower is better, so we invert it)
                    db_scores = list(davies_scores.values())
                    db_range = max(db_scores) - min(db_scores)
                    db_norm = [
                        1 - ((x - min(db_scores)) / db_range if db_range > 0 else 0.0)
                        for x in db_scores
                    ]
                    fig_metrics.add_trace(
                        go.Scatter(
                            x=n_values,
                            y=db_norm,
                            mode="lines+markers",
                            name="Davies-Bouldin (inverted, normalized)",
                            hovertemplate="n=%{x}<br>Score=%{y:.4f}<extra></extra>",
                        )
                    )

                    fig_metrics.update_layout(
                        title="Clustering Metrics Comparison",
                        xaxis_title="Number of Components (n)",
                        yaxis_title="Score",
                        showlegend=True,
                    )

                    st.plotly_chart(fig_metrics, use_container_width=True)

                    # Display raw scores in a table
                    metrics_df = pd.DataFrame(
                        {
                            "n_components": n_values,
                            "Silhouette Score": list(silhouette_scores.values()),
                            "Calinski-Harabasz Score": list(calinski_scores.values()),
                            "Davies-Bouldin Score": list(davies_scores.values()),
                        }
                    ).set_index("n_components")

                    st.write("Raw Clustering Metrics:")
                    st.dataframe(metrics_df.round(4), use_container_width=True)

                with metrics_tab2:
                    # Get information criteria scores
                    bic_scores = {
                        k: results["bic"]
                        for k, results in st.session_state.gmm_results.items()
                    }
                    aic_scores = {
                        k: results["aic"]
                        for k, results in st.session_state.gmm_results.items()
                    }

                    # Create information criteria plot
                    fig_ic = go.Figure()

                    # Add traces for BIC and AIC
                    fig_ic.add_trace(
                        go.Scatter(
                            x=n_values,
                            y=list(bic_scores.values()),
                            mode="lines+markers",
                            name="BIC",
                            hovertemplate="n=%{x}<br>BIC=%{y:.0f}<extra></extra>",
                        )
                    )

                    fig_ic.add_trace(
                        go.Scatter(
                            x=n_values,
                            y=list(aic_scores.values()),
                            mode="lines+markers",
                            name="AIC",
                            hovertemplate="n=%{x}<br>AIC=%{y:.0f}<extra></extra>",
                        )
                    )

                    fig_ic.update_layout(
                        title="Information Criteria",
                        xaxis_title="Number of Components (n)",
                        yaxis_title="Score (lower is better)",
                        showlegend=True,
                    )

                    st.plotly_chart(fig_ic, use_container_width=True)

                    # Display raw scores in a table
                    ic_df = pd.DataFrame(
                        {
                            "n_components": n_values,
                            "BIC": list(bic_scores.values()),
                            "AIC": list(aic_scores.values()),
                        }
                    ).set_index("n_components")

                    st.write("Raw Information Criteria Scores:")
                    st.dataframe(ic_df.round(4), use_container_width=True)

                # Dropdown to select n with state management
                n_options = list(st.session_state.gmm_results.keys())

                def on_n_change():
                    st.session_state.selected_n = st.session_state.n_select
                    logger.info(
                        f"Selected GMM n_components changed to {st.session_state.selected_n}"
                    )

                st.selectbox(
                    "Select number of components (n)",
                    n_options,
                    index=n_options.index(st.session_state.selected_n),
                    format_func=lambda x: f"n={x.split('_')[1]} (silhouette={silhouette_scores[x]:.4f})",
                    key="n_select",
                    on_change=on_n_change,
                )

                # Update scatter plot with cluster colors
                st.subheader("Cluster Visualization")

                # Get PCA components for visualization
                col1, col2 = st.columns(2)
                with col1:
                    # Get available components dynamically
                    components = [
                        col for col in viz_data.columns if col.startswith("PC")
                    ]

                    # Ensure valid default selections
                    if st.session_state.gmm_x_component not in components:
                        st.session_state.gmm_x_component = (
                            components[0] if components else None
                        )

                    if st.session_state.gmm_y_component not in components:
                        st.session_state.gmm_y_component = (
                            components[1]
                            if len(components) > 1
                            else components[0]
                            if components
                            else None
                        )

                    def on_x_component_change():
                        st.session_state.gmm_x_component = st.session_state.gmm_x_select

                    st.selectbox(
                        "X-axis Component",
                        components,
                        index=(
                            components.index(st.session_state.gmm_x_component)
                            if st.session_state.gmm_x_component in components
                            else 0
                        ),
                        key="gmm_x_select",
                        on_change=on_x_component_change,
                    )
                with col2:

                    def on_y_component_change():
                        st.session_state.gmm_y_component = st.session_state.gmm_y_select

                    st.selectbox(
                        "Y-axis Component",
                        components,
                        index=(
                            components.index(st.session_state.gmm_y_component)
                            if st.session_state.gmm_y_component in components
                            else (1 if len(components) > 1 else 0)
                        ),
                        key="gmm_y_select",
                        on_change=on_y_component_change,
                    )

                # Get cluster labels and responsibilities
                cluster_labels = st.session_state.gmm_results[
                    st.session_state.selected_n
                ]["labels"]
                responsibilities = st.session_state.gmm_results[
                    st.session_state.selected_n
                ]["responsibilities"]

                # Create and display clustering scatter plot
                fig_scatter_gmm = create_pca_scatter(
                    viz_data,
                    st.session_state.gmm_x_component,
                    st.session_state.gmm_y_component,
                    cluster_labels=cluster_labels,
                )

                # Add confidence ellipses
                means = st.session_state.gmm_results[st.session_state.selected_n][
                    "means"
                ]
                covariances = st.session_state.gmm_results[st.session_state.selected_n][
                    "covariances"
                ]
                weights = st.session_state.gmm_results[st.session_state.selected_n][
                    "weights"
                ]

                n_components = len(means)
                color_map = get_cluster_colors(n_components)

                # Determine which components to plot based on the selected cluster
                selected_cluster_idx = (
                    int(st.session_state.selected_cluster_explore.split()[-1]) - 1
                )

                # Ensure the selected cluster index is within the range of available components
                n_components = len(means)
                selected_cluster_idx = min(selected_cluster_idx, n_components - 1)
                components_to_plot = [selected_cluster_idx]

                for i in components_to_plot:
                    if covariance_type == "full":
                        cov = (
                            covariances[i] if i < len(covariances) else covariances[-1]
                        )
                    elif covariance_type == "tied":
                        cov = covariances
                    elif covariance_type == "diag":
                        cov = np.diag(
                            covariances[i] if i < len(covariances) else covariances[-1]
                        )
                    else:  # spherical
                        cov = np.eye(2) * (
                            covariances[i] if i < len(covariances) else covariances[-1]
                        )

                    v, w = np.linalg.eigh(cov[:2, :2])
                    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                    u = w[0] / np.linalg.norm(w[0])

                    angle = np.arctan(u[1] / u[0])
                    angle = 180.0 * angle / np.pi

                    # Plot ellipse
                    for nstd in [1, 2, 3]:
                        fig_scatter_gmm.add_shape(
                            type="circle",
                            xref="x",
                            yref="y",
                            x0=means[i, 0] - v[0] * nstd,
                            y0=means[i, 1] - v[1] * nstd,
                            x1=means[i, 0] + v[0] * nstd,
                            y1=means[i, 1] + v[1] * nstd,
                            line=dict(
                                color=color_map[str(i)],
                                width=1,
                                dash="dash",
                            ),
                            opacity=0.2,
                        )

                st.plotly_chart(fig_scatter_gmm, use_container_width=True)

                # Show cluster analysis
                (
                    cluster_analysis_tab1,
                    cluster_analysis_tab2,
                    cluster_analysis_tab3,
                    cluster_analysis_tab4,
                ) = st.tabs(
                    [
                        "Cluster Sizes & Weights",
                        "Feature Importance",
                        "Assignment Probabilities",
                        "Silhouette Analysis",
                    ]
                )

                with cluster_analysis_tab1:
                    # Show cluster sizes and mixing weights
                    cluster_sizes = (
                        pd.Series([f"Component {i + 1}" for i in cluster_labels])
                        .value_counts()
                        .sort_index()
                    )
                    n_components = len(cluster_sizes)
                    color_map = get_cluster_colors(n_components)

                    # Create DataFrame with both sizes and weights
                    size_weight_df = pd.DataFrame(
                        {
                            "Size": cluster_sizes,
                            "Weight": weights,
                        }
                    )
                    size_weight_df["Proportion"] = size_weight_df["Size"] / len(
                        cluster_labels
                    )

                    # Create bar chart
                    fig_sizes = go.Figure()

                    # Add bars for actual proportions
                    fig_sizes.add_trace(
                        go.Bar(
                            x=size_weight_df.index,
                            y=size_weight_df["Proportion"],
                            name="Actual Proportion",
                            marker_color=[
                                color_map[str(i)] for i in range(n_components)
                            ],
                        )
                    )

                    # Add markers for mixing weights
                    fig_sizes.add_trace(
                        go.Scatter(
                            x=size_weight_df.index,
                            y=size_weight_df["Weight"],
                            mode="markers",
                            name="Mixing Weight",
                            marker=dict(
                                symbol="diamond",
                                size=10,
                                color="white",
                                line=dict(color="black", width=2),
                            ),
                        )
                    )

                    fig_sizes.update_layout(
                        title="Component Sizes and Mixing Weights",
                        xaxis_title="Component",
                        yaxis_title="Proportion",
                        showlegend=True,
                    )

                    st.plotly_chart(fig_sizes, use_container_width=True)

                    # Display raw values in a table
                    st.write("Component Statistics:")
                    st.dataframe(size_weight_df.round(4), use_container_width=True)

                with cluster_analysis_tab2:
                    # Show feature importance per component
                    feature_importance = st.session_state.gmm_results[
                        st.session_state.selected_n
                    ]["feature_importance"]

                    # Create feature importance visualization with state management
                    def on_importance_component_change():
                        st.session_state.selected_component_importance = (
                            int(
                                st.session_state.importance_component_select.split()[-1]
                            )
                            - 1
                        )

                    component_names = [
                        f"Component {i + 1}" for i in range(n_components)
                    ]
                    selected_component_name = st.selectbox(
                        "Select Component for Feature Analysis",
                        component_names,
                        key="importance_component_select",
                        index=0,
                        on_change=on_importance_component_change,
                    )
                    selected_component = int(selected_component_name.split()[-1]) - 1

                    # Get feature importance for selected component
                    component_importance = feature_importance[selected_component]
                    importance_df = pd.DataFrame(
                        {
                            "Feature": component_importance.index,
                            "Importance": component_importance.values,
                        }
                    ).sort_values("Importance", ascending=True)

                    # Show top 15 most important features
                    color_map = get_cluster_colors(n_components)
                    component_color = color_map[str(selected_component)]

                    fig_importance = go.Figure()
                    fig_importance.add_trace(
                        go.Bar(
                            y=importance_df["Feature"].tail(15),
                            x=importance_df["Importance"].tail(15),
                            orientation="h",
                            marker_color=component_color,
                        )
                    )

                    fig_importance.update_layout(
                        title=f"Top 15 Most Important Features for {selected_component_name}",
                        xaxis_title="Feature Importance (Z-score)",
                        yaxis_title="Feature",
                        height=600,
                    )

                    st.plotly_chart(fig_importance, use_container_width=True)

                with cluster_analysis_tab3:
                    # Show assignment probabilities for each sample
                    st.write("### Component Assignment Analysis")

                    # Create DataFrame with game info and probabilities
                    prob_data = {
                        "game_id": st.session_state.data["game_id"],
                        "name": st.session_state.data["name"],
                        "Primary Component": [
                            f"Component {i + 1}" for i in cluster_labels
                        ],
                    }

                    # Debug: Print out responsibilities details
                    st.write("Responsibilities Array Details:")
                    st.write(f"Shape: {responsibilities.shape}")
                    st.write(f"Min value: {responsibilities.min()}")
                    st.write(f"Max value: {responsibilities.max()}")
                    st.write(f"Mean value: {responsibilities.mean()}")
                    st.write("Sum of probabilities per sample:")
                    sample_sums = responsibilities.sum(axis=1)
                    st.write(sample_sums)
                    st.write("Samples with sum != 1:")
                    st.write(np.where(np.abs(sample_sums - 1) > 1e-10)[0])

                    # Add probability columns
                    for i in range(n_components):
                        prob_data[f"P(Component {i + 1})"] = responsibilities[:, i]

                    prob_df = pd.DataFrame(prob_data)

                    # Calculate entropy of assignment probabilities for each game
                    def calculate_entropy(probs):
                        # Avoid log(0) by adding small epsilon
                        probs = probs + 1e-10
                        return -np.sum(probs * np.log2(probs))

                    # Add entropy and max probability
                    component_probs = responsibilities
                    prob_df["Assignment Entropy"] = [
                        calculate_entropy(probs) for probs in component_probs
                    ]
                    prob_df["Max Probability"] = component_probs.max(axis=1)

                    # Create tabs for different views
                    prob_tab1, prob_tab2 = st.tabs(
                        ["Probability Distribution", "Game Analysis"]
                    )

                    with prob_tab1:
                        # Show distribution of probabilities
                        fig_prob = go.Figure()

                        # Add violin plots for each component
                        for i in range(n_components):
                            fig_prob.add_trace(
                                go.Violin(
                                    y=responsibilities[:, i],
                                    name=f"Component {i + 1}",
                                    box_visible=True,
                                    meanline_visible=True,
                                    points="outliers",
                                    line_color=color_map[str(i)],
                                )
                            )

                        fig_prob.update_layout(
                            title="Distribution of Assignment Probabilities by Component",
                            yaxis_title="Probability",
                            showlegend=True,
                            violinmode="overlay",
                            height=500,
                        )

                        st.plotly_chart(fig_prob, use_container_width=True)

                        # Show entropy distribution
                        fig_entropy = go.Figure()
                        fig_entropy.add_trace(
                            go.Histogram(
                                x=prob_df["Assignment Entropy"],
                                nbinsx=50,
                                name="Entropy Distribution",
                            )
                        )

                        fig_entropy.update_layout(
                            title="Distribution of Assignment Entropy",
                            xaxis_title="Entropy (bits)",
                            yaxis_title="Count",
                            height=300,
                        )

                        st.plotly_chart(fig_entropy, use_container_width=True)

                        st.write(
                            """
                        ### Interpreting the Results
                        - **Assignment Probability**: Higher values indicate stronger component membership
                        - **Entropy**: Lower values indicate more certain assignments
                            - 0 bits: Game belongs entirely to one component
                            - Higher bits: Game has mixed membership across components
                        """
                        )

                    with prob_tab2:
                        # Controls for filtering and analysis
                        col1, col2 = st.columns(2)

                        with col1:
                            min_prob = st.slider(
                                "Minimum Probability",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.5,
                                step=0.05,
                            )

                        with col2:
                            max_entropy = st.slider(
                                "Maximum Entropy (bits)",
                                min_value=0.0,
                                max_value=float(prob_df["Assignment Entropy"].max()),
                                value=float(prob_df["Assignment Entropy"].median()),
                                step=0.1,
                            )

                        # Filter based on both criteria
                        filtered_df = prob_df[
                            (prob_df["Max Probability"] >= min_prob)
                            & (prob_df["Assignment Entropy"] <= max_entropy)
                        ].sort_values("Max Probability", ascending=False)

                        st.write(
                            f"""
                        Showing games with:
                        - Probability ≥ {min_prob:.2f} for at least one component
                        - Assignment entropy ≤ {max_entropy:.2f} bits
                        """
                        )

                        # Display filtered results
                        st.dataframe(
                            filtered_df[
                                [
                                    "name",
                                    "game_id",
                                    "Primary Component",
                                    "Max Probability",
                                    "Assignment Entropy",
                                ]
                                + [f"P(Component {i + 1})" for i in range(n_components)]
                            ],
                            use_container_width=True,
                            height=400,
                        )

                with cluster_analysis_tab4:
                    # Get silhouette values for each sample
                    silhouette_values = st.session_state.gmm_results[
                        st.session_state.selected_n
                    ]["sample_silhouette_values"]

                    # Create a DataFrame with game info and silhouette scores
                    silhouette_df = pd.DataFrame(
                        {
                            "game_id": st.session_state.data["game_id"],
                            "name": st.session_state.data["name"],
                            "component": [f"Component {i + 1}" for i in cluster_labels],
                            "silhouette_score": silhouette_values,
                            "max_responsibility": responsibilities.max(axis=1),
                        }
                    )

                    # Sort by silhouette score within each component
                    silhouette_df = silhouette_df.sort_values(
                        ["component", "silhouette_score"], ascending=[True, False]
                    )

                    # Create box plot with individual points
                    n_components = int(st.session_state.selected_n.split("_")[1])
                    color_map = get_cluster_colors(n_components)

                    fig_silhouette = go.Figure()

                    # Add box plots for each component
                    for i in range(n_components):
                        component_name = f"Component {i + 1}"
                        component_data = silhouette_df[
                            silhouette_df["component"] == component_name
                        ]

                        # Add box plot
                        fig_silhouette.add_trace(
                            go.Box(
                                y=component_data["silhouette_score"],
                                name=component_name,
                                boxpoints="all",
                                jitter=0.3,
                                pointpos=-1.8,
                                marker=dict(
                                    color=color_map[str(i)],
                                    size=4,
                                    opacity=0.7,
                                ),
                                hovertemplate=(
                                    "Game: %{customdata[0]}<br>"
                                    + "Score: %{y:.4f}<br>"
                                    + "Responsibility: %{customdata[1]:.4f}<br>"
                                    + "<extra></extra>"
                                ),
                                customdata=component_data[
                                    ["name", "max_responsibility"]
                                ].values,
                            )
                        )

                    # Add horizontal line for average silhouette score
                    avg_score = silhouette_scores[st.session_state.selected_n]
                    fig_silhouette.add_shape(
                        type="line",
                        x0=0,
                        x1=1,
                        y0=avg_score,
                        y1=avg_score,
                        line=dict(
                            color="red",
                            dash="dash",
                        ),
                        xref="paper",
                        yref="y",
                    )
                    fig_silhouette.add_annotation(
                        text=f"Average silhouette score: {avg_score:.4f}",
                        x=1,
                        y=avg_score,
                        xref="paper",
                        yref="y",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        xshift=10,
                    )

                    fig_silhouette.update_layout(
                        title="Distribution of Silhouette Scores by Component",
                        yaxis_title="Silhouette Score (-1 to 1, higher is better)",
                        height=600,
                        showlegend=True,
                        hovermode="closest",
                    )

                    st.plotly_chart(fig_silhouette, use_container_width=True)

                    # Add table view of silhouette scores
                    st.subheader("Individual Game Silhouette Scores")

                    # Allow filtering by component
                    selected_component = st.selectbox(
                        "Filter by Component",
                        ["All Components"]
                        + [f"Component {i + 1}" for i in range(n_components)],
                    )

                    # Filter data based on selection
                    if selected_component == "All Components":
                        filtered_df = silhouette_df
                    else:
                        filtered_df = silhouette_df[
                            silhouette_df["component"] == selected_component
                        ]

                    # Display the data
                    st.dataframe(
                        filtered_df[
                            [
                                "name",
                                "game_id",
                                "component",
                                "silhouette_score",
                                "max_responsibility",
                            ]
                        ].sort_values("silhouette_score", ascending=False),
                        use_container_width=True,
                        height=400,
                    )

        else:
            st.warning(
                "PCA results are not available. Please compute PCA first in the PCA tab."
            )

    # Find a Neighbor Tab
    with tab_neighbor:
        st.header("Find a Neighbor")

        # Check if K-Means clustering has been performed
        if st.session_state.clustering_results is None:
            st.warning("Please run K-Means clustering first in the K-Means tab.")
        else:
            # Get all available K-Means clustering results
            available_k_values = list(st.session_state.clustering_results.keys())

            # Cluster count selection
            col1, col2 = st.columns(2)

            with col1:
                # Dropdown to select number of clusters
                selected_k = st.selectbox(
                    "Select Number of Clusters",
                    available_k_values,
                    format_func=lambda x: f"k={x.split('_')[1]}",
                    key="neighbor_k_select",
                )

            with col2:
                # Distance metric selection
                distance_metric = st.selectbox(
                    "Distance Metric", ["Euclidean", "Manhattan", "Cosine"], index=0
                )

            # Get the selected clustering results
            cluster_labels = st.session_state.clustering_results[selected_k]["labels"]

            # Game selection
            st.subheader("Select a Game")

            # Prepare game selection data
            game_selection_data = st.session_state.data[
                ["game_id", "name", "year_published"]
            ].copy()
            game_selection_data["cluster"] = cluster_labels

            # Game search with game_id and year_published in parentheses
            game_selection_data["display_name"] = game_selection_data.apply(
                lambda row: f"{row['name']} (ID: {row['game_id']}, Year: {row['year_published']})",
                axis=1,
            )

            selected_game_display = st.selectbox(
                "Search for a game",
                options=game_selection_data["display_name"].tolist(),
                key="game_neighbor_search",
            )

            # Find the original game name
            selected_game = game_selection_data[
                game_selection_data["display_name"] == selected_game_display
            ]["name"].iloc[0]

            if selected_game:
                # Find the selected game's details
                selected_game_row = game_selection_data[
                    game_selection_data["name"] == selected_game
                ].iloc[0]
                selected_game_id = selected_game_row["game_id"]
                selected_game_cluster = selected_game_row["cluster"]

                # Import distance functions
                from scipy.spatial.distance import euclidean, cityblock, cosine

                # Get PCA-transformed data for distance calculation
                pca_components = [
                    col for col in viz_data.columns if col.startswith("PC")
                ]

                # Find the index of the selected game
                selected_game_index = st.session_state.data[
                    st.session_state.data["game_id"] == selected_game_id
                ].index[0]

                # Get the PCA coordinates of the selected game
                selected_game_coords = viz_data.loc[
                    selected_game_index, pca_components
                ].values

                # Calculate distances for all games
                distances = []
                for idx, row in enumerate(viz_data[pca_components].values):
                    # Skip the selected game itself
                    if idx == selected_game_index:
                        continue

                    # Calculate distance based on selected metric
                    if distance_metric == "Euclidean":
                        dist = euclidean(selected_game_coords, row)
                    elif distance_metric == "Manhattan":
                        dist = cityblock(selected_game_coords, row)
                    else:  # Cosine
                        dist = cosine(selected_game_coords, row)

                    distances.append((idx, dist))

                # Sort distances
                distances.sort(key=lambda x: x[1])

                # Prepend the selected game with zero distance
                distances.insert(0, (selected_game_index, 0.0))

                # Filter distances to only include games in the same cluster
                cluster_distances = [
                    (idx, dist)
                    for idx, dist in distances
                    if cluster_labels[idx] == selected_game_cluster
                ]

                # Prepare results DataFrame
                results_df = st.session_state.data.loc[
                    [d[0] for d in cluster_distances]
                ].copy()
                results_df["distance"] = [d[1] for d in cluster_distances]
                results_df["cluster"] = cluster_labels[
                    [d[0] for d in cluster_distances]
                ]

                # Get the color map for clusters
                n_clusters = int(selected_k.split("_")[1])
                color_map = get_cluster_colors(n_clusters)

                # Prepare default columns from the main games table
                default_columns = [
                    "game_id",
                    "name",
                    "year_published",
                    "geek_rating",
                    "users_rated",
                    "average_rating",
                    "predicted_complexity",
                ]

                # Filter columns that exist in the dataframe
                available_columns = [
                    col for col in default_columns if col in results_df.columns
                ]

                # Prepare results table with new column order
                results_table = results_df.copy()
                results_table.insert(1, "Distance", results_df["distance"])

                # Reorder columns to put Cluster and Distance first, then add default columns
                column_order = ["cluster", "Distance"] + available_columns
                results_table = results_table[column_order]

                # Increase Pandas Styler cell limit
                import pandas as pd

                pd.set_option("styler.render.max_elements", 500000)

                # Limit the number of rows displayed
                max_display_rows = 150
                if len(results_table) > max_display_rows:
                    st.info(
                        f"Showing first {max_display_rows} of {len(results_table)} games in the same cluster."
                    )
                    results_table = results_table.head(max_display_rows)

                # Create a styled DataFrame with color gradient for distance
                def color_distance(val):
                    """Create color gradient based on distance."""
                    # Normalize distance to 0-1 range
                    max_distance = results_table["Distance"].max()
                    normalized = 1 - (val / max_distance)

                    # Create a color gradient from light blue (far) to deep blue (near)
                    r = int(255 * (1 - normalized))
                    g = int(255 * (1 - normalized))
                    b = int(255 * (1 - normalized) + 255 * normalized)

                    return f"background-color: rgb({r},{g},{b})"

                # Apply styling using .map instead of deprecated .applymap
                styled_results = results_table.style.map(
                    color_distance, subset=["Distance"]
                )

                # Display results
                st.subheader("Nearest Neighbors in Cluster")
                st.dataframe(styled_results, use_container_width=True)

                # Create PCA scatter plot highlighting the selected game's cluster
                st.subheader("Neighbor Cluster Visualization")

                # Filter data to only include the selected game's cluster
                cluster_mask = cluster_labels == selected_game_cluster
                cluster_viz_data = viz_data[cluster_mask].copy()
                cluster_viz_data["Selected Game"] = False

                # Mark the selected game
                cluster_viz_data.loc[selected_game_index, "Selected Game"] = True

                # Create scatter plot highlighting the selected cluster among all observations
                fig_neighbor_scatter = px.scatter(
                    viz_data,
                    x=st.session_state.kmeans_x_component,
                    y=st.session_state.kmeans_y_component,
                    color_discrete_map={
                        "Selected Cluster": "lightblue",
                        "Nearest Neighbors": "blue",
                        "Other Clusters": "lightgrey",
                    },
                    title=f"Cluster {selected_game_cluster + 1} Visualization",
                    hover_data=["game_id", "name"],
                )

                # Add color column to distinguish clusters and neighbors
                viz_data["Cluster_Highlight"] = np.where(
                    cluster_labels == selected_game_cluster,
                    "Selected Cluster",
                    "Other Clusters",
                )

                # Identify nearest neighbors
                neighbor_indices = [
                    d[0] for d in cluster_distances[1:11]
                ]  # Top 10 neighbors
                viz_data.loc[neighbor_indices, "Cluster_Highlight"] = (
                    "Nearest Neighbors"
                )

                # Update scatter plot with cluster and neighbor highlighting
                fig_neighbor_scatter = px.scatter(
                    viz_data,
                    x=st.session_state.kmeans_x_component,
                    y=st.session_state.kmeans_y_component,
                    color="Cluster_Highlight",
                    color_discrete_map={
                        "Selected Cluster": "blue",
                        "Nearest Neighbors": "skyblue",
                        "Other Clusters": "black",
                    },
                    hover_data=["game_id", "name"],
                )

                # Customize the plot
                fig_neighbor_scatter.update_traces(
                    marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey"))
                )

                # Highlight the selected game with a star marker
                selected_game_coords = viz_data.loc[selected_game_index]
                fig_neighbor_scatter.add_trace(
                    go.Scatter(
                        x=[selected_game_coords[st.session_state.kmeans_x_component]],
                        y=[selected_game_coords[st.session_state.kmeans_y_component]],
                        mode="markers",
                        marker=dict(
                            color="gold",  # Bright, attention-grabbing color
                            size=40,  # Larger size to ensure visibility
                            symbol="star",
                            line=dict(color="black", width=3),  # Thick black border
                            opacity=1.0,  # Full opacity
                        ),
                        name="Selected Game",
                        hovertext=selected_game,
                        hoverinfo="text",
                        showlegend=False,  # Remove from legend to reduce clutter
                    )
                )

                fig_neighbor_scatter.update_layout(
                    height=600,
                    xaxis=dict(
                        zeroline=True,
                        zerolinewidth=2,
                        showgrid=False,
                    ),
                    yaxis=dict(
                        zeroline=True,
                        zerolinewidth=2,
                        showgrid=False,
                    ),
                )

                st.plotly_chart(fig_neighbor_scatter, use_container_width=True)


if __name__ == "__main__":
    main()
