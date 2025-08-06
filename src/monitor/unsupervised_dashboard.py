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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.data.loader import BGGDataLoader
from src.data.config import load_config
from src.features.preprocessor import create_bgg_preprocessor
from src.features.unsupervised import perform_pca, perform_kmeans
from src.utils.logging import setup_logging

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
            # st.info(f"Loading data from cache: {cache_file}")
            cached_data = pd.read_parquet(cache_file)
            # st.info(f"Cached data loaded successfully. Shape: {cached_data.shape}")
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
        cluster_std = data[cluster_mask].std()
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
    if "selected_cluster_explore" not in st.session_state:
        st.session_state.selected_cluster_explore = "0"
    if "selected_cluster_importance" not in st.session_state:
        st.session_state.selected_cluster_importance = 0
    if "kmeans_x_component" not in st.session_state:
        st.session_state.kmeans_x_component = "PC1"
    if "kmeans_y_component" not in st.session_state:
        st.session_state.kmeans_y_component = "PC2"
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
        "PCA Variance Explained", min_value=0.0, max_value=0.9, value=0.5, step=0.1
    )
    # n_components = st.sidebar.slider(
    #     "Number of Components", min_value=2, max_value=50, value=50
    # )

    # Tabs for different analyses
    tab_data, tab_pca, tab_kmeans = st.tabs(["Data", "PCA", "K-Means"])

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
        st.title("Data")
        # Allow user to control sample size
        sample_size = st.slider(
            "Sample Size", min_value=50, max_value=5000, step=50, value=1000
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
            # Get available components
            available_components = [
                f"PC{i+1}" for i in range(viz_data.shape[1] - 3)
            ]  # -3 for the additional columns

            # Component selection for scatter plot
            col1, col2 = st.columns(2)
            with col1:
                x_component = st.selectbox(
                    "X-axis Component", available_components, index=0
                )
            with col2:
                y_component = st.selectbox(
                    "Y-axis Component", available_components, index=1
                )

            # Color options
            color_options = ["None", "Predicted Complexity", "Year Published"]

            def create_pca_scatter(
                data, x_comp, y_comp, color=None, cluster_labels=None
            ):
                """Create scatter plot for PCA visualization with optional clustering."""
                if cluster_labels is not None:
                    n_clusters = len(np.unique(cluster_labels))
                    color_map = get_cluster_colors(n_clusters)
                    # Convert cluster labels to "Cluster X" format
                    cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]
                    cluster_labels_str = np.array(
                        [f"Cluster {i+1}" for i in cluster_labels]
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
            # Get loadings for selected components
            loadings = pca_results["loadings"]

            # Calculate absolute loadings for selected components
            loadings_subset = loadings.copy()
            loadings_subset["abs_loading"] = np.sqrt(
                loadings_subset[x_component] ** 2 + loadings_subset[y_component] ** 2
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
            loadings_component = st.selectbox(
                "Component", available_components, index=0
            )

            # Create separate charts for each component
            for component in [loadings_component]:
                # Get absolute loadings for this component
                component_loadings = loadings.copy()
                component_loadings["AbsLoading"] = np.abs(component_loadings[component])

                # Get top 25 features by absolute loading
                top_features = component_loadings.nlargest(25, "AbsLoading")

                # Sort features by loading value for better visualization
                top_features = top_features.sort_values(component)

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
                    ch_norm = [
                        (x - min(ch_scores)) / (max(ch_scores) - min(ch_scores))
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
                    db_norm = [
                        1 - (x - min(db_scores)) / (max(db_scores) - min(db_scores))
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

                    # Display raw scores in a table
                    metrics_df = pd.DataFrame(
                        {
                            "k": k_values,
                            "Silhouette Score": list(silhouette_scores.values()),
                            "Calinski-Harabasz Score": list(calinski_scores.values()),
                            "Davies-Bouldin Score": list(davies_scores.values()),
                        }
                    ).set_index("k")

                    st.write("Raw Clustering Metrics:")
                    st.dataframe(metrics_df.round(4), use_container_width=True)

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
                    components = [f"PC{i+1}" for i in range(viz_data.shape[1] - 3)]

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
                        pd.Series([f"Cluster {i+1}" for i in cluster_labels])
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
                            "color": [f"Cluster {i+1}" for i in range(n_clusters)]
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

                    cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]
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
                            "cluster": [f"Cluster {i+1}" for i in cluster_labels],
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
                        cluster_name = f"Cluster {i+1}"
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
                        + [f"Cluster {i+1}" for i in range(n_clusters)],
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
                                "Cluster": f"Cluster {i+1}",
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
                    st.session_state.selected_cluster_explore = "0"

                # Select cluster to explore with state management
                def on_explore_cluster_change():
                    st.session_state.selected_cluster_explore = (
                        st.session_state.cluster_explorer_select
                    )

                cluster_names = [f"Cluster {i+1}" for i in range(n_clusters)]
                selected_cluster_name = st.selectbox(
                    "Select Cluster to Explore",
                    cluster_names,
                    key="cluster_explorer_select",
                    index=int(st.session_state.selected_cluster_explore),
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
                    cluster_idx = int(st.session_state.selected_cluster_explore)
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
        #     # Define hover columns for clustering visualization
        #     hover_columns = ["game_id", "name", "year_published"]

        #     st.sidebar.header("Clustering")
        #     clustering_method = st.sidebar.selectbox(
        #         "Clustering Method", ["KMeans", "DBSCAN"], index=0
        #     )

        #     if clustering_method == "KMeans":
        #         n_clusters = st.sidebar.slider(
        #             "Number of Clusters", min_value=2, max_value=10, value=5
        #         )
        #         eps = 0.5
        #         min_samples = 5
        #     else:
        #         eps = st.sidebar.slider(
        #             "EPS", min_value=0.1, max_value=2.0, value=0.5, step=0.1
        #         )
        #         min_samples = st.sidebar.slider(
        #             "Min Samples", min_value=2, max_value=20, value=5
        #         )

        #     # Clustering
        #     with st.spinner(f"Performing {clustering_method} clustering..."):
        #         labels = clustering(
        #             st.session_state.X,
        #             method=clustering_method,
        #             n_clusters=n_clusters if clustering_method == "KMeans" else None,
        #             eps=eps if clustering_method == "DBSCAN" else None,
        #             min_samples=min_samples if clustering_method == "DBSCAN" else None,
        #         )

        #     # Add cluster labels to visualization data
        #     viz_data["Cluster"] = labels

        #     # Color by options for clustering
        #     color_by_cluster = st.selectbox(
        #         "Color By", ["Cluster", "Predicted Complexity", "Year Published"], index=0
        #     )

        #     # Clustering Visualization
        #     if n_components == 2:
        #         # 2D Plot
        #         if color_by_cluster == "Cluster":
        #             # Use discrete color for clusters
        #             fig_clustering = px.scatter(
        #                 viz_data,
        #                 x="Component 1",
        #                 y="Component 2",
        #                 color="Cluster",
        #                 hover_data=hover_columns + ["Cluster"],
        #                 title=f"{clustering_method} Clustering Visualization",
        #                 color_discrete_sequence=px.colors.qualitative.Plotly,  # Use Plotly's qualitative color palette
        #             )
        #         else:
        #             # Use continuous color for other options
        #             fig_clustering = px.scatter(
        #                 viz_data,
        #                 x="Component 1",
        #                 y="Component 2",
        #                 color=color_by_cluster,
        #                 color_continuous_scale="viridis",
        #                 hover_data=hover_columns + ["Cluster"],
        #                 title=f"{clustering_method} Clustering Visualization",
        #             )
        #     else:
        #         # 3D Plot
        #         if color_by_cluster == "Cluster":
        #             # Use discrete color for clusters
        #             fig_clustering = px.scatter_3d(
        #                 viz_data,
        #                 x="Component 1",
        #                 y="Component 2",
        #                 z="Component 3",
        #                 color="Cluster",
        #                 hover_data=hover_columns + ["Cluster"],
        #                 title=f"{clustering_method} 3D Clustering Visualization",
        #                 color_discrete_sequence=px.colors.qualitative.Plotly,  # Use Plotly's qualitative color palette
        #             )
        #         else:
        #             # Use continuous color for other options
        #             fig_clustering = px.scatter_3d(
        #                 viz_data,
        #                 x="Component 1",
        #                 y="Component 2",
        #                 z="Component 3",
        #                 color=color_by_cluster,
        #                 color_continuous_scale="viridis",
        #                 hover_data=hover_columns + ["Cluster"],
        #                 title=f"{clustering_method} 3D Clustering Visualization",
        #             )

        #     st.plotly_chart(fig_clustering, use_container_width=True)

        # # Clustering metrics
        # st.header("Clustering Metrics")
        # try:
        #     silhouette = silhouette_score(st.session_state.X, labels)
        #     st.metric("Silhouette Score", f"{silhouette:.4f}")
        # except Exception as e:
        #     st.warning(f"Could not compute silhouette score: {e}")


if __name__ == "__main__":
    main()
