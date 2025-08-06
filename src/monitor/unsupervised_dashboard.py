import os
import sys
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


def clustering(X, method="KMeans", n_clusters=5, eps=0.5, min_samples=5):
    # """Perform clustering."""
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    if method == "KMeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "DBSCAN":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    labels = clusterer.fit_predict(X)
    return labels


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
    tab_data, tab_pca, tab2 = st.tabs(["Data", "PCA", "Clustering"])

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
            color_by = st.selectbox("Color By", color_options, index=0)

            # Scatter Plot
            if color_by == "None":
                fig_scatter = px.scatter(
                    viz_data,
                    opacity=0.4,
                    x=x_component,
                    y=y_component,
                    hover_data=["game_id", "name", "year_published"],
                    title=f"PCA Scatter Plot: {x_component} vs {y_component}",
                )
            else:
                fig_scatter = px.scatter(
                    viz_data,
                    x=x_component,
                    y=y_component,
                    opacity=0.4,
                    color=color_by,
                    color_continuous_scale="viridis",
                    hover_data=["game_id", "name", "year_published"],
                    title=f"PCA Scatter Plot: {x_component} vs {y_component}",
                )

            fig_scatter.update_layout(
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

            st.plotly_chart(fig_scatter, use_container_width=True)
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

            # Create separate charts for each component
            for component in [x_component]:
                # Get absolute loadings for this component
                component_loadings = loadings.copy()
                component_loadings["AbsLoading"] = np.abs(component_loadings[component])

                # Get top 25 features by absolute loading
                top_features = component_loadings.nlargest(25, "AbsLoading")

                # Sort features by loading value for better visualization
                top_features = top_features.sort_values(component)

                # Create horizontal bar chart
                fig_loadings = go.Figure(
                    data=[
                        go.Bar(
                            y=top_features["feature"],  # Features on y-axis
                            x=top_features[component],  # Loadings on x-axis
                            orientation="h",  # Horizontal bars
                            marker_color="blue" if component == x_component else "red",
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
                    marker_color="red",
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

        # # Clustering Tab
        # with tab2:
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
