"""
Streamlit page for exploring embedding experiment results locally.

This page allows you to:
- Select and compare embedding experiments
- View scree plots and explained variance
- Explore component loadings
- Visualize embeddings in 2D
- Search for games and find similar games
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, project_root)

import json
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics.pairwise import cosine_similarity

from src.streamlit.components.footer import render_footer

# Page config
st.set_page_config(page_title="Embeddings | BGG Models Dashboard", layout="wide")
st.title("Embedding Explorer")

# Constants
EXPERIMENTS_DIR = Path(project_root) / "models" / "experiments" / "embeddings"


@st.cache_data
def list_experiments() -> list[dict]:
    """List all available embedding experiments."""
    experiments = []
    if not EXPERIMENTS_DIR.exists():
        return experiments

    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if exp_dir.is_dir():
            # Find all versions
            for version_dir in sorted(exp_dir.iterdir(), reverse=True):
                if version_dir.is_dir() and version_dir.name.startswith("v"):
                    metadata_path = version_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        experiments.append({
                            "name": f"{exp_dir.name}/{version_dir.name}",
                            "path": version_dir,
                            "metadata": metadata,
                        })
    return experiments


@st.cache_data
def load_artifacts(exp_path: str) -> dict:
    """Load experiment artifacts."""
    exp_path = Path(exp_path)
    artifacts_path = exp_path / "artifacts.json"
    if artifacts_path.exists():
        with open(artifacts_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_embeddings(exp_path: str, dataset: str = "train") -> pl.DataFrame:
    """Load embeddings for a dataset."""
    exp_path = Path(exp_path)
    emb_path = exp_path / f"{dataset}_embeddings.parquet"
    if emb_path.exists():
        return pl.read_parquet(emb_path)
    return None


@st.cache_data
def load_game_data(exp_path: str, dataset: str = "train") -> pl.DataFrame:
    """Load game metadata for a dataset."""
    exp_path = Path(exp_path)
    data_path = exp_path / f"{dataset}_data.parquet"
    if data_path.exists():
        return pl.read_parquet(data_path)
    return None


@st.cache_data
def load_component_loadings(exp_path: str) -> pd.DataFrame:
    """Load component loadings CSV."""
    exp_path = Path(exp_path)
    loadings_path = exp_path / "component_loadings.csv"
    if loadings_path.exists():
        return pd.read_csv(loadings_path, index_col=0)
    return None


def create_scree_plot(artifacts: dict) -> go.Figure:
    """Create scree plot from artifacts."""
    if "explained_variance_ratio" not in artifacts:
        return None

    var_ratio = np.array(artifacts["explained_variance_ratio"])
    cumulative_var = np.cumsum(var_ratio)
    n_components = len(var_ratio)
    x = np.arange(1, n_components + 1)

    fig = go.Figure()

    # Bar plot for individual variance
    fig.add_trace(go.Bar(
        x=x,
        y=var_ratio * 100,
        name="Individual",
        marker_color="#3498db",
        opacity=0.7,
    ))

    # Line plot for cumulative variance
    fig.add_trace(go.Scatter(
        x=x,
        y=cumulative_var * 100,
        mode="lines+markers",
        name="Cumulative",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=4),
    ))

    # Add threshold lines
    for thresh in [80, 90, 95]:
        fig.add_hline(y=thresh, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Scree Plot - Total Variance Explained: {cumulative_var[-1]:.1%}",
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        yaxis=dict(range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )

    return fig


def create_loadings_plot(loadings_df: pd.DataFrame, component: int, top_n: int = 15) -> go.Figure:
    """Create bar plot for component loadings."""
    col_name = f"PC{component}"
    if col_name not in loadings_df.columns:
        return None

    # Get top features by absolute loading
    loadings = loadings_df[col_name].copy()
    top_indices = loadings.abs().nlargest(top_n).index
    top_loadings = loadings.loc[top_indices].sort_values()

    # Create colors based on sign
    colors = ["#e74c3c" if v < 0 else "#3498db" for v in top_loadings.values]

    fig = go.Figure(go.Bar(
        x=top_loadings.values,
        y=top_loadings.index,
        orientation="h",
        marker_color=colors,
    ))

    fig.add_vline(x=0, line_color="black", line_width=1)

    fig.update_layout(
        title=f"PC{component} - Top {top_n} Feature Loadings",
        xaxis_title="Loading",
        yaxis_title="Feature",
        height=max(400, top_n * 25),
    )

    return fig


def create_2d_embedding_plot(
    embeddings_df: pl.DataFrame,
    game_data: pl.DataFrame,
    color_by: str = "predicted_complexity",
    sample_size: int = 10000,
    dim_x: int = 1,
    dim_y: int = 2,
) -> go.Figure:
    """Create 2D scatter plot of embeddings."""
    # Join embeddings with game data
    df = embeddings_df.join(game_data, on="game_id", how="left")

    # Sample if too large
    if len(df) > sample_size:
        df = df.sample(n=sample_size, seed=42)

    # Extract selected dimensions (convert to 0-indexed)
    all_embeddings = np.array([np.array(e) for e in df["embedding"].to_list()])
    x_vals = all_embeddings[:, dim_x - 1]
    y_vals = all_embeddings[:, dim_y - 1]

    # Prepare plot data
    plot_df = pd.DataFrame({
        "x": x_vals,
        "y": y_vals,
        "name": df["name"].to_list(),
        "game_id": df["game_id"].to_list(),
    })

    # Add color column if available
    if color_by in df.columns:
        plot_df[color_by] = df[color_by].to_list()
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color=color_by,
            hover_data=["name", "game_id"],
            color_continuous_scale="viridis",
            title=f"2D Embedding Projection (n={len(plot_df)})",
        )
    else:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            hover_data=["name", "game_id"],
            title=f"2D Embedding Projection (n={len(plot_df)})",
        )

    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.update_layout(
        xaxis_title=f"Component {dim_x}",
        yaxis_title=f"Component {dim_y}",
        height=600,
    )

    return fig


def find_similar_games(
    query_game_id: int,
    embeddings_df: pl.DataFrame,
    game_data: pl.DataFrame,
    top_k: int = 10,
    metric: str = "cosine",
) -> pd.DataFrame:
    """Find most similar games using cosine similarity or euclidean distance."""
    from sklearn.metrics.pairwise import euclidean_distances

    # Get all embeddings as matrix
    game_ids = embeddings_df["game_id"].to_list()
    embeddings = np.array([np.array(e) for e in embeddings_df["embedding"].to_list()])

    # Find query game index
    if query_game_id not in game_ids:
        return None

    query_idx = game_ids.index(query_game_id)
    query_embedding = embeddings[query_idx].reshape(1, -1)

    # Compute distances/similarities
    if metric == "cosine":
        scores = cosine_similarity(query_embedding, embeddings)[0]
        # Higher is better for cosine similarity
        top_indices = np.argsort(scores)[::-1][1:top_k + 1]
        score_label = "similarity"
    else:  # euclidean
        scores = euclidean_distances(query_embedding, embeddings)[0]
        # Lower is better for euclidean distance
        top_indices = np.argsort(scores)[1:top_k + 1]
        score_label = "distance"

    # Build results
    results = []
    for idx in top_indices:
        game_id = game_ids[idx]
        score = scores[idx]

        # Get game name from game_data
        game_info = game_data.filter(pl.col("game_id") == game_id)
        name = game_info["name"].to_list()[0] if len(game_info) > 0 else "Unknown"

        results.append({
            "game_id": game_id,
            "name": name,
            score_label: score,
        })

    return pd.DataFrame(results)


# Main UI
experiments = list_experiments()

if not experiments:
    st.warning("No embedding experiments found in models/experiments/embeddings/")
    st.info("Run an embedding training experiment first to see results here.")
    st.stop()

# Sidebar: Experiment selection
st.sidebar.header("Experiment Selection")
exp_names = [exp["name"] for exp in experiments]
selected_exp_name = st.sidebar.selectbox("Select Experiment", exp_names)

selected_exp = next(exp for exp in experiments if exp["name"] == selected_exp_name)
exp_path = selected_exp["path"]
metadata = selected_exp["metadata"]

# Show experiment info in sidebar
st.sidebar.subheader("Experiment Info")
st.sidebar.write(f"**Algorithm:** {metadata['metadata'].get('algorithm', 'N/A')}")
st.sidebar.write(f"**Dimensions:** {metadata['metadata'].get('embedding_dim', 'N/A')}")
st.sidebar.write(f"**Train samples:** {metadata['metadata'].get('train_samples_after_filter', 'N/A')}")
st.sidebar.write(f"**Min ratings:** {metadata['metadata'].get('min_ratings', 'N/A')}")

# Load data
artifacts = load_artifacts(str(exp_path))
loadings_df = load_component_loadings(str(exp_path))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Variance & Loadings",
    "2D Visualization",
    "Similar Games",
    "Experiment Details",
])

with tab1:
    st.header("Variance Explained & Component Loadings")

    st.subheader("Scree Plot")
    scree_fig = create_scree_plot(artifacts)
    if scree_fig:
        st.plotly_chart(scree_fig, use_container_width=True)
    else:
        # Try loading saved image
        scree_path = exp_path / "scree_plot.png"
        if scree_path.exists():
            st.image(str(scree_path))
        else:
            st.info("No variance data available")

    st.subheader("Component Loadings")
    if loadings_df is not None:
        n_components = len(loadings_df.columns)
        selected_pc = st.selectbox(
            "Select Component",
            list(range(1, min(n_components + 1, 65))),
            key="loadings_pc",
        )
        top_n_features = st.slider("Top N Features", 10, 30, 15, key="loadings_top_n")

        loadings_fig = create_loadings_plot(loadings_df, selected_pc, top_n_features)
        if loadings_fig:
            st.plotly_chart(loadings_fig, use_container_width=True)
    else:
        # Try loading saved image
        loadings_path = exp_path / "component_loadings.png"
        if loadings_path.exists():
            st.image(str(loadings_path))
        else:
            st.info("No component loadings available")

with tab2:
    st.header("2D Embedding Visualization")

    dataset = st.selectbox("Dataset", ["train", "tune", "test"], key="viz_dataset")

    embeddings_df = load_embeddings(str(exp_path), dataset)
    game_data = load_game_data(str(exp_path), dataset)

    if embeddings_df is not None and game_data is not None:
        # Get embedding dimension
        sample_embedding = embeddings_df["embedding"].to_list()[0]
        n_dims = len(np.array(sample_embedding))

        # Dimension selectors
        col1, col2 = st.columns(2)
        with col1:
            dim_x = st.selectbox("X-axis dimension", list(range(1, n_dims + 1)), index=0, key="dim_x")
        with col2:
            dim_y = st.selectbox("Y-axis dimension", list(range(1, n_dims + 1)), index=1, key="dim_y")

        # Color options
        available_colors = [col for col in game_data.columns if col not in ["game_id", "name"]]
        color_by = st.selectbox(
            "Color by",
            ["predicted_complexity"] + available_colors if "predicted_complexity" in available_colors else available_colors,
            key="viz_color",
        )

        sample_size = st.slider("Sample size", 1000, 50000, 10000, step=1000, key="viz_sample")

        fig = create_2d_embedding_plot(embeddings_df, game_data, color_by, sample_size, dim_x, dim_y)
        st.plotly_chart(fig, use_container_width=True)

        # Show table with all components
        st.subheader("Embedding Components")

        # Join embeddings with game data
        df = embeddings_df.join(game_data, on="game_id", how="left")

        # Sample same as visualization
        if len(df) > sample_size:
            df = df.sample(n=sample_size, seed=42)

        # Extract all embedding components into columns
        all_embeddings = np.array([np.array(e) for e in df["embedding"].to_list()])

        # Build table with game info and components
        table_data = {
            "game_id": df["game_id"].to_list(),
            "name": df["name"].to_list(),
        }

        # Add color column if available
        if color_by in df.columns:
            table_data[color_by] = df[color_by].to_list()

        # Add all embedding components
        for i in range(all_embeddings.shape[1]):
            table_data[f"PC{i+1}"] = all_embeddings[:, i]

        table_df = pd.DataFrame(table_data)
        st.dataframe(table_df, use_container_width=True, height=400)
    else:
        # Try loading saved image
        viz_path = exp_path / f"{dataset}_visualization_2d.png"
        if viz_path.exists():
            st.image(str(viz_path))
        else:
            st.warning(f"No embedding data available for {dataset} dataset")

with tab3:
    st.header("Find Similar Games")

    # Combine all datasets
    all_embeddings = []
    all_game_data = []
    for ds in ["train", "tune", "test"]:
        emb = load_embeddings(str(exp_path), ds)
        data = load_game_data(str(exp_path), ds)
        if emb is not None:
            all_embeddings.append(emb)
        if data is not None:
            all_game_data.append(data)

    if all_embeddings and all_game_data:
        embeddings_df = pl.concat(all_embeddings).unique(subset=["game_id"])
        game_data = pl.concat(all_game_data).unique(subset=["game_id"])
        st.info(f"Searching across {len(embeddings_df):,} games from all datasets")
        # Search for a game
        search_term = st.text_input("Search for a game by name", key="game_search")

        if search_term:
            # Filter games matching search
            matches = game_data.filter(
                pl.col("name").str.to_lowercase().str.contains(search_term.lower())
            ).head(20)

            if len(matches) > 0:
                # Let user select from matches - sort by name length then alphabetically for consistent ordering
                matches_sorted = matches.sort(
                    pl.col("name").str.len_chars(),
                    pl.col("name"),
                )
                match_options = {}
                for row in matches_sorted.iter_rows(named=True):
                    year = row.get("year_published", "")
                    year_str = f", {int(year)}" if year else ""
                    label = f"{row['name']} (ID: {row['game_id']}{year_str})"
                    match_options[label] = row["game_id"]
                options_list = list(match_options.keys())

                selected_game = st.selectbox(
                    "Select a game",
                    options_list,
                    key="selected_game_stable",
                )

                if selected_game and selected_game in match_options:
                    game_id = match_options[selected_game]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        top_k = st.slider("Number of similar games", 5, 50, 10, key="top_k")
                    with col2:
                        metric = st.selectbox(
                            "Distance metric",
                            ["cosine", "euclidean"],
                            key="distance_metric",
                        )
                    with col3:
                        min_ratings = st.slider("Min user ratings", 0, 100, 25, key="min_ratings")

                    # Complexity filter
                    col4, col5 = st.columns([1, 2])
                    with col4:
                        filter_complexity = st.checkbox("Filter by complexity", key="filter_complexity")
                    with col5:
                        complexity_mode = st.selectbox(
                            "Mode",
                            ["Within band (±)", "Less complex", "More complex"],
                            key="complexity_mode",
                            disabled=not filter_complexity,
                        )

                    # For directional modes, optionally limit the range
                    is_directional = complexity_mode in ["Less complex", "More complex"]
                    if filter_complexity and is_directional:
                        col6, col7 = st.columns([1, 2])
                        with col6:
                            limit_range = st.checkbox("Limit range", key="limit_range")
                        with col7:
                            complexity_band = st.slider(
                                "Max difference",
                                min_value=0.25,
                                max_value=2.0,
                                value=1.0,
                                step=0.25,
                                key="complexity_band_directional",
                                disabled=not limit_range,
                            )
                    elif filter_complexity:
                        # Band mode always uses range
                        limit_range = True
                        complexity_band = st.slider(
                            "Band (±)",
                            min_value=0.25,
                            max_value=2.0,
                            value=0.5,
                            step=0.25,
                            key="complexity_band_band",
                        )
                    else:
                        limit_range = False
                        complexity_band = 0.5

                    # Get query game's complexity
                    query_game_info = game_data.filter(pl.col("game_id") == game_id)
                    query_complexity = None
                    if len(query_game_info) > 0 and "predicted_complexity" in query_game_info.columns:
                        query_complexity = query_game_info["predicted_complexity"].to_list()[0]

                    if filter_complexity and query_complexity is not None:
                        if complexity_mode == "Within band (±)":
                            range_str = f"[{query_complexity - complexity_band:.2f}, {query_complexity + complexity_band:.2f}]"
                        elif complexity_mode == "Less complex":
                            if limit_range:
                                range_str = f"[{max(0, query_complexity - complexity_band):.2f}, {query_complexity:.2f}]"
                            else:
                                range_str = f"[0, {query_complexity:.2f}]"
                        else:  # More complex
                            if limit_range:
                                range_str = f"[{query_complexity:.2f}, {query_complexity + complexity_band:.2f}]"
                            else:
                                range_str = f"[{query_complexity:.2f}, 5]"
                        st.caption(f"Query game complexity: {query_complexity:.2f} → searching {range_str}")

                    # Build filter conditions
                    filter_conditions = pl.col("game_id") == game_id  # Always include query game

                    # Min ratings filter
                    if min_ratings > 0 and "users_rated" in game_data.columns:
                        filter_conditions = filter_conditions | (pl.col("users_rated") >= min_ratings)
                    else:
                        filter_conditions = pl.lit(True)

                    # Complexity filter
                    if filter_complexity and query_complexity is not None and "predicted_complexity" in game_data.columns:
                        if complexity_mode == "Within band (±)":
                            complexity_condition = (
                                (pl.col("predicted_complexity") >= query_complexity - complexity_band) &
                                (pl.col("predicted_complexity") <= query_complexity + complexity_band)
                            )
                        elif complexity_mode == "Less complex":
                            if limit_range:
                                complexity_condition = (
                                    (pl.col("predicted_complexity") >= query_complexity - complexity_band) &
                                    (pl.col("predicted_complexity") <= query_complexity)
                                )
                            else:
                                # No lower bound - all games less complex
                                complexity_condition = pl.col("predicted_complexity") <= query_complexity
                        else:  # More complex
                            if limit_range:
                                complexity_condition = (
                                    (pl.col("predicted_complexity") >= query_complexity) &
                                    (pl.col("predicted_complexity") <= query_complexity + complexity_band)
                                )
                            else:
                                # No upper bound - all games more complex
                                complexity_condition = pl.col("predicted_complexity") >= query_complexity
                        # Combine: (meets min_ratings AND meets complexity) OR is query game
                        if min_ratings > 0 and "users_rated" in game_data.columns:
                            filter_conditions = (
                                (pl.col("game_id") == game_id) |
                                ((pl.col("users_rated") >= min_ratings) & complexity_condition)
                            )
                        else:
                            filter_conditions = (pl.col("game_id") == game_id) | complexity_condition

                    filtered_game_ids = game_data.filter(filter_conditions)["game_id"].to_list()
                    filtered_embeddings = embeddings_df.filter(pl.col("game_id").is_in(filtered_game_ids))

                    similar_df = find_similar_games(game_id, filtered_embeddings, game_data, top_k, metric)

                    if similar_df is not None:
                        st.subheader(f"Games similar to: {selected_game.split(' (ID:')[0]}")

                        # Get all game IDs to compare (query + similar)
                        compare_ids = [game_id] + similar_df["game_id"].tolist()

                        # Get embeddings for these games
                        compare_embeddings = embeddings_df.filter(
                            pl.col("game_id").is_in(compare_ids)
                        )
                        compare_game_data = game_data.filter(
                            pl.col("game_id").is_in(compare_ids)
                        )

                        # Build embedding matrix
                        emb_matrix = np.array([np.array(e) for e in compare_embeddings["embedding"].to_list()])
                        game_ids_ordered = compare_embeddings["game_id"].to_list()

                        # Create mapping of game_id to name and embeddings
                        id_to_name = {
                            row["game_id"]: row["name"]
                            for row in compare_game_data.iter_rows(named=True)
                        }
                        id_to_emb = {
                            gid: emb_matrix[idx]
                            for idx, gid in enumerate(game_ids_ordered)
                        }

                        n_dims = emb_matrix.shape[1]

                        # Add components to the similar_df table
                        table_df = similar_df.copy()
                        for i in range(n_dims):
                            table_df[f"PC{i+1}"] = table_df["game_id"].apply(
                                lambda gid, idx=i: id_to_emb.get(gid, [None] * n_dims)[idx]
                            )

                        # Format columns for display
                        score_col = "similarity" if metric == "cosine" else "distance"
                        format_dict = {score_col: "{:.4f}"}
                        for i in range(n_dims):
                            format_dict[f"PC{i+1}"] = "{:.3f}"

                        st.dataframe(
                            table_df.style.format(format_dict),
                            use_container_width=True,
                        )

                        # Component profile plot
                        st.subheader("Embedding Comparison")

                        # Build long-form data for line plot
                        line_data = []
                        for idx, gid in enumerate(game_ids_ordered):
                            name = id_to_name.get(gid, str(gid))
                            is_query = gid == game_id
                            for comp in range(n_dims):
                                line_data.append({
                                    "game": name,
                                    "component": comp + 1,
                                    "value": emb_matrix[idx, comp],
                                    "is_query": is_query,
                                })

                        line_df = pd.DataFrame(line_data)

                        # Create line plot
                        fig = px.line(
                            line_df,
                            x="component",
                            y="value",
                            color="game",
                            title="Embedding Profile by Component",
                        )

                        # Make query game thicker/more prominent
                        query_name = id_to_name.get(game_id, str(game_id))
                        for trace in fig.data:
                            if trace.name == query_name:
                                trace.line.width = 4
                            else:
                                trace.line.width = 1.5
                                trace.opacity = 0.6

                        fig.update_layout(
                            height=500,
                            xaxis_title="Component",
                            yaxis_title="Value",
                            legend_title="Game",
                            legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.15,
                                xanchor="left",
                                x=0,
                            ),
                            margin=dict(b=100),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Game not found in embeddings")
            else:
                st.info("No games found matching your search")
    else:
        st.warning("No embedding data available")

with tab4:
    st.header("Experiment Details")

    st.subheader("Metadata")
    st.json(metadata)

    # Show metrics if available
    for dataset in ["train", "tune", "test"]:
        metrics_path = exp_path / f"{dataset}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.subheader(f"{dataset.title()} Metrics")
            st.json(metrics)

# Footer
render_footer()
