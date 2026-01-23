"""
Streamlit page for exploring text embeddings from game descriptions.

This page allows you to:
- Search for words and find similar words
- Compare word similarities
- Generate document embeddings from descriptions
- Visualize word embeddings in 2D
- Explore vocabulary and component loadings
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from sklearn.decomposition import PCA

# Add project root to Python path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, project_root)

from src.models.text_embeddings.trainer import TextEmbeddingGenerator  # noqa: E402
from src.streamlit.components.footer import render_footer  # noqa: E402

# Page config
st.set_page_config(page_title="Text Embeddings | BGG Models Dashboard", layout="wide")
st.title("📝 Text Embeddings Explorer")

# Constants
EXPERIMENTS_DIR = Path(project_root) / "models" / "experiments" / "text_embeddings"
GAME_FEATURES_PATH = Path(project_root) / "data" / "raw" / "game_features.parquet"


def get_available_experiments() -> list[str]:
    """Get list of available text embedding experiments."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted([d.name for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])


def get_available_versions(experiment_name: str) -> list[str]:
    """Get list of available versions for an experiment."""
    exp_path = EXPERIMENTS_DIR / experiment_name
    if not exp_path.exists():
        return []
    versions = sorted([d.name for d in exp_path.iterdir() if d.is_dir()])
    return versions


def get_version_dir(experiment_name: str, version: str) -> Path:
    """Get the directory for a specific experiment version."""
    return EXPERIMENTS_DIR / experiment_name / version


@st.cache_resource
def load_generator(experiment_name: str, version: int) -> TextEmbeddingGenerator:
    """Load the text embedding generator."""
    return TextEmbeddingGenerator(experiment_name, version=version)


@st.cache_data
def load_vocab_stats(version_dir: str) -> dict:
    """Load vocabulary statistics."""
    stats_path = Path(version_dir) / "vocab_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_word_similarities(version_dir: str) -> dict:
    """Load pre-computed word similarities."""
    sim_path = Path(version_dir) / "word_similarities.json"
    if sim_path.exists():
        with open(sim_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_svd_analysis(version_dir: str) -> dict:
    """Load SVD analysis."""
    svd_path = Path(version_dir) / "svd_analysis.json"
    if svd_path.exists():
        with open(svd_path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_component_loadings(version_dir: str) -> pl.DataFrame:
    """Load component loadings."""
    loadings_path = Path(version_dir) / "component_loadings.csv"
    if loadings_path.exists():
        return pl.read_csv(loadings_path)
    return pl.DataFrame()


@st.cache_data
def load_2d_projection(version_dir: str) -> pl.DataFrame:
    """Load 2D word embedding projection."""
    proj_path = Path(version_dir) / "word_embeddings_2d.parquet"
    if proj_path.exists():
        return pl.read_parquet(proj_path)
    return pl.DataFrame()


@st.cache_data
def load_game_data(top_n: int = 25000) -> pl.DataFrame:
    """Load top games by user ratings with descriptions."""
    if not GAME_FEATURES_PATH.exists():
        return pl.DataFrame()

    df = pl.read_parquet(GAME_FEATURES_PATH)

    # Filter to games with descriptions
    df = df.filter(pl.col("description").is_not_null() & (pl.col("description") != ""))

    # Get top N by geek_rating
    df = df.sort("geek_rating", descending=True).head(top_n)

    return df


@st.cache_data
def compute_document_embeddings(
    _generator: TextEmbeddingGenerator,
    descriptions: list[str],
    game_ids: list[int],
    names: list[str],
    experiment_version: str,  # Include in cache key to invalidate on model change
) -> pl.DataFrame:
    """Compute document embeddings and 2D projection."""
    # Generate embeddings
    embeddings = _generator.embed_documents(descriptions)

    # PCA to 2D
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings)

    # Truncate and wrap descriptions for tooltip
    def format_description(desc: str, max_len: int = 200, line_len: int = 50) -> str:
        """Truncate and wrap description for tooltip display."""
        if len(desc) > max_len:
            desc = desc[:max_len] + "..."
        # Wrap at word boundaries
        words = desc.split()
        lines = []
        current_line = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > line_len and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_len = len(word)
            else:
                current_line.append(word)
                current_len += len(word) + 1
        if current_line:
            lines.append(" ".join(current_line))
        return "<br>".join(lines)

    truncated_descriptions = [format_description(d) for d in descriptions]

    # Create dataframe
    df = pl.DataFrame(
        {
            "game_id": game_ids,
            "name": names,
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "description": truncated_descriptions,
        }
    )

    return df, embeddings, pca.explained_variance_ratio_


def find_similar_documents(
    query_idx: int,
    embeddings: np.ndarray,
    n: int = 10,
) -> list[tuple[int, float]]:
    """Find most similar documents by cosine similarity."""
    query_vec = embeddings[query_idx]
    query_norm = np.linalg.norm(query_vec)

    similarities = []
    for i, vec in enumerate(embeddings):
        if i == query_idx:
            continue
        sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec) + 1e-10)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


# Sidebar for experiment selection
st.sidebar.header("Experiment Selection")

experiments = get_available_experiments()
if not experiments:
    st.warning("No text embedding experiments found. Run `make text_embeddings` first.")
    render_footer()
    st.stop()

selected_experiment = st.sidebar.selectbox(
    "Experiment",
    experiments,
    index=0,
)

# Get versions for selected experiment
versions = get_available_versions(selected_experiment)
if not versions:
    st.warning(f"No versions found for experiment '{selected_experiment}'.")
    render_footer()
    st.stop()

selected_version = st.sidebar.selectbox(
    "Version",
    versions,
    index=len(versions) - 1,  # Default to latest version
)

# Get version directory path
version_dir = str(get_version_dir(selected_experiment, selected_version))

# Parse version number for generator (strip 'v' prefix if present)
version_num = (
    int(selected_version.lstrip("v"))
    if selected_version.startswith("v")
    else int(selected_version)
)

# Load model and artifacts
try:
    generator = load_generator(selected_experiment, version_num)
    vocab_stats = load_vocab_stats(version_dir)
    word_similarities = load_word_similarities(version_dir)
    svd_analysis = load_svd_analysis(version_dir)
    component_loadings = load_component_loadings(version_dir)
    projection_2d = load_2d_projection(version_dir)
except Exception as e:
    st.error(f"Error loading experiment: {e}")
    render_footer()
    st.stop()

# Display model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
if vocab_stats:
    st.sidebar.metric("Vocabulary Size", f"{vocab_stats.get('vocab_size', 0):,}")
st.sidebar.metric("Embedding Dim", generator.word_model.get_embedding_dim())

# Cache controls
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Word Explorer",
        "Word Relationships",
        "Document Embeddings",
        "Word Visualization",
        "Document Explorer",
        "Vocabulary & Analysis",
    ]
)

# Tab 1: Word Explorer
with tab1:
    st.header("Word Explorer")
    st.markdown("Search for a word to see its similar words and embedding information.")

    col1, col2 = st.columns([1, 2])

    with col1:
        search_word = (
            st.text_input(
                "Enter a word",
                placeholder="strategy",
                key="word_search",
            )
            .lower()
            .strip()
        )

        n_similar = st.slider("Number of similar words", 5, 50, 15)

    with col2:
        if search_word:
            vec = generator.get_word_vector(search_word)
            if vec is not None:
                st.success(f"Found '{search_word}' in vocabulary")

                # Get similar words
                similar_words = generator.most_similar_words(search_word, n=n_similar)

                if similar_words:
                    st.subheader("Most Similar Words")
                    sim_df = pl.DataFrame(
                        [
                            {"word": w, "similarity": round(s, 4)}
                            for w, s in similar_words
                        ]
                    )
                    st.dataframe(
                        sim_df.to_pandas(),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Show vector stats
                with st.expander("Embedding Vector Details"):
                    st.write(f"**Dimension:** {len(vec)}")
                    st.write(f"**Norm:** {np.linalg.norm(vec):.4f}")
                    st.write(f"**Mean:** {vec.mean():.4f}")
                    st.write(f"**Std:** {vec.std():.4f}")
                    st.code(f"[{', '.join(f'{v:.3f}' for v in vec[:10])}...]")
            else:
                st.warning(f"'{search_word}' not found in vocabulary")

# Tab 2: Word Relationships
with tab2:
    st.header("Word Relationships")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Compare Two Words")
        word1 = (
            st.text_input("First word", placeholder="strategy", key="word1")
            .lower()
            .strip()
        )
        word2 = (
            st.text_input("Second word", placeholder="tactical", key="word2")
            .lower()
            .strip()
        )

        if word1 and word2:
            vec1 = generator.get_word_vector(word1)
            vec2 = generator.get_word_vector(word2)

            if vec1 is not None and vec2 is not None:
                similarity = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10
                )
                st.metric("Cosine Similarity", f"{similarity:.4f}")

                # Visual indicator
                if similarity > 0.7:
                    st.success("Very similar")
                elif similarity > 0.4:
                    st.info("Somewhat similar")
                elif similarity > 0.1:
                    st.warning("Weakly related")
                else:
                    st.error("Not related")
            else:
                missing = []
                if vec1 is None:
                    missing.append(word1)
                if vec2 is None:
                    missing.append(word2)
                st.warning(f"Word(s) not in vocabulary: {', '.join(missing)}")

    with col2:
        st.subheader("Pre-computed Similarities")
        st.markdown("Similar words for common game terms:")

        if word_similarities:
            seed_word = st.selectbox(
                "Select seed word",
                list(word_similarities.keys()),
            )
            if seed_word in word_similarities:
                sim_list = word_similarities[seed_word]
                sim_df = pl.DataFrame(sim_list)
                st.dataframe(
                    sim_df.to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No pre-computed similarities available")

# Tab 3: Document Embeddings
with tab3:
    st.header("Document Embeddings")
    st.markdown("Enter a game description to see its embedding.")

    sample_descriptions = [
        "A strategy game where players build civilizations and compete for resources.",
        "A cooperative card game where players work together to defeat monsters.",
        "A fast-paced dice rolling game perfect for family game night.",
        "An economic trading game set in medieval Europe.",
    ]

    use_sample = st.checkbox("Use sample description")
    if use_sample:
        description = st.selectbox("Select sample", sample_descriptions)
    else:
        description = st.text_area(
            "Enter description",
            placeholder="Enter a game description...",
            height=150,
        )

    if description:
        embedding = generator.embed_document(description)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Embedding Stats")
            st.metric("Dimension", len(embedding))
            st.metric("Norm", f"{np.linalg.norm(embedding):.4f}")
            st.metric("Mean", f"{embedding.mean():.4f}")
            st.metric("Std", f"{embedding.std():.4f}")

        with col2:
            st.subheader("Embedding Vector")
            # Bar chart of embedding values
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(range(len(embedding))),
                        y=embedding,
                        marker_color="steelblue",
                    )
                ]
            )
            fig.update_layout(
                xaxis_title="Dimension",
                yaxis_title="Value",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw Vector"):
            st.code(str(embedding.tolist()))

# Tab 4: Word Visualization
with tab4:
    st.header("Word Embeddings Visualization")

    if projection_2d.is_empty():
        st.warning("No 2D projection available for this experiment.")
    else:
        st.markdown("2D PCA projection of word embeddings.")

        # Filtering options
        col1, col2 = st.columns(2)
        with col1:
            min_freq = st.slider(
                "Minimum word frequency",
                10,
                int(projection_2d["frequency"].max()),
                int(projection_2d["frequency"].quantile(0.5)),
            )
        with col2:
            max_words = st.slider(
                "Maximum words to display",
                100,
                25000,
                1000,
                step=100,
            )

        # Filter data
        filtered = projection_2d.filter(pl.col("frequency") >= min_freq)
        if len(filtered) > max_words:
            filtered = filtered.sort("frequency", descending=True).head(max_words)

        # Color options
        color_by = st.radio(
            "Color by",
            ["Frequency (log)", "None"],
            horizontal=True,
        )

        # Create scatter plot
        df_plot = filtered.to_pandas()

        # Add log frequency column (in thousands for display)
        df_plot["freq_k"] = df_plot["frequency"] / 1000
        df_plot["log_freq"] = np.log10(df_plot["frequency"] + 1)

        if color_by == "Frequency (log)":
            fig = px.scatter(
                df_plot,
                x="x",
                y="y",
                color="log_freq",
                color_continuous_scale="Viridis",
                custom_data=["word", "frequency"],
                title=f"Word Embeddings ({len(df_plot):,} words)",
            )
        else:
            fig = px.scatter(
                df_plot,
                x="x",
                y="y",
                custom_data=["word", "frequency"],
                title=f"Word Embeddings ({len(df_plot):,} words)",
            )

        fig.update_traces(
            marker=dict(size=5, opacity=0.7),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,}<extra></extra>",
        )
        fig.update_layout(
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show some sample words
        with st.expander("Sample High-Frequency Words"):
            top_words = filtered.sort("frequency", descending=True).head(50)
            st.dataframe(
                top_words.select(["word", "frequency", "x", "y"]).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )

# Tab 5: Document Explorer
with tab5:
    st.header("Document Explorer")
    st.markdown(
        "Visualize document embeddings for sample games and find similar descriptions."
    )

    # Load game data
    game_data = load_game_data(top_n=25000)

    if game_data.is_empty():
        st.warning(
            f"Game data not found at `{GAME_FEATURES_PATH}`. "
            "Run data loading first to use this feature."
        )
    else:
        st.info(f"Loaded {len(game_data):,} games with descriptions.")

        # Compute document embeddings
        with st.spinner("Computing document embeddings..."):
            descriptions = game_data["description"].to_list()
            game_ids = game_data["game_id"].to_list()
            names = game_data["name"].to_list()

            doc_projection, doc_embeddings, doc_variance = compute_document_embeddings(
                generator, descriptions, game_ids, names,
                experiment_version=f"{selected_experiment}/{selected_version}",
            )

        # Show variance explained
        st.caption(
            f"2D PCA explains {sum(doc_variance) * 100:.1f}% of variance "
            f"(PC1: {doc_variance[0] * 100:.1f}%, PC2: {doc_variance[1] * 100:.1f}%)"
        )

        # Create the visualization
        df_plot = doc_projection.to_pandas()

        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            custom_data=["name", "game_id", "description"],
            title=f"Document Embeddings ({len(df_plot):,} games)",
        )
        fig.update_traces(
            marker=dict(size=6, opacity=0.7),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{customdata[2]}<extra></extra>",
        )
        fig.update_layout(hoverlabel=dict(align="left"))
        fig.update_layout(
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Similarity search
        st.markdown("---")
        st.subheader("Find Similar Games by Description")

        # Game selector
        game_options = {
            f"{row['name']} ({row['game_id']})": i
            for i, row in enumerate(game_data.iter_rows(named=True))
        }
        selected_game = st.selectbox(
            "Select a game to find similar descriptions",
            options=list(game_options.keys()),
            index=0,
        )

        if selected_game:
            query_idx = game_options[selected_game]
            query_row = game_data.row(query_idx, named=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Selected Game:**")
                st.markdown(f"**{query_row['name']}** (ID: {query_row['game_id']})")
                with st.expander("View Description"):
                    st.write(
                        query_row["description"][:1000] + "..."
                        if len(query_row["description"]) > 1000
                        else query_row["description"]
                    )

            with col2:
                st.markdown("**Most Similar Games:**")
                similar = find_similar_documents(query_idx, doc_embeddings, n=10)

                similar_data = []
                for idx, sim in similar:
                    row = game_data.row(idx, named=True)
                    similar_data.append(
                        {
                            "name": row["name"],
                            "game_id": row["game_id"],
                            "similarity": round(sim, 4),
                        }
                    )

                sim_df = pl.DataFrame(similar_data)
                st.dataframe(
                    sim_df.to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                )

            # Visualization of actual embeddings
            if similar:
                st.markdown("---")
                st.subheader("Embedding Vectors")

                # Get top 5 neighbors for visualization
                top_neighbors = similar[:5]
                neighbor_indices = [idx for idx, _ in top_neighbors]

                # Build data for heatmap
                games_to_show = [query_idx] + neighbor_indices
                game_names = [query_row["name"][:30]] + [
                    game_data.row(idx, named=True)["name"][:30]
                    for idx in neighbor_indices
                ]
                similarities_labels = ["Selected"] + [
                    f"{sim:.3f}" for _, sim in top_neighbors
                ]
                labels = [
                    f"{name} ({sim})"
                    for name, sim in zip(game_names, similarities_labels)
                ]

                # Get embedding vectors
                embedding_matrix = np.array(
                    [doc_embeddings[idx] for idx in games_to_show]
                )

                # Create heatmap
                fig_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=embedding_matrix,
                        y=labels,
                        x=[f"D{i}" for i in range(embedding_matrix.shape[1])],
                        colorscale="RdBu",
                        zmid=0,
                    )
                )
                fig_heatmap.update_layout(
                    title="Document Embedding Vectors",
                    xaxis_title="Dimension",
                    yaxis_title="Game",
                    height=350,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Line plot overlay
                fig_lines = go.Figure()
                colors = ["red"] + ["steelblue"] * len(neighbor_indices)
                for i, (idx, label, color) in enumerate(
                    zip(games_to_show, labels, colors)
                ):
                    fig_lines.add_trace(
                        go.Scatter(
                            x=list(range(embedding_matrix.shape[1])),
                            y=doc_embeddings[idx],
                            mode="lines",
                            name=label,
                            line=dict(color=color, width=2 if i == 0 else 1),
                            opacity=1.0 if i == 0 else 0.6,
                        )
                    )
                fig_lines.update_layout(
                    title="Embedding Vector Comparison",
                    xaxis_title="Dimension",
                    yaxis_title="Value",
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_lines, use_container_width=True)

            # Show description comparison for top match
            if similar:
                top_idx, top_sim = similar[0]
                top_row = game_data.row(top_idx, named=True)

                st.markdown("---")
                st.subheader(f"Description Comparison (Similarity: {top_sim:.4f})")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{query_row['name']}**")
                    desc1 = query_row["description"]
                    st.write(desc1[:800] + "..." if len(desc1) > 800 else desc1)

                with col2:
                    st.markdown(f"**{top_row['name']}**")
                    desc2 = top_row["description"]
                    st.write(desc2[:800] + "..." if len(desc2) > 800 else desc2)


# Tab 6: Vocabulary & Analysis
with tab6:
    st.header("Vocabulary & Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Top Words by Frequency")
        if vocab_stats:
            top_words = vocab_stats.get("top_words", [])
            if top_words:
                top_df = pl.DataFrame(top_words)
                st.dataframe(
                    top_df.to_pandas(),
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )
        else:
            st.info("No vocabulary statistics available.")

    with col2:
        st.subheader("Statistics")
        if vocab_stats:
            st.metric("Total Vocabulary", f"{vocab_stats.get('vocab_size', 0):,}")
            top_words = vocab_stats.get("top_words", [])
            if top_words:
                st.metric("Most Common Word", top_words[0]["word"])
                st.metric("Top Word Count", f"{top_words[0]['count']:,}")

    # SVD Analysis
    st.markdown("---")
    st.subheader("SVD Analysis")

    if svd_analysis:
        explained = svd_analysis.get("cumulative_variance_ratio", [])
        if explained:
            # Scree plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(explained) + 1)),
                    y=[e * 100 for e in explained],
                    mode="lines+markers",
                    name="Cumulative Variance",
                )
            )
            fig.update_layout(
                xaxis_title="Number of Components",
                yaxis_title="Cumulative Variance Explained (%)",
                title="Variance Explained by SVD Components",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No SVD analysis available.")

    # Component Loadings
    st.markdown("---")
    st.subheader("Component Loadings")

    if not component_loadings.is_empty():
        selected_component = st.selectbox(
            "Select Component",
            sorted(component_loadings["component"].unique().to_list()),
        )

        comp_data = component_loadings.filter(
            pl.col("component") == selected_component
        ).sort("loading")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Positive Loadings**")
            pos_data = (
                comp_data.filter(pl.col("direction") == "positive")
                .sort("loading", descending=True)
                .head(15)
            )
            st.dataframe(
                pos_data.select(["word", "loading"]).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )

        with col2:
            st.markdown("**Top Negative Loadings**")
            neg_data = (
                comp_data.filter(pl.col("direction") == "negative")
                .sort("loading")
                .head(15)
            )
            st.dataframe(
                neg_data.select(["word", "loading"]).to_pandas(),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No component loadings available.")

# Footer
render_footer()
