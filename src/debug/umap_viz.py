"""Streamlit app to visualize and compare 2D projections."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Game Embeddings Projections", layout="wide")
st.title("Game Embeddings 2D Projections")

data_dir = Path(__file__).parent.parent.parent / "data"
exp_dir = Path(__file__).parent.parent.parent / "models/experiments/embeddings/svd-embeddings/v1"

@st.cache_data
def load_data():
    # Load projections
    proj = pd.read_parquet(data_dir / "projection_comparison.parquet")

    # Load data files with additional fields
    train_data = pd.read_parquet(exp_dir / "train_data.parquet")
    tune_data = pd.read_parquet(exp_dir / "tune_data.parquet")
    test_data = pd.read_parquet(exp_dir / "test_data.parquet")

    all_data = pd.concat([train_data, tune_data, test_data], ignore_index=True)

    # Merge projections with data
    df = proj.merge(all_data, on="game_id", how="left", suffixes=("", "_data"))

    # Clean up duplicate name column if exists
    if "name_data" in df.columns:
        df = df.drop(columns=["name_data"])

    # Add log users_rated
    df["users_rated_log"] = np.log10(df["users_rated"].clip(lower=1))

    # Calculate geek rating (Bayesian average)
    df["geek_rating"] = ((2000 * 5.5) + (df["average_rating"] * df["users_rated"])) / (df["users_rated"] + 2000)

    return df

df = load_data()

st.write(f"Loaded {len(df):,} games")

# Sidebar controls
st.sidebar.header("Controls")

# Game selector - searchable multi-select (use full df for options)
game_options = df.sort_values("users_rated", ascending=False)["name"].tolist()
selected_games = st.sidebar.multiselect(
    "Select Games to Highlight",
    options=game_options,
    max_selections=15,
    placeholder="Type to search and select...",
)

# Users rated filter
min_ratings = st.sidebar.slider(
    "Minimum Users Rated",
    min_value=0,
    max_value=1000,
    value=0,
    step=10,
)

# Filter data
df_filtered = df[df["users_rated"] >= min_ratings]
st.sidebar.write(f"Showing {len(df_filtered):,} games")

# Projection selector
# Build projection options dynamically
projection_options = [("PCA", "pca")]
for metric in ['euclidean', 'cosine']:
    metric_short = 'euc' if metric == 'euclidean' else 'cos'
    for n in [100]:
        for d in [0.1, 0.5, 0.8]:
            dist_short = str(d).replace('.', '')
            label = f"UMAP ({metric}, n={n}, d={d})"
            value = f"umap_{metric_short}_n{n}_d{dist_short}"
            projection_options.append((label, value))

projection = st.sidebar.selectbox(
    "Projection Method",
    options=projection_options,
    format_func=lambda x: x[0],
)

# Color by selector
color_by = st.sidebar.selectbox(
    "Color By",
    options=[
        ("None", None),
        ("Geek Rating", "geek_rating"),
        ("Average Rating", "average_rating"),
        ("Users Rated (log)", "users_rated_log"),
        ("Predicted Complexity", "predicted_complexity"),
        ("Year Published", "year_published"),
    ],
    format_func=lambda x: x[0],
)

proj_prefix = projection[1]
x_col = f"{proj_prefix}_1"
y_col = f"{proj_prefix}_2"
color_col = color_by[1]

# Build plot
hover_template = (
    "<b>%{customdata[0]}</b> (%{customdata[2]})<br>"
    "Game ID: %{customdata[1]}<br>"
    "Users Rated: %{customdata[3]:,}<br>"
    "Avg Rating: %{customdata[4]:.2f}<br>"
    "Geek Rating: %{customdata[5]:.2f}<br>"
    "Complexity: %{customdata[6]:.2f}"
    "<extra></extra>"
)

custom_data = df_filtered[["name", "game_id", "year_published", "users_rated", "average_rating", "geek_rating", "predicted_complexity"]]

if color_col:
    # Set color range bounds
    range_color = None
    if color_col == "year_published":
        range_color = [1980, 2026]
    elif color_col == "average_rating":
        range_color = [5, 9]
    elif color_col == "geek_rating":
        range_color = [5, 8]
    elif color_col == "users_rated_log":
        range_color = [1, 5]  # 10 to 100K

    fig = px.scatter(
        df_filtered,
        x=x_col,
        y=y_col,
        color=color_col,
        opacity=0.6,
        color_continuous_scale="Viridis",
        range_color=range_color,
    )

    # Format colorbar for users_rated_log to show actual counts
    if color_col == "users_rated_log":
        fig.update_coloraxes(
            colorbar=dict(
                title="Users Rated",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["10", "100", "1K", "10K", "100K"],
            )
        )
else:
    fig = px.scatter(
        df_filtered,
        x=x_col,
        y=y_col,
        opacity=0.5,
    )

fig.update_traces(
    customdata=custom_data,
    hovertemplate=hover_template,
)

fig.update_traces(marker=dict(size=3))

# Add highlights for selected games
if selected_games:
    matches = df_filtered[df_filtered["name"].isin(selected_games)]
    if len(matches) > 0:
        # Calculate centroid of selected points
        centroid_x = matches[x_col].mean()
        centroid_y = matches[y_col].mean()

        # Build labels with positions pointing outward from centroid
        labels = []
        for _, row in matches.iterrows():
            x_val = row[x_col]
            y_val = row[y_col]

            # Direction from centroid to this point
            dx = x_val - centroid_x
            dy = y_val - centroid_y
            dist = np.sqrt(dx**2 + dy**2)

            # Place label outward from centroid
            offset = 120
            if dist > 0.01:
                ax = offset * (dx / dist)
                ay = -offset * (dy / dist)  # Negative because screen y is inverted
            else:
                ax = offset
                ay = -offset

            labels.append({
                "name": row["name"],
                "x_data": x_val,
                "y_data": y_val,
                "ax": ax,
                "ay": ay,
                "width": len(row["name"]) * 7 + 16,
                "height": 22,
            })

        # Repulsion pass for overlapping labels
        for _ in range(100):
            moved = False
            for i, li in enumerate(labels):
                for j, lj in enumerate(labels):
                    if i >= j:
                        continue
                    # Label centers (using ax/ay as offsets)
                    li_cx = li["ax"]
                    li_cy = li["ay"]
                    lj_cx = lj["ax"]
                    lj_cy = lj["ay"]

                    dx = li_cx - lj_cx
                    dy = li_cy - lj_cy
                    min_dist_x = (li["width"] + lj["width"]) / 2 + 5
                    min_dist_y = (li["height"] + lj["height"]) / 2 + 3

                    if abs(dx) < min_dist_x and abs(dy) < min_dist_y:
                        dist = np.sqrt(dx**2 + dy**2) + 0.1
                        if dist < 5:
                            dx += np.random.uniform(-1, 1)
                            dy += np.random.uniform(-1, 1)
                            dist = np.sqrt(dx**2 + dy**2) + 0.1
                        push = 20
                        li["ax"] += push * dx / dist
                        li["ay"] += push * dy / dist
                        lj["ax"] -= push * dx / dist
                        lj["ay"] -= push * dy / dist
                        moved = True
            if not moved:
                break

        # Add annotations
        for label in labels:
            fig.add_annotation(
                x=label["x_data"],
                y=label["y_data"],
                text=f"<b>{label['name']}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="white",
                ax=label["ax"],
                ay=label["ay"],
                font=dict(size=11, color="black"),
                bgcolor="rgba(255, 255, 255, 0.9)",
                borderpad=4,
            )

        # Add highlighted points on top
        match_custom_data = matches[["name", "game_id", "year_published", "users_rated", "average_rating", "geek_rating", "predicted_complexity"]]
        fig.add_trace(
            go.Scatter(
                x=matches[x_col],
                y=matches[y_col],
                mode="markers",
                marker=dict(size=10, color="white", line=dict(width=2, color="black")),
                customdata=match_custom_data,
                hovertemplate=hover_template,
                name="Selected Games",
            )
        )

fig.update_layout(
    height=800,
    xaxis_title=f"{proj_prefix} 1",
    yaxis_title=f"{proj_prefix} 2",
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
)

st.plotly_chart(fig, use_container_width=True)
