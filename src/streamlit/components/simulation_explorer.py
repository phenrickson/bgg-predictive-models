"""Helper for calling the scoring service samples endpoint and plotting posteriors."""

import logging
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAMES = {
    "complexity": "complexity-v2026",
    "rating": "rating-v2026",
    "users_rated": "users_rated-v2026",
    "geek_rating": "geek_rating-v2026",
}

OUTCOME_COLORS = {
    "complexity": "steelblue",
    "rating": "forestgreen",
    "users_rated": "darkorange",
    "geek_rating": "purple",
}

OUTCOME_LABELS = {
    "complexity": "Complexity",
    "rating": "Rating",
    "users_rated": "Users Rated",
    "geek_rating": "Geek Rating",
}


def call_simulate_samples(
    game_ids: List[int],
    service_url: str,
    n_samples: int = 500,
    model_names: Optional[Dict[str, str]] = None,
) -> dict:
    """Call the /simulate_game_samples endpoint and return the JSON response."""
    models = model_names or DEFAULT_MODEL_NAMES
    payload = {
        "game_ids": game_ids,
        "complexity_model_name": models["complexity"],
        "rating_model_name": models["rating"],
        "users_rated_model_name": models["users_rated"],
        "geek_rating_model_name": models["geek_rating"],
        "n_samples": n_samples,
    }
    url = f"{service_url.rstrip('/')}/simulate_game_samples"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def plot_posterior_distributions(game: dict) -> go.Figure:
    """Create a 2x2 subplot with posterior histograms for one game.

    Parameters
    ----------
    game : dict
        A single element from SimulateGameSamplesResponse.games
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Complexity", "Rating", "Users Rated", "Geek Rating"],
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    outcomes = [
        ("complexity", 1, 1),
        ("rating", 1, 2),
        ("users_rated", 2, 1),
        ("geek_rating", 2, 2),
    ]

    for outcome, row, col in outcomes:
        samples = np.array(game[f"{outcome}_samples"])
        point = game[f"{outcome}_point"]
        actual = game.get(f"actual_{outcome}")
        color = OUTCOME_COLORS[outcome]

        # For users_rated, convert from log scale to count scale
        if outcome == "users_rated":
            samples = np.expm1(samples)
            point = np.expm1(point)
            if actual is not None and actual != 0:
                actual = np.expm1(actual)

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=samples,
                nbinsx=50,
                histnorm="probability density",
                marker_color=color,
                opacity=0.7,
                showlegend=False,
                hovertemplate="%{x:.2f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # 90% credible interval shading
        low = float(np.percentile(samples, 5))
        high = float(np.percentile(samples, 95))
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color, opacity=0.15, line_width=0,
            row=row, col=col,
        )

        # Point estimate line
        fig.add_vline(
            x=point, line_dash="dash", line_color="darkblue", line_width=2,
            annotation_text=f"Point: {point:.2f}" if outcome != "users_rated" else f"Point: {point:,.0f}",
            annotation_position="top right",
            annotation_font_size=10,
            row=row, col=col,
        )

        # Actual value line
        if actual is not None and actual != 0:
            fig.add_vline(
                x=actual, line_dash="solid", line_color="red", line_width=2,
                annotation_text=f"Actual: {actual:.2f}" if outcome != "users_rated" else f"Actual: {actual:,.0f}",
                annotation_position="top left",
                annotation_font_size=10,
                row=row, col=col,
            )

        # Axis labels
        fig.update_xaxes(title_text=OUTCOME_LABELS[outcome], row=row, col=col)
        fig.update_yaxes(title_text="Density", row=row, col=col)

        # Log x-axis for users_rated
        if outcome == "users_rated":
            fig.update_xaxes(type="log", row=row, col=col)

    fig.update_layout(
        title=f"{game['game_name']} (ID: {game['game_id']})",
        height=600,
        showlegend=False,
    )

    return fig
