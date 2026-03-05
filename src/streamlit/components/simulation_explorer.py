"""Helper for calling scoring service endpoints and plotting results."""

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
    "users_rated": "Users Rated (log scale)",
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


def _format_value(outcome: str, value: float) -> str:
    """Format a value for display based on outcome type."""
    if outcome == "users_rated":
        return f"{np.expm1(value):,.0f}"
    return f"{value:.2f}"


def plot_posterior_distributions(game: dict) -> go.Figure:
    """Create a 2x2 subplot with posterior histograms for one game."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Complexity", "Rating", "Users Rated", "Geek Rating"],
        horizontal_spacing=0.12,
        vertical_spacing=0.28,
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

        # For users_rated: keep in log scale, convert actual to match
        if outcome == "users_rated" and actual is not None and actual > 0:
            actual = float(np.log1p(actual))

        # 90% CI bounds
        low = float(np.percentile(samples, 5))
        high = float(np.percentile(samples, 95))

        # Histogram — plain counts, no density
        fig.add_trace(
            go.Histogram(
                x=samples,
                nbinsx=50,
                marker_color=color,
                opacity=0.7,
                showlegend=False,
                hovertemplate="%{x:.2f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # 90% CI shading
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color, opacity=0.12, line_width=0,
            row=row, col=col,
        )

        # Point estimate
        fig.add_vline(
            x=point, line_dash="dash", line_color="white", line_width=1.5,
            row=row, col=col,
        )

        # Actual value
        if actual is not None and actual != 0:
            fig.add_vline(
                x=actual, line_dash="solid", line_color="red", line_width=1.5,
                row=row, col=col,
            )

        # Build subtitle with point/actual/CI info
        parts = [f"Point: {_format_value(outcome, point)}"]
        if actual is not None and actual != 0:
            parts.append(f"Actual: {_format_value(outcome, actual)}")
        parts.append(f"90% CI: [{_format_value(outcome, low)}, {_format_value(outcome, high)}]")

        fig.update_xaxes(title_text=OUTCOME_LABELS[outcome], row=row, col=col)
        fig.update_yaxes(title_text="", showticklabels=False, row=row, col=col)

    # Update subplot titles to include summary stats
    for i, (outcome, row, col) in enumerate(outcomes):
        samples = np.array(game[f"{outcome}_samples"])
        point = game[f"{outcome}_point"]
        actual = game.get(f"actual_{outcome}")
        if outcome == "users_rated" and actual is not None and actual > 0:
            actual = float(np.log1p(actual))

        low = float(np.percentile(samples, 5))
        high = float(np.percentile(samples, 95))

        label = OUTCOME_LABELS[outcome].replace(" (log scale)", "")
        pred_str = f"Predicted: {_format_value(outcome, point)}"
        actual_str = f" | Actual: {_format_value(outcome, actual)}" if actual and actual != 0 else ""
        ci_str = f"90% CI: [{_format_value(outcome, low)}, {_format_value(outcome, high)}]"

        fig.layout.annotations[i].text = (
            f"<b>{label}</b><br>"
            f"<span style='font-size:11px'>{pred_str}{actual_str}<br>{ci_str}</span>"
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{game['game_name']}</b> (ID: {game['game_id']})",
            y=0.98,
            yanchor="top",
        ),
        height=750,
        showlegend=False,
        margin=dict(t=120),
    )

    return fig


def call_explain_game(
    game_id: int,
    service_url: str,
    top_n: int = 15,
    model_names: Optional[Dict[str, str]] = None,
) -> dict:
    """Call the /explain_game endpoint and return the JSON response."""
    models = model_names or DEFAULT_MODEL_NAMES
    payload = {
        "game_id": game_id,
        "complexity_model_name": models["complexity"],
        "rating_model_name": models["rating"],
        "users_rated_model_name": models["users_rated"],
        "geek_rating_model_name": models["geek_rating"],
        "top_n": top_n,
    }
    url = f"{service_url.rstrip('/')}/explain_game"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def plot_explanation(explanations: dict, game_name: str, game_id: int) -> go.Figure:
    """Create a 2x2 waterfall/bar chart showing feature contributions per outcome."""
    outcomes = [
        ("complexity", 1, 1),
        ("rating", 1, 2),
        ("users_rated", 2, 1),
        ("geek_rating", 2, 2),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Complexity", "Rating", "Users Rated", "Geek Rating"],
        horizontal_spacing=0.12,
        vertical_spacing=0.28,
    )

    for outcome, row, col in outcomes:
        exp = explanations.get(outcome)
        if not exp:
            continue

        contributions = exp["contributions"]
        prediction = exp["prediction"]
        intercept = exp["intercept"]

        # Sort by absolute contribution descending
        contributions = sorted(
            contributions, key=lambda c: abs(c.get("contribution", 0)), reverse=True
        )

        features = [c["feature"] for c in contributions]
        values = [c["contribution"] for c in contributions]
        colors = ["#e74c3c" if v < 0 else OUTCOME_COLORS[outcome] for v in values]

        # Reverse for bottom-to-top display (largest at top)
        features = features[::-1]
        values = values[::-1]
        colors = colors[::-1]

        fig.add_trace(
            go.Bar(
                y=features,
                x=values,
                orientation="h",
                marker_color=colors,
                showlegend=False,
                hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # Zero line
        fig.add_vline(x=0, line_color="white", line_width=0.5, row=row, col=col)

        fig.update_xaxes(title_text="Contribution", row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=9), row=row, col=col)

    # Update subplot titles with prediction values
    for i, (outcome, row, col) in enumerate(outcomes):
        exp = explanations.get(outcome)
        if not exp:
            continue
        pred = exp["prediction"]
        label = OUTCOME_LABELS[outcome].replace(" (log scale)", "")
        if outcome == "users_rated":
            pred_display = f"{np.expm1(pred):,.0f}"
        else:
            pred_display = f"{pred:.2f}"
        fig.layout.annotations[i].text = (
            f"<b>{label}</b><br>"
            f"<span style='font-size:11px'>Prediction: {pred_display}</span>"
        )

    fig.update_layout(
        title=f"<b>{game_name}</b> (ID: {game_id}) — Feature Contributions",
        height=700,
        showlegend=False,
        margin=dict(t=100, l=200),
    )

    return fig
