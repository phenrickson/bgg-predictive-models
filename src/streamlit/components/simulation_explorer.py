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

# Sequential blues, light to dark, with geek_rating anchoring the dark end
# to signal it's the composite built on top of the upstream outcomes.
OUTCOME_COLORS = {
    "complexity": "#9ecae1",
    "rating": "#6baed6",
    "users_rated": "#3182bd",
    "geek_rating": "#08519c",
}

# Desaturated same-hue counterparts used for negative contributions.
OUTCOME_NEG_COLORS = {
    "complexity": "#b8c4cc",
    "rating": "#8fa1ad",
    "users_rated": "#6b8090",
    "geek_rating": "#4a5a6b",
}

OUTCOME_LABELS = {
    "complexity": "Complexity",
    "rating": "Rating",
    "users_rated": "Users Rated (log scale)",
    "geek_rating": "Geek Rating",
}

_FEATURE_PREFIX_LABELS = {
    "designer_": "Designer: ",
    "artist_": "Artist: ",
    "publisher_": "Publisher: ",
    "category_": "Category: ",
    "mechanic_": "Mechanic: ",
    "family_": "Family: ",
    "emb_": "Embedding ",
    "missingindicator_": "Missing: ",
    "player_count_": "Player count ",
    "predicted_": "Predicted ",
    "year_published": "Year published",
    "min_age": "Min age",
    "min_playtime": "Min playtime",
    "max_playtime": "Max playtime",
}


_TRAILING_NOISE = ("_log", "_transformed", "_count")


def _prettify_feature(name: str) -> str:
    """Turn raw feature names into readable labels.

    Keeps the category prefix (Designer/Publisher/etc.), title-cases the
    remainder, and strips redundant trailing suffixes like `_log`.
    """
    for prefix, replacement in _FEATURE_PREFIX_LABELS.items():
        if name.startswith(prefix):
            rest = name[len(prefix):]
            for suffix in _TRAILING_NOISE:
                if rest.endswith(suffix):
                    rest = rest[: -len(suffix)]
                    break
            rest = rest.replace("_", " ").strip()
            if rest:
                return f"{replacement}{rest.title()}"
            return replacement.rstrip(": ").rstrip()
    cleaned = name
    for suffix in _TRAILING_NOISE:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned.replace("_", " ").title()


def _format_feature_value(name: str, raw_value, value) -> Optional[str]:
    """Format the feature's value for display next to the feature name."""
    v = raw_value if raw_value is not None else value
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)

    # Binary indicators: show only present/absent rather than 1.0/0.0
    binary_prefixes = ("designer_", "artist_", "publisher_", "category_",
                       "mechanic_", "family_", "missingindicator_", "player_count_")
    if any(name.startswith(p) for p in binary_prefixes) and fv in (0.0, 1.0):
        return "yes" if fv == 1.0 else "no"

    if name.startswith("emb_"):
        return f"{fv:.2f}"
    if abs(fv) >= 1000:
        return f"{fv:,.0f}"
    if abs(fv) >= 10:
        return f"{fv:.1f}"
    return f"{fv:.2f}"


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
    """Create a 2x2 bar chart showing feature contributions per outcome.

    Sign is encoded by direction of the bar; color stays in the outcome's hue
    family (muted for negative) to avoid red=bad framing. Feature labels are
    cleaned up and show the feature value alongside the name.
    """
    outcomes = [
        ("complexity", 1, 1),
        ("rating", 1, 2),
        ("users_rated", 2, 1),
        ("geek_rating", 2, 2),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Complexity", "Rating", "Users Rated", "Geek Rating"],
        horizontal_spacing=0.28,
        vertical_spacing=0.18,
    )

    for outcome, row, col in outcomes:
        exp = explanations.get(outcome)
        if not exp:
            continue

        contributions = sorted(
            exp["contributions"],
            key=lambda c: abs(c.get("contribution", 0)),
            reverse=True,
        )

        pos_color = OUTCOME_COLORS[outcome]
        neg_color = OUTCOME_NEG_COLORS[outcome]

        labels = []
        values = []
        colors = []
        raw_values_display = []
        raw_names = []
        for c in contributions:
            name = c["feature"]
            contribution = c["contribution"]
            value_str = _format_feature_value(name, c.get("raw_value"), c.get("value"))
            pretty = _prettify_feature(name)
            label = f"{pretty} = {value_str}" if value_str is not None else pretty
            labels.append(label)
            values.append(contribution)
            colors.append(pos_color if contribution >= 0 else neg_color)
            raw_values_display.append(value_str if value_str is not None else "")
            raw_names.append(name)

        # Reverse so largest |contribution| is on top
        labels = labels[::-1]
        values = values[::-1]
        colors = colors[::-1]
        raw_values_display = raw_values_display[::-1]
        raw_names = raw_names[::-1]

        customdata = list(zip(raw_names, raw_values_display))
        fig.add_trace(
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=colors,
                showlegend=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "value: %{customdata[1]}<br>"
                    "contribution: %{x:+.2f}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

        fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)", line_width=0.5,
                      row=row, col=col)
        fig.update_xaxes(title_text="Contribution", row=row, col=col)
        fig.update_yaxes(tickfont=dict(size=10), automargin=True, row=row, col=col)

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
        height=1000,
        showlegend=False,
        margin=dict(t=100, l=60, r=40, b=60),
        bargap=0.25,
    )

    return fig
