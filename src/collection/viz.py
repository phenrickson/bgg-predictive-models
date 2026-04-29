"""Visualizations for collection-model artifacts.

Each public function does the data prep once and then renders either a
static plotnine figure (default; for notebooks) or an interactive plotly
figure (``interactive=True``; for Dash apps).

Functions take pre-computed DataFrames (e.g. output of
:meth:`CollectionModel.feature_importance`) and return a figure object.
No fitting or scoring happens here.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    facet_wrap,
    geom_col,
    geom_vline,
    ggplot,
    labs,
    scale_fill_distiller,
    theme,
    theme_minimal,
)


# Map feature-name prefix to display-group label. Extend as new feature
# families show up.
FEATURE_GROUPS: dict[str, str] = {
    "category_": "Categories",
    "mechanic_": "Mechanics",
    "designer_": "Designers",
    "artist_": "Artists",
    "publisher_": "Publishers",
    "family_": "Families",
    "player_count_": "Players",
    "missingindicator_": "Missingness",
}


# Singular display tags for each family prefix. Appears in front of the
# feature name as ``Tag: Value`` so a label is self-explanatory in mixed
# plots (where designers, publishers, etc. share the same axis).
_PREFIX_TAGS: dict[str, str] = {
    "category_": "Category",
    "mechanic_": "Mechanic",
    "designer_": "Designer",
    "artist_": "Artist",
    "publisher_": "Publisher",
    "family_": "Family",
    "missingindicator_": "Missing",
    "player_count_": "Players",
}


# --- Public API ---


def feature_group(feature_name: str) -> str:
    """Return the display-group label for a feature, or ``"Other"`` if no
    prefix matches."""
    for prefix, label in FEATURE_GROUPS.items():
        if feature_name.startswith(prefix):
            return label
    return "Other"


def tidy_feature_name(
    name: str, max_len: int = 40, include_tag: bool = True
) -> str:
    """Render a raw feature name for display.

    Strips the family prefix, swaps underscores for spaces, title-cases,
    and (when ``include_tag`` is ``True``) prepends a singular tag so the
    family stays visible:

    - ``designer_uwe_rosenberg`` -> ``Designer: Uwe Rosenberg``
    - ``publisher_fantasy_flight_games`` -> ``Publisher: Fantasy Flight Games``
    - ``player_count_6`` -> ``Players: 6``
    - ``missingindicator_min_age`` -> ``Missing: Min Age``

    Pass ``include_tag=False`` when the surrounding context already
    identifies the family (e.g. a single-group plot with ``Designers`` in
    the title) and the tag would just be visual noise.

    Truncates to ``max_len`` with an ellipsis.
    """
    tag: Optional[str] = None
    for p, t in _PREFIX_TAGS.items():
        if name.startswith(p):
            name = name[len(p):]
            tag = t
            break
    name = name.replace("_", " ").strip()
    body = name.title() if name else name
    if include_tag and tag is not None and body:
        body = f"{tag}: {body}"
    if len(body) > max_len:
        body = body[: max_len - 3] + "..."
    return body


def plot_feature_importance(
    importance_df: pd.DataFrame,
    group: Optional[str] = None,
    top_pos: int = 25,
    top_neg: int = 25,
    title: Optional[str] = None,
    interactive: bool = False,
    name_formatter: Optional[Callable[[str], str]] = tidy_feature_name,
) -> Union[ggplot, go.Figure]:
    """One diverging-bar feature-importance plot.

    Args:
        importance_df: Must have ``feature`` and ``value`` columns.
        group: If set (e.g. ``"Designers"``), filter to features in that
            group and strip the prefix from labels. ``None`` plots across
            all features.
        top_pos: Top N positive-value features to keep.
        top_neg: Top N negative-value features to keep.
        title: Plot title. Defaults to ``group`` (or ``"Feature Importance"``).
        interactive: If ``True``, return a plotly figure for Dash.
            Otherwise (default) return a plotnine figure for notebooks.
        name_formatter: Applied to each feature label before plotting.
            Defaults to :func:`tidy_feature_name`. Pass ``None`` for raw
            names (still with the group prefix stripped when ``group`` is set).
    """
    df = _prepare(
        importance_df,
        group=group,
        top_pos=top_pos,
        top_neg=top_neg,
        name_formatter=name_formatter,
    )
    plot_title = title or group or "Feature Importance"
    if interactive:
        return _render_plotly_bars(df, title=plot_title)
    return _render_plotnine_bars(df, title=plot_title)


def plot_feature_importance_grid(
    importance_df: pd.DataFrame,
    groups: Sequence[str],
    top_pos: int = 15,
    top_neg: int = 15,
    cols: int = 2,
    title: Optional[str] = None,
    interactive: bool = False,
    name_formatter: Optional[Callable[[str], str]] = tidy_feature_name,
) -> Union[ggplot, go.Figure]:
    """Faceted grid of feature-importance plots, one panel per group.

    Color scale is shared across all panels.

    Args:
        importance_df: Must have ``feature`` and ``value`` columns.
        groups: Display-group labels to facet over.
        top_pos: Top N positive features per panel.
        top_neg: Top N negative features per panel.
        cols: Number of columns in the grid (plotly only; plotnine uses
            ``facet_wrap`` and computes its own layout).
        title: Overall plot title.
        interactive: If ``True``, return a plotly figure for Dash.
            Otherwise return a plotnine figure.
        name_formatter: Applied to each feature label before plotting.
            Defaults to :func:`tidy_feature_name`. Pass ``None`` for raw names.
    """
    parts = []
    for g in groups:
        sub = _prepare(
            importance_df,
            group=g,
            top_pos=top_pos,
            top_neg=top_neg,
            name_formatter=name_formatter,
        )
        sub = sub.assign(group=g)
        parts.append(sub)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["feature", "value", "group"]
    )
    plot_title = title or "Feature Importance by Group"

    if interactive:
        return _render_plotly_grid(df, groups=groups, cols=cols, title=plot_title)
    return _render_plotnine_grid(df, title=plot_title)


# --- Shared data prep ---


def _prepare(
    importance_df: pd.DataFrame,
    group: Optional[str],
    top_pos: int,
    top_neg: int,
    name_formatter: Optional[Callable[[str], str]] = tidy_feature_name,
) -> pd.DataFrame:
    """Filter to ``group`` (if set), take top-N each side, sort descending,
    then apply ``name_formatter``. Returns a fresh frame with ``feature``
    and ``value`` columns ready to plot.

    When ``group`` is set the surrounding plot already identifies the
    family, so the default formatter is invoked with ``include_tag=False``
    to drop the redundant ``Family:`` prefix. Custom formatters are passed
    through unchanged.
    """
    df = importance_df.copy()
    if group is not None:
        mask = df["feature"].map(feature_group) == group
        df = df.loc[mask].copy()
    pos = df[df["value"] > 0].nlargest(top_pos, "value")
    neg = df[df["value"] < 0].nsmallest(top_neg, "value")
    out = (
        pd.concat([pos, neg], ignore_index=True)
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    if name_formatter is not None:
        if name_formatter is tidy_feature_name and group is not None:
            # Drop the "Family:" tag when the surrounding chart already
            # identifies the group — except where the bare body is
            # ambiguous on its own:
            #   - Players: "4" reads as numeric and Plotly will switch
            #     the y-axis to continuous.
            #   - Missingness: "Min Age" collides with the underlying
            #     feature of the same name in any cross-group context,
            #     and within-group it's clearer to keep the "Missing:"
            #     prefix so the chart says what it's measuring.
            include_tag = group in ("Players", "Missingness")
            out["feature"] = out["feature"].map(
                lambda f: tidy_feature_name(f, include_tag=include_tag)
            )
        else:
            out["feature"] = out["feature"].map(name_formatter)
    return out


# --- plotnine renderers (static, notebook-friendly) ---


def _render_plotnine_bars(df: pd.DataFrame, title: str) -> ggplot:
    # Preserve the sort order from _prepare (largest positive at top).
    feature_order = list(df["feature"])[::-1]  # ggplot draws bottom-up, so reverse
    df = df.assign(feature=pd.Categorical(df["feature"], categories=feature_order))
    cmax = float(df["value"].abs().max()) if len(df) else 1.0
    return (
        ggplot(df, aes(x="feature", y="value", fill="value"))
        + geom_col()
        + geom_vline(xintercept=0, color="grey", linetype="dotted")
        + coord_flip()
        + scale_fill_distiller(type="div", palette="RdBu", limits=(-cmax, cmax))
        + labs(title=title, x="", y="Effect on outcome", fill="Effect")
        + theme_minimal()
        + theme(panel_grid_major_y=element_blank())
    )


def _render_plotnine_grid(df: pd.DataFrame, title: str) -> ggplot:
    # Per-group ordered factor so each facet sorts correctly. We salt each
    # level with the group name to keep them unique across facets, then
    # strip the salt at draw time via scale_x_discrete(labels=...).
    df = df.copy()
    df["feature"] = df["group"].astype(str) + "::" + df["feature"].astype(str)
    feature_order = list(df.sort_values("value", ascending=True)["feature"].unique())
    df["feature"] = pd.Categorical(df["feature"], categories=feature_order)
    cmax = float(df["value"].abs().max()) if len(df) else 1.0
    from plotnine import scale_x_discrete

    def _drop_salt(labels):
        # plotnine passes the whole list of breaks; return the matching list.
        # ``GroupName::Actual Feature`` -> ``Actual Feature``
        return [str(lbl).split("::", 1)[-1] for lbl in labels]

    return (
        ggplot(df, aes(x="feature", y="value", fill="value"))
        + geom_col()
        + geom_vline(xintercept=0, color="grey", linetype="dotted")
        + coord_flip()
        + facet_wrap("~ group", scales="free_y")
        + scale_x_discrete(labels=_drop_salt)
        + scale_fill_distiller(type="div", palette="RdBu", limits=(-cmax, cmax))
        + labs(title=title, x="", y="Effect on outcome", fill="Effect")
        + theme_minimal()
        + theme(panel_grid_major_y=element_blank())
    )


# --- plotly renderers (interactive, Dash-friendly) ---


def _plotly_bar_trace(df: pd.DataFrame, cmax: float, show_colorbar: bool) -> go.Bar:
    return go.Bar(
        x=df["value"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["value"],
            colorscale="RdBu",
            cmid=0,
            cmin=-cmax,
            cmax=cmax,
            showscale=show_colorbar,
            colorbar=dict(title="Effect", thickness=12, len=0.6) if show_colorbar else None,
        ),
        hovertemplate="<b>%{y}</b><br>effect: %{x:.4f}<extra></extra>",
        showlegend=False,
    )


def _render_plotly_bars(df: pd.DataFrame, title: str) -> go.Figure:
    cmax = float(df["value"].abs().max()) if len(df) else 1.0
    fig = go.Figure(_plotly_bar_trace(df, cmax=cmax, show_colorbar=True))
    fig.update_layout(
        title=title,
        xaxis_title="Effect on outcome",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=max(400, 22 * len(df) + 100),
        margin=dict(l=180, r=60, t=60, b=60),
    )
    return fig


def _render_plotly_grid(
    df: pd.DataFrame, groups: Sequence[str], cols: int, title: str
) -> go.Figure:
    cmax = float(df["value"].abs().max()) if len(df) else 1.0
    rows = (len(groups) + cols - 1) // cols
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=list(groups),
        horizontal_spacing=0.18,
        vertical_spacing=0.08,
    )
    for i, g in enumerate(groups):
        row = i // cols + 1
        col = i % cols + 1
        sub = df[df["group"] == g]
        fig.add_trace(
            _plotly_bar_trace(sub, cmax=cmax, show_colorbar=(i == 0)),
            row=row,
            col=col,
        )
        fig.update_yaxes(autorange="reversed", row=row, col=col)
        fig.update_xaxes(title_text="Effect", row=row, col=col)
    fig.update_layout(
        title=title,
        height=max(500, 350 * rows),
        margin=dict(l=180, r=60, t=80, b=60),
    )
    return fig
