"""Visualizations for collection-model artifacts.

Each public function does the data prep once and then renders either a
static plotnine figure (default; for notebooks) or an interactive plotly
figure (``interactive=True``; for Dash apps).

Functions take pre-computed DataFrames (e.g. output of
:meth:`CollectionModel.feature_importance`) and return a figure object.
No fitting or scoring happens here.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    facet_wrap,
    geom_area,
    geom_col,
    geom_histogram,
    geom_vline,
    ggplot,
    labs,
    scale_fill_distiller,
    scale_x_continuous,
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


def extract_finalized_importance(
    pipeline,
    train_sample: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Return feature importance for a fitted Pipeline.

    Pulls ``feature_importances_`` (tree models) or ``coef_`` (linear
    models) from ``pipeline.named_steps['model']``. Recovers
    post-preprocessing feature names by transforming a small sample of
    canonical training data through ``pipeline.named_steps['preprocessor']``
    — sklearn's ``get_feature_names_out`` is unreliable on this stack.

    Returns a DataFrame with columns ``feature``, ``value``, ``abs_value``,
    sorted by ``abs_value`` descending. Returns ``None`` if the model
    exposes neither attribute.
    """
    import numpy as np

    model_step = pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        values = np.asarray(model_step.feature_importances_)
    elif hasattr(model_step, "coef_"):
        values = np.asarray(model_step.coef_).ravel()
    else:
        return None

    names: Optional[list[str]] = None
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(train_sample.head(5))
        if hasattr(transformed, "columns"):
            names = list(transformed.columns)
    except Exception:
        names = None
    if names is None:
        try:
            names = list(pipeline[:-1].get_feature_names_out())
        except Exception:
            names = None
    if names is None or len(names) != len(values):
        names = [f"f{i}" for i in range(len(values))]

    out = pd.DataFrame({"feature": names, "value": values})
    out["abs_value"] = out["value"].abs()
    return out.sort_values("abs_value", ascending=False).reset_index(drop=True)


def metrics_table(registration: dict) -> pd.DataFrame:
    """One-row-per-split metrics frame from a registration.json.

    Splits surfaced (in this order): ``val``, ``oof``, ``test``. Missing
    splits are dropped. Numeric metrics are kept as-is so downstream
    formatters can apply their own rounding.
    """
    rows: list[dict[str, Any]] = []
    splits = {
        "val": registration.get("val_metrics") or {},
        "oof": (registration.get("oof_metrics") or {}).get("overall") or {},
        "test": registration.get("metrics") or {},
    }
    for split_name, metrics in splits.items():
        if not metrics:
            continue
        row: dict[str, Any] = {"split": split_name}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["split"])
    return pd.DataFrame(rows)


def plot_separation(
    predictions,
    title: Optional[str] = None,
) -> go.Figure:
    """Predicted-proba area chart with true-positive vertical lines.

    Sorts predictions by ``proba`` descending, plots ``proba`` vs rank as
    an area, and overlays a thin vertical line at every rank where
    ``label`` is truthy.
    """
    import polars as pl

    if predictions.height == 0 or "proba" not in predictions.columns:
        return go.Figure(layout={"title": title or "Separation"})

    sorted_preds = predictions.sort("proba", descending=True).with_row_index(
        "rank", offset=1
    )
    pdf = sorted_preds.select(["rank", "proba", "label"]).to_pandas()
    true_ranks = pdf.loc[pdf["label"].astype(bool), "rank"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pdf["rank"],
            y=pdf["proba"],
            mode="lines",
            fill="tozeroy",
            line={"color": "#444444", "width": 1},
            fillcolor="rgba(80,80,80,0.25)",
            hovertemplate="rank=%{x}<br>proba=%{y:.4f}<extra></extra>",
            showlegend=False,
        )
    )
    shapes = [
        {
            "type": "line",
            "x0": x,
            "x1": x,
            "y0": 0,
            "y1": 1,
            "yref": "y domain",
            "line": {"color": "#4fc3f7", "width": 1},
            "opacity": 0.6,
        }
        for x in true_ranks
    ]
    fig.update_layout(
        title=title or "Separation",
        shapes=shapes,
        xaxis_title="rank (proba descending)",
        yaxis_title="proba",
        height=240,
        margin={"t": 40, "b": 40, "l": 50, "r": 12},
    )
    return fig


def top_n_by_year_table(predictions, top_n: int = 15, min_year: int | None = 2015):
    """Pivot predictions into rank × year, top-N per year.

    Each column is a year (as a string for stable header names); each
    row is rank 1..top_n. Cells contain the game ``name``.

    Args:
        predictions: Must have ``proba``, ``year_published``, ``name``.
        top_n: Number of rows per year to keep.
        min_year: Drop rows with ``year_published`` below this. Pass
            ``None`` to disable. Defaults to 2015 — older years tend to
            be dominated by very high-proba "classics" that crowd out
            the recent picks people actually want to see.
    """
    import polars as pl

    if predictions.height == 0 or "year_published" not in predictions.columns:
        return pl.DataFrame()

    view = predictions.with_columns(pl.col("year_published").cast(pl.Int64))
    if min_year is not None:
        view = view.filter(pl.col("year_published") >= min_year)
    if view.height == 0:
        return pl.DataFrame()
    view = view.with_columns(
        pl.col("proba")
        .rank(method="ordinal", descending=True)
        .over("year_published")
        .alias("_rank")
    ).filter(pl.col("_rank") <= top_n)

    if view.height == 0:
        return pl.DataFrame()

    pivot = (
        view.pivot(values="name", index="_rank", on="year_published")
        .sort("_rank")
        .rename({"_rank": "rank"})
    )
    year_cols = sorted(int(y) for y in view["year_published"].unique().to_list())
    ordered = ["rank"] + [str(y) for y in year_cols]
    return pivot.select([c for c in ordered if c in pivot.columns])


_PREDICTION_DISPLAY_COLS = [
    "proba",
    "predicted_prob",
    "label",
    "predicted_label",
    "name",
    "year_published",
    "min_players",
    "max_players",
    "min_playtime",
    "max_playtime",
    "average_rating",
    "average_weight",
    "users_rated",
    "game_id",
]


def predictions_datatable(
    predictions,
    games,
    top_n: int = 500,
    min_users_rated: int = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Sortable predictions table for embedding in the report.

    Returns a pandas DataFrame; the qmd wraps it with `itables.show(...)`.
    By default, surfaces a curated set of columns (proba/label, name,
    year, player counts, playtime, ratings) and drops list-typed columns
    that render badly in tables.
    """
    import polars as pl

    view = predictions
    if min_users_rated > 0 and "users_rated" in view.columns:
        view = view.filter(pl.col("users_rated") >= min_users_rated)
    if "proba" in view.columns:
        view = view.sort("proba", descending=True)
    elif "predicted_prob" in view.columns:
        view = view.sort("predicted_prob", descending=True)
    view = view.head(top_n)

    if games is not None and games.height > 0 and "game_id" in games.columns:
        meta_cols = [
            c for c in games.columns if c == "game_id" or c not in view.columns
        ]
        view = view.join(games.select(meta_cols), on="game_id", how="left")

    # Drop list-typed and binary-blob columns; itables can't sort them
    # and they make the table unreadable.
    drop_cols = {
        "categories", "mechanics", "designers", "artists", "publishers",
        "families", "description", "thumbnail", "image",
        "load_timestamp", "last_updated",
    }
    keep = columns or [c for c in _PREDICTION_DISPLAY_COLS if c in view.columns]
    if not keep:
        keep = [c for c in view.columns if c not in drop_cols]
    return view.select([c for c in keep if c in view.columns]).to_pandas()


def plot_collection_by_year(collection, games) -> go.Figure:
    """Histogram of ``year_published`` for owned games."""
    import polars as pl

    if collection.height == 0:
        return go.Figure(layout={"title": "Games by year"})
    owned = (
        collection.filter(pl.col("owned") == True)
        .select("game_id")
        .join(games.select(["game_id", "year_published"]), on="game_id", how="inner")
    )
    if owned.height == 0:
        return go.Figure(layout={"title": "Games by year"})
    counts = owned.group_by("year_published").len().sort("year_published")
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts["year_published"].to_list(),
                y=counts["len"].to_list(),
            )
        ]
    )
    fig.update_layout(
        title="Games by year",
        xaxis_title="year_published",
        yaxis_title="count",
        height=320,
    )
    return fig


def plot_collection_by_category(collection, games, top_n: int = 15) -> go.Figure:
    """Top-N feature flags in the user's owned games, faceted by family.

    Aggregates dummy columns matching known feature-group prefixes
    (categories, mechanics, designers, etc.) over the joined collection,
    then plots the most-frequent within each group.
    """
    import polars as pl

    if collection.height == 0:
        return go.Figure(layout={"title": "Types of games"})
    owned = collection.filter(pl.col("owned") == True).select("game_id")
    joined = owned.join(games, on="game_id", how="inner")
    if joined.height == 0:
        return go.Figure(layout={"title": "Types of games"})

    list_groups = {
        "categories": "Categories",
        "mechanics": "Mechanics",
        "designers": "Designers",
        "artists": "Artists",
        "publishers": "Publishers",
        "families": "Families",
    }
    present_cols = [c for c in list_groups if c in joined.columns]
    rows: list[dict[str, Any]] = []
    for col in present_cols:
        try:
            exploded = (
                joined.select(pl.col(col))
                .explode(col)
                .drop_nulls()
                .group_by(col)
                .len()
                .sort("len", descending=True)
                .head(top_n)
            )
        except Exception:
            continue
        for record in exploded.to_dicts():
            value = record[col]
            count = record["len"]
            if value in (None, ""):
                continue
            rows.append(
                {"feature": str(value), "group": list_groups[col], "count": int(count)}
            )

    # Fallback: legacy dummy columns (category_*, mechanic_*, etc.)
    if not rows:
        for col in joined.columns:
            group = feature_group(col)
            if group == "Other":
                continue
            try:
                total = int(joined.select(pl.col(col).sum()).item())
            except Exception:
                continue
            if total <= 0:
                continue
            rows.append(
                {
                    "feature": tidy_feature_name(col, include_tag=False),
                    "group": group,
                    "count": total,
                }
            )

    if not rows:
        return go.Figure(layout={"title": "Types of games"})

    groups_order = [
        list_groups[c] for c in present_cols if list_groups[c] in {r["group"] for r in rows}
    ] or sorted({r["group"] for r in rows})

    n_groups = len(groups_order)
    cols_per_row = 2
    rows_grid = (n_groups + cols_per_row - 1) // cols_per_row
    fig = make_subplots(
        rows=rows_grid,
        cols=cols_per_row,
        subplot_titles=groups_order,
        horizontal_spacing=0.3,
        vertical_spacing=0.08,
    )
    for i, group in enumerate(groups_order):
        sub = (
            pd.DataFrame([r for r in rows if r["group"] == group])
            .sort_values("count", ascending=True)
        )
        r = i // cols_per_row + 1
        c = i % cols_per_row + 1
        fig.add_trace(
            go.Bar(
                x=sub["count"],
                y=sub["feature"],
                orientation="h",
                showlegend=False,
                marker_color="#4fc3f7",
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        title="Types of games",
        height=max(360, 220 * rows_grid),
        margin={"l": 80, "r": 20, "t": 80, "b": 40},
    )
    return fig


_COLLECTION_DISPLAY_COLS = [
    "game_name",
    "name",
    "year_published",
    "user_rating",
    "owned",
    "previously_owned",
    "want",
    "wishlist",
    "preordered",
    "for_trade",
    "min_players",
    "max_players",
    "min_playtime",
    "max_playtime",
    "average_rating",
    "average_weight",
    "users_rated",
    "game_id",
]


def collection_datatable(collection, games) -> pd.DataFrame:
    """Sortable table of a user's collection.

    Joins in game metadata when available. Returns a pandas DataFrame
    with a curated set of columns; the qmd wraps with `itables.show`.
    Drops list-typed and blob columns that render poorly in tables.
    """
    import polars as pl

    if collection.height == 0:
        return pd.DataFrame()
    view = collection
    if games is not None and games.height > 0 and "game_id" in games.columns:
        meta_cols = [
            c for c in games.columns if c == "game_id" or c not in view.columns
        ]
        view = view.join(games.select(meta_cols), on="game_id", how="left")

    # Show only owned games by default — that's what users mean by
    # "their collection" in this report.
    if "owned" in view.columns:
        view = view.filter(pl.col("owned") == True)

    if "user_rating" in view.columns:
        view = view.sort("user_rating", descending=True, nulls_last=True)

    keep = [c for c in _COLLECTION_DISPLAY_COLS if c in view.columns]
    if not keep:
        keep = view.columns
    return view.select(keep).to_pandas()


def plot_partial_effects_by_group(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
) -> dict[str, go.Figure]:
    """Build one feature-importance plot per known group.

    Returns a dict keyed by group label. Empty groups are omitted.
    """
    if feature_importance is None or len(feature_importance) == 0:
        return {}
    groups = sorted(
        {feature_group(name) for name in feature_importance["feature"].tolist()}
    )
    out: dict[str, go.Figure] = {}
    for group in groups:
        if group == "Other":
            continue
        try:
            fig = plot_feature_importance(
                feature_importance,
                group=group,
                top_pos=top_n,
                top_neg=top_n,
                interactive=True,
                title=group,
            )
        except Exception:
            continue
        out[group] = fig
    return out


# --- Static (plotnine) variants — for embedding in Quarto reports ---
#
# These exist alongside the plotly versions because plotly figures bundle
# 4–5 MB of JS per chart, blowing up the embedded-resources HTML. plotnine
# renders to PNG which is kilobytes. Use the static variants in the qmd
# template; keep plotly versions for Streamlit/Dash.


def plot_separation_static(predictions, title: Optional[str] = None) -> ggplot:
    """plotnine version of `plot_separation`.

    Sorts predictions by ``proba`` descending, plots ``proba`` as an area
    chart against rank, and overlays a vertical line at every rank
    where ``label`` is truthy.
    """
    import polars as pl

    if predictions.height == 0 or "proba" not in predictions.columns:
        return (
            ggplot(pd.DataFrame({"rank": [], "proba": []}), aes("rank", "proba"))
            + labs(title=title or "Separation")
            + theme_minimal()
        )

    sorted_preds = predictions.sort("proba", descending=True).with_row_index(
        "rank", offset=1
    )
    pdf = sorted_preds.select(["rank", "proba", "label"]).to_pandas()
    pdf["rank"] = pdf["rank"].astype(int)
    true_ranks = pdf.loc[pdf["label"].astype(bool), "rank"].tolist()

    plot = (
        ggplot(pdf, aes("rank", "proba"))
        + geom_area(fill="#888888", alpha=0.4)
        + theme_minimal()
        + labs(
            title=title or "Separation",
            x="rank (proba descending)",
            y="proba",
        )
        + theme(figure_size=(8, 2.2))
    )
    if true_ranks:
        plot = plot + geom_vline(
            xintercept=true_ranks, color="#1976d2", alpha=0.6, size=0.3
        )
    return plot


def plot_collection_by_year_static(collection, games) -> ggplot:
    """plotnine version of `plot_collection_by_year`."""
    import polars as pl

    if collection.height == 0:
        return (
            ggplot(pd.DataFrame({"year_published": [], "n": []}))
            + labs(title="Games by year")
            + theme_minimal()
        )
    owned = (
        collection.filter(pl.col("owned") == True)
        .select("game_id")
        .join(games.select(["game_id", "year_published"]), on="game_id", how="inner")
    )
    if owned.height == 0:
        return (
            ggplot(pd.DataFrame({"year_published": [], "n": []}))
            + labs(title="Games by year")
            + theme_minimal()
        )
    counts = owned.group_by("year_published").len().sort("year_published")
    pdf = counts.to_pandas().rename(columns={"len": "n"})
    pdf["year_published"] = pdf["year_published"].astype(int)
    return (
        ggplot(pdf, aes("year_published", "n"))
        + geom_col(fill="#4fc3f7")
        + labs(title="Games by year", x="year published", y="count")
        + theme_minimal()
        + theme(figure_size=(9, 3.5))
    )


def plot_collection_by_category_static(collection, games, top_n: int = 12) -> ggplot:
    """plotnine version of `plot_collection_by_category`.

    Renders a single faceted figure (one panel per family) instead of a
    plotly subplots grid. Reuses the same data-shaping logic as the
    plotly version.
    """
    import polars as pl

    if collection.height == 0 or games is None or games.height == 0:
        return (
            ggplot(pd.DataFrame({"feature": [], "count": [], "group": []}))
            + labs(title="Types of games")
            + theme_minimal()
        )

    owned_ids = collection.filter(pl.col("owned") == True).select("game_id")
    joined = owned_ids.join(games, on="game_id", how="inner")
    if joined.height == 0:
        return (
            ggplot(pd.DataFrame({"feature": [], "count": [], "group": []}))
            + labs(title="Types of games")
            + theme_minimal()
        )

    list_groups = {
        "categories": "Categories",
        "mechanics": "Mechanics",
        "designers": "Designers",
        "artists": "Artists",
        "publishers": "Publishers",
        "families": "Families",
    }
    present_cols = [c for c in list_groups if c in joined.columns]
    rows: list[dict[str, Any]] = []
    for col in present_cols:
        try:
            exploded = (
                joined.select(pl.col(col))
                .explode(col)
                .drop_nulls()
                .group_by(col)
                .len()
                .sort("len", descending=True)
                .head(top_n)
            )
        except Exception:
            continue
        for record in exploded.to_dicts():
            value = record[col]
            count = record["len"]
            if value in (None, ""):
                continue
            rows.append(
                {"feature": str(value), "group": list_groups[col], "count": int(count)}
            )

    if not rows:
        return (
            ggplot(pd.DataFrame({"feature": [], "count": [], "group": []}))
            + labs(title="Types of games")
            + theme_minimal()
        )

    pdf = pd.DataFrame(rows)
    # Order features within each facet by count so the bars sort cleanly.
    pdf["feature"] = pd.Categorical(
        pdf["feature"],
        categories=pdf.sort_values(["group", "count"])["feature"].tolist(),
        ordered=True,
    )
    n_groups = pdf["group"].nunique()
    return (
        ggplot(pdf, aes("feature", "count"))
        + geom_col(fill="#4fc3f7")
        + coord_flip()
        + facet_wrap("group", scales="free", ncol=2)
        + labs(title="Types of games", x="", y="count")
        + theme_minimal()
        + theme(
            figure_size=(8, 2.6 * ((n_groups + 1) // 2)),
            strip_text=element_text(weight="bold"),
        )
    )
