"""Table builders for the collection report.

Each function returns a pandas DataFrame ready for `itables_show`, except
`build_topn_by_year_html` which returns a raw HTML string for direct
emission inside an `output: asis` chunk.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from src.reports.format import (
    bgg_link,
    format_range,
    img_tag,
    maybe,
    short_dt,
    short_sha,
    truncate,
    truncate_name,
)


def _safe_col(pdf: pd.DataFrame, col: str, default=None) -> list:
    return pdf[col].tolist() if col in pdf.columns else [default] * len(pdf)


_STATUS_PRIORITY = ["Own", "Preordered", "Wishlist", "Want", "Prev. Owned"]


def _sort_player_counts(raw) -> str:
    """Render a comma-separated player-count string in ascending numeric
    order. The warehouse stores them vote-ordered (e.g. "4, 3"); we want
    "3, 4". Non-numeric buckets like "4+" sort to the end, order kept."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    numeric = sorted((p for p in parts if p.isdigit()), key=int)
    other = [p for p in parts if not p.isdigit()]
    return ", ".join(numeric + other)


def format_collection_table(
    collection: pl.DataFrame, games: pl.DataFrame
) -> pd.DataFrame:
    """Tagged collection table: Game / Status / Your rating / Players /
    Playtime / Complexity.

    Includes everything the user has marked — owned, wishlist, want,
    preordered, previously owned — sorted with owned games first then
    by rating. Year is encoded inside the Game link, so no separate
    column.
    """
    if collection.height == 0:
        return pd.DataFrame()
    view = collection
    if games is not None and games.height > 0 and "game_id" in games.columns:
        meta_cols = [
            c for c in games.columns if c == "game_id" or c not in view.columns
        ]
        view = view.join(games.select(meta_cols), on="game_id", how="left")

    pdf = view.to_pandas()
    if pdf.empty:
        return pd.DataFrame()

    ids = _safe_col(pdf, "game_id", default=0)
    names = [
        n if isinstance(n, str) and n else (gn or "")
        for n, gn in zip(_safe_col(pdf, "name", ""), _safe_col(pdf, "game_name", ""))
    ]
    years = _safe_col(pdf, "year_published", None)
    ratings = _safe_col(pdf, "user_rating", None)
    min_p = _safe_col(pdf, "min_players", None)
    max_p = _safe_col(pdf, "max_players", None)
    min_t = _safe_col(pdf, "min_playtime", None)
    max_t = _safe_col(pdf, "max_playtime", None)
    weights = _safe_col(pdf, "average_weight", None)
    best_pc = _safe_col(pdf, "best_player_counts", "")
    rec_pc = _safe_col(pdf, "recommended_player_counts", "")

    def _row_status(r: pd.Series) -> str:
        if bool(r.get("owned", False)):
            return "Own"
        if bool(r.get("preordered", False)):
            return "Preordered"
        if bool(r.get("wishlist", False)):
            return "Wishlist"
        if bool(r.get("want", False)) or bool(r.get("want_to_buy", False)):
            return "Want"
        if bool(r.get("previously_owned", False)) or bool(r.get("prev_owned", False)):
            return "Prev. Owned"
        return "—"

    statuses = [_row_status(pdf.iloc[i]) for i in range(len(pdf))]

    def _fmt_rating(r) -> str:
        if r is None or pd.isna(r) or r == 0:
            return ""
        return f"{float(r):.1f}"

    def _fmt_weight(w) -> str:
        if w is None or pd.isna(w):
            return ""
        return f"{float(w):.2f}"

    def _fmt_int(x):
        if x is None or pd.isna(x):
            return None
        return int(x)

    out = pd.DataFrame(
        {
            "Game": [
                bgg_link(int(g) if g else 0, n, y)
                for g, n, y in zip(ids, names, years)
            ],
            "Status": statuses,
            "Your rating": [_fmt_rating(r) for r in ratings],
            "Players": [
                format_range(_fmt_int(lo), _fmt_int(hi))
                for lo, hi in zip(min_p, max_p)
            ],
            "Best": [_sort_player_counts(c) for c in best_pc],
            "Recommended": [_sort_player_counts(c) for c in rec_pc],
            "Playtime": [
                format_range(_fmt_int(lo), _fmt_int(hi), " min")
                for lo, hi in zip(min_t, max_t)
            ],
            "Complexity": [_fmt_weight(w) for w in weights],
        }
    )

    # Sort: status priority (Own first), then rating descending. Pandas
    # sort_values handles NaN-last natively.
    rating_numeric = pd.Series(
        [
            float(r) if r not in (None, "") and not pd.isna(r) else float("nan")
            for r in ratings
        ]
    )
    status_rank = pd.Series(
        [
            _STATUS_PRIORITY.index(s) if s in _STATUS_PRIORITY else len(_STATUS_PRIORITY)
            for s in statuses
        ]
    )
    out = out.assign(_status_rank=status_rank.values, _rating=rating_numeric.values)
    out = out.sort_values(
        ["_status_rank", "_rating"], ascending=[True, False], na_position="last"
    )
    return out.drop(columns=["_status_rank", "_rating"]).reset_index(drop=True)


def format_eval_table(df: pl.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Compact eval-set predictions table: Rank | Game | Pr(Yes) | Owned."""
    view = df.head(top_n).to_pandas()
    ids = view["game_id"].tolist()
    names = view["name"].tolist() if "name" in view.columns else ["" for _ in ids]
    years = (
        view["year_published"].tolist()
        if "year_published" in view.columns
        else [None] * len(ids)
    )
    probs = view["proba"].tolist()
    rows = {
        "Rank": list(range(1, len(view) + 1)),
        "Game": [bgg_link(int(g), n, y) for g, n, y in zip(ids, names, years)],
        "Pr(Yes)": [round(float(p), 3) for p in probs],
    }
    if "label" in view.columns:
        rows["Owned"] = ["yes" if bool(x) else "no" for x in view["label"].tolist()]
    return pd.DataFrame(rows)


def format_eval_predictions(
    df: pl.DataFrame, *, top_n: int = 200
) -> pd.DataFrame:
    """Compact eval-set predictions: Game | Year | Pr(Yes) | Actual."""
    view = df.head(top_n).to_pandas()
    rank = list(range(1, len(view) + 1))
    ids = view["game_id"].tolist()
    names = view["name"].tolist() if "name" in view.columns else [""] * len(view)
    years = (
        view["year_published"].tolist()
        if "year_published" in view.columns
        else [None] * len(view)
    )
    probs = view["proba"].tolist() if "proba" in view.columns else [0.0] * len(view)
    out = {
        "Rank": rank,
        "Game": [bgg_link(g, n, y) for g, n, y in zip(ids, names, years)],
        "Pr(Yes)": [round(float(p), 3) for p in probs],
    }
    if "label" in view.columns:
        out["Owned"] = ["yes" if bool(x) else "no" for x in view["label"].tolist()]
    return pd.DataFrame(out)


def format_predictions_with_images(
    df: pl.DataFrame,
    *,
    status_lookup: dict[int, str],
    top_n: int = 50,
    show_actual: bool = False,
) -> pd.DataFrame:
    """Image+description-style predictions table for itables."""
    view = df.head(top_n).to_pandas()

    rank = list(range(1, len(view) + 1))
    images = view["image"].tolist() if "image" in view.columns else [None] * len(view)
    names = view["name"].tolist() if "name" in view.columns else [""] * len(view)
    years = (
        view["year_published"].tolist()
        if "year_published" in view.columns
        else [None] * len(view)
    )
    ids = view["game_id"].tolist()
    descs = (
        view["description"].tolist()
        if "description" in view.columns
        else [""] * len(view)
    )
    probs = (
        view["predicted_prob"].tolist()
        if "predicted_prob" in view.columns
        else view["proba"].tolist()
    )

    rows = {
        "Rank": rank,
        "Image": [img_tag(u) for u in images],
        "Game": [bgg_link(g, n, y) for g, n, y in zip(ids, names, years)],
        "Description": [truncate(d) for d in descs],
        "Pr(Yes)": [round(float(p), 3) for p in probs],
        "Status": [status_lookup.get(int(g), "—") for g in ids],
    }
    if show_actual and "label" in view.columns:
        rows["Owned"] = ["yes" if bool(x) else "no" for x in view["label"].tolist()]
    # Hidden helper column: carried through only when the caller's frame
    # has it (New & Upcoming). The qmd hides this column and uses its
    # value in a rowCallback to highlight rows for newly-predicted games.
    # Older Games / model report don't pass it, so this is a no-op there.
    if "is_new_7d" in view.columns:
        rows["_is_new"] = [bool(x) for x in view["is_new_7d"].tolist()]
    return pd.DataFrame(rows)


def format_menu_table(
    collection: pl.DataFrame, games: pl.DataFrame
) -> pd.DataFrame:
    """Static menu table matching the explorer's columns (no JS/filters):
    Game | Status | Your rating | Players | Recommended | Playtime | Complexity,
    with Recommended rendered as player-count badges and Complexity as a heat
    chip (same helpers as the explorer / section tables)."""
    from src.reports.game_cards import complexity_chip, player_badges

    base = format_collection_table(collection, games)  # Game|Status|rating|Players|Best|Recommended|Playtime|Complexity
    if base.empty:
        return pd.DataFrame()
    # Map game metadata for badge inputs, keyed by the bgg link's id.
    meta = {int(r["game_id"]): r for r in games.iter_rows(named=True)} if games.height else {}

    def _gid(link_html):
        import re
        m = re.search(r"/boardgame/(\d+)", str(link_html))
        return int(m.group(1)) if m else None

    out = pd.DataFrame(
        {
            "Game": base["Game"],
            "Status": base["Status"],
            "Your rating": base["Your rating"],
            "Players": base["Players"],
            "Playtime": base["Playtime"],
        }
    )
    gids = [_gid(x) for x in base["Game"]]
    out.insert(
        4,
        "Recommended",
        [
            player_badges(
                (meta.get(g) or {}).get("best_player_counts"),
                (meta.get(g) or {}).get("recommended_player_counts"),
            )
            if g is not None
            else ""
            for g in gids
        ],
    )
    out["Complexity"] = [
        complexity_chip((meta.get(g) or {}).get("average_weight")) if g is not None else ""
        for g in gids
    ]
    return out


def format_selection_table(
    games: pl.DataFrame, game_ids: list[int]
) -> pd.DataFrame:
    """Image+description table for the menu report's Locks/Maybes/Others.

    Columns: Image | Game | Description | Recommended | Complexity | Playtime.
    Rows follow `game_ids` order, skipping ids absent from `games`. Reuses the
    same badge/heat helpers as the explorer so all reports read alike.
    """
    from src.reports.game_cards import complexity_chip, player_badges

    if not game_ids or games is None or games.height == 0:
        return pd.DataFrame()
    by_id = {int(r["game_id"]): r for r in games.iter_rows(named=True)}
    rows = []
    for gid in game_ids:
        r = by_id.get(int(gid))
        if r is None:
            continue
        lo, hi = r.get("min_playtime"), r.get("max_playtime")
        rows.append(
            {
                "Image": img_tag(r.get("image") or r.get("thumbnail")),
                "Game": bgg_link(
                    int(gid), r.get("name") or r.get("game_name") or "", r.get("year_published")
                ),
                "Description": truncate(r.get("description")),
                "Recommended": player_badges(
                    r.get("best_player_counts"), r.get("recommended_player_counts")
                ),
                "Complexity": complexity_chip(r.get("average_weight")),
                "Playtime": format_range(
                    int(lo) if lo not in (None, 0) and not pd.isna(lo) else None,
                    int(hi) if hi not in (None, 0) and not pd.isna(hi) else None,
                    " min",
                ),
            }
        )
    return pd.DataFrame(rows)


def build_status_lookup(collection: pl.DataFrame) -> dict[int, str]:
    """Per-game status from the user's collection.

    Priority: Own > Preordered > Wishlist > Want > Prev. Owned > —.
    """
    lookup: dict[int, str] = {}
    if collection.height == 0 or "game_id" not in collection.columns:
        return lookup
    pdf = collection.to_pandas()
    for _, r in pdf.iterrows():
        gid = int(r["game_id"])
        if r.get("owned", False):
            lookup[gid] = "Own"
        elif r.get("preordered", False):
            lookup[gid] = "Preordered"
        elif r.get("wishlist", False):
            lookup[gid] = "Wishlist"
        elif r.get("want", False) or r.get("want_to_buy", False):
            lookup[gid] = "Want"
        elif r.get("previously_owned", False) or r.get("prev_owned", False):
            lookup[gid] = "Prev. Owned"
    return lookup


def build_topn_by_year_html(
    eval_view: pl.DataFrame, top_n_per_year: int = 15, since_year: int = 2015
) -> str:
    """Render the side-by-side Top Games by Year table as raw HTML.

    Returns an empty string if there's nothing to display.
    """
    view = eval_view.with_columns(pl.col("year_published").cast(pl.Int64))
    view = view.filter(pl.col("year_published") >= since_year)
    view = view.with_columns(
        pl.col("proba")
        .rank(method="ordinal", descending=True)
        .over("year_published")
        .alias("_rank")
    ).filter(pl.col("_rank") <= top_n_per_year)

    if view.height == 0:
        return ""

    years = sorted(int(y) for y in view["year_published"].unique().to_list())
    name_pivot = (
        view.pivot(values="name", index="_rank", on="year_published").sort("_rank")
    )
    label_pivot_pdf = None
    if "label" in view.columns:
        label_pivot_pdf = (
            view.pivot(values="label", index="_rank", on="year_published")
            .sort("_rank")
            .to_pandas()
            .set_index("_rank")
        )

    name_pdf = name_pivot.to_pandas().set_index("_rank").fillna("")
    name_pdf = name_pdf[[str(y) for y in years if str(y) in name_pdf.columns]]

    rows: list[str] = []
    rows.append('<div class="topn-by-year-scroll">')
    rows.append('<table class="topn-by-year">')
    rows.append("<thead><tr>")
    rows.append('<th class="rank-col">Rank</th>')
    for y in name_pdf.columns:
        rows.append(f"<th>{y}</th>")
    rows.append("</tr></thead><tbody>")

    for rank in name_pdf.index:
        rows.append("<tr>")
        rows.append(f'<td class="rank-col">{int(rank)}</td>')
        for col in name_pdf.columns:
            val = str(name_pdf.at[rank, col]) if name_pdf.at[rank, col] else ""
            owned = False
            if (
                label_pivot_pdf is not None
                and col in label_pivot_pdf.columns
                and rank in label_pivot_pdf.index
            ):
                truth = label_pivot_pdf.at[rank, col]
                owned = bool(truth) if truth is not None else False
            cell_class = "owned" if owned else ""
            display = truncate_name(val) if val else ""
            title_attr = f' title="{val}"' if val and len(val) > 28 else ""
            rows.append(f'<td class="{cell_class}"{title_attr}>{display}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    rows.append("</div>")
    return "\n".join(rows)


def format_model_details(
    registration: dict, outcome: str, candidate: str, version: str
) -> pd.DataFrame:
    """Provenance table: model identity, training metadata, headline metrics."""
    metrics = registration.get("metrics") or {}
    oof_overall = (registration.get("oof_metrics") or {}).get("overall") or {}

    return pd.DataFrame(
        [
            ("User", registration.get("username") or "—"),
            ("Outcome", outcome),
            ("Candidate", f"{candidate} (v{version})"),
            ("Task", maybe(registration.get("task"))),
            ("Tuning strategy", maybe(registration.get("tuning_strategy"))),
            ("Splits version", maybe(registration.get("splits_version"))),
            ("Trained through year", maybe(registration.get("finalize_through"))),
            (
                "Decision threshold",
                maybe(registration.get("threshold"), lambda v: f"{float(v):.3f}"),
            ),
            (
                "Train rows",
                maybe(registration.get("n_train_used"), lambda v: f"{int(v):,}"),
            ),
            ("Validation rows", maybe(registration.get("n_val"), lambda v: f"{int(v):,}")),
            ("Test rows", maybe(registration.get("n_test"), lambda v: f"{int(v):,}")),
            (
                "Best params",
                maybe(
                    registration.get("best_params"),
                    lambda v: ", ".join(f"{k}={vv}" for k, vv in v.items()),
                ),
            ),
            ("Trained at", short_dt(registration.get("trained_at"))),
            ("Finalized at", short_dt(registration.get("finalized_at"))),
            ("Git SHA", short_sha(registration.get("git_sha"))),
            (
                "Test ROC-AUC",
                maybe(metrics.get("roc_auc"), lambda v: f"{float(v):.4f}"),
            ),
            (
                "Test PR-AUC",
                maybe(metrics.get("pr_auc"), lambda v: f"{float(v):.4f}"),
            ),
            ("Test F1", maybe(metrics.get("f1"), lambda v: f"{float(v):.4f}")),
            (
                "OOF ROC-AUC",
                maybe(oof_overall.get("roc_auc"), lambda v: f"{float(v):.4f}"),
            ),
        ],
        columns=["Field", "Value"],
    )
