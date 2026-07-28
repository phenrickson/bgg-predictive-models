"""Themed 'top games' sections for the collection report.

Each section is the user's top-N owned games (by their rating) matching a
criterion — a best-player-count bucket or a category/mechanic. Returns
ordered game_id lists that the report renders as tile cards via
``game_cards_html``.
"""

from __future__ import annotations

import polars as pl


def _best_counts(raw) -> set[int]:
    if raw is None:
        return set()
    return {int(t.strip()) for t in str(raw).split(",") if t.strip().isdigit()}


def _has(seq, value) -> bool:
    return bool(seq) and value in seq


# Player-count recommendation buckets: label -> predicate over the best set.
_PLAYER_BUCKETS = [
    ("Best at 2 Players", lambda b: 2 in b),
    ("Best at 3–4 Players", lambda b: bool(b & {3, 4})),
    ("Best at 5 Players", lambda b: 5 in b),
    ("Best at 6+ Players", lambda b: any(n >= 6 for n in b)),
]

# Game-type buckets: label -> (field, value). field in categories/mechanics.
_TYPE_BUCKETS = [
    ("Economic", "categories", "Economic"),
    ("War", "categories", "Wargame"),
    ("Cooperative", "mechanics", "Cooperative Game"),
    ("Party", "categories", "Party Game"),
]


def _owned_with_rating(collection: pl.DataFrame, games: pl.DataFrame) -> list[dict]:
    """Owned games joined to metadata, each with a `_rating` sort key."""
    if collection.height == 0:
        return []
    owned = (
        collection.filter(pl.col("owned")) if "owned" in collection.columns else collection
    )
    rating = {
        int(r["game_id"]): (r.get("user_rating") or 0.0) for r in owned.iter_rows(named=True)
    }
    meta = {int(r["game_id"]): r for r in games.iter_rows(named=True)}
    out = []
    for gid in rating:
        m = meta.get(gid)
        if m is None:
            continue
        d = dict(m)
        d["_rating"] = float(rating[gid] or 0.0)
        out.append(d)
    # highest rating first; ties broken by name for stable order
    out.sort(key=lambda d: (-d["_rating"], str(d.get("name") or "")))
    return out


def build_sections(
    collection: pl.DataFrame, games: pl.DataFrame, top_n: int = 6
) -> list[tuple[str, str, list[int]]]:
    """Return ordered ``(group, label, game_ids)`` sections.

    group is "Recommendations" or "Game Types"; game_ids are the user's top
    `top_n` owned games (by rating) matching the section, best first.
    """
    rows = _owned_with_rating(collection, games)
    sections: list[tuple[str, str, list[int]]] = []

    for label, pred in _PLAYER_BUCKETS:
        ids = [
            int(d["game_id"])
            for d in rows
            if pred(_best_counts(d.get("best_player_counts")))
        ][:top_n]
        sections.append(("Recommendations", label, ids))

    for label, field, value in _TYPE_BUCKETS:
        ids = [
            int(d["game_id"]) for d in rows if _has(d.get(field), value)
        ][:top_n]
        sections.append(("Game Types", label, ids))

    return sections
