"""Render games as cover-tile cards for the menu report.

Card = image, name+year, description, recommended player-count badges,
complexity heat chip, playtime, and Pr(Yes). Badge/chip styling mirrors the
JS explorer so the two reports read the same.
"""

from __future__ import annotations

import html

import pandas as pd
import polars as pl

# Complexity heat anchors — same range the explorer uses (>=25-rating games).
_WMIN, _WMID, _WMAX = 1.0, 2.75, 4.5
_W_LOW = (59, 130, 246)      # blue
_W_MID = (241, 243, 246)     # near-white
_W_HIGH = (234, 120, 40)     # orange


def _mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def complexity_chip(weight) -> str:
    if weight is None or (isinstance(weight, float) and pd.isna(weight)) or weight == 0:
        return '<span class="badge-none">—</span>'
    w = float(weight)
    if w <= _WMID:
        t = max(0.0, (w - _WMIN) / (_WMID - _WMIN))
        rgb = _mix(_W_LOW, _W_MID, t)
    else:
        t = min(1.0, (w - _WMID) / (_WMAX - _WMID))
        rgb = _mix(_W_MID, _W_HIGH, t)
    lum = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
    ink = "#1e222b" if lum > 170 else "#fff"
    return (
        f'<span class="weight-badge" style="--wc:rgb({rgb[0]} {rgb[1]} {rgb[2]});'
        f'--wi:{ink};">{w:.2f}</span>'
    )


def _counts(raw) -> list[int]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    out = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return sorted(out)


def player_badges(best_raw, rec_raw) -> str:
    """Recommended counts as badges; best ones filled, recommended-only outlined."""
    best = set(_counts(best_raw))
    rec = sorted(set(_counts(rec_raw)) | best)
    if not rec:
        return '<span class="badge-none">—</span>'
    out = []
    for n in rec:
        is_best = n in best
        cls = "pc-badge pc-best" if is_best else "pc-badge pc-rec"
        title = f"Best at {n}" if is_best else f"Recommended at {n}"
        out.append(f'<span class="{cls}" title="{title}">{n}</span>')
    return "".join(out)


def _playtime(row) -> str:
    lo, hi = row.get("min_playtime"), row.get("max_playtime")
    def _i(x):
        return None if x is None or (isinstance(x, float) and pd.isna(x)) or x == 0 else int(x)
    lo, hi = _i(lo), _i(hi)
    if lo is None and hi is None:
        return ""
    if lo == hi or hi is None:
        return f"{lo} min"
    if lo is None:
        return f"{hi} min"
    return f"{lo}–{hi} min"


def _bgg_link(gid, name, year) -> str:
    yr = ""
    if year is not None and not (isinstance(year, float) and pd.isna(year)):
        yr = f" ({int(year)})"
    nm = html.escape(str(name or f"Game {gid}"))
    return (
        f'<a href="https://boardgamegeek.com/boardgame/{int(gid)}" '
        f'target="_blank" rel="noopener">{nm}{yr}</a>'
    )


def _img(url) -> str:
    if not url or (isinstance(url, float) and pd.isna(url)):
        return '<div class="card-thumb card-thumb-empty"></div>'
    return f'<div class="card-thumb"><img src="{html.escape(str(url))}" loading="lazy"/></div>'


def _truncate(text, n=240) -> str:
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).replace("&#10;", " ").replace("\n", " ").strip()
    t = html.unescape(t)
    if len(t) > n:
        t = t[: n - 1].rstrip() + "…"
    return html.escape(t)


def game_cards_html(games: pl.DataFrame, game_ids: list[int], proba: dict | None = None) -> str:
    """Card grid for `game_ids`, in order, skipping ids absent from `games`.

    `proba` maps game_id -> float for the Pr(Yes) badge (optional).
    """
    if not game_ids or games is None or games.height == 0:
        return ""
    proba = proba or {}
    by_id = {int(r["game_id"]): r for r in games.iter_rows(named=True)}
    cards = []
    for gid in game_ids:
        row = by_id.get(int(gid))
        if row is None:
            continue
        title = _bgg_link(gid, row.get("name") or row.get("game_name"), row.get("year_published"))
        badges = player_badges(row.get("best_player_counts"), row.get("recommended_player_counts"))
        pr = proba.get(int(gid))
        pr_html = f'<span class="card-meta-val">{float(pr):.3f}</span>' if pr is not None else ""
        cards.append(
            '<div class="game-card">'
            + _img(row.get("image") or row.get("thumbnail"))
            + '<div class="card-body">'
            + f'<div class="card-title">{title}</div>'
            + f'<div class="card-desc">{_truncate(row.get("description"))}</div>'
            + '<div class="card-meta">'
            + f'<span class="card-meta-item"><span class="card-meta-label">Players</span>{badges}</span>'
            + f'<span class="card-meta-item"><span class="card-meta-label">Complexity</span>{complexity_chip(row.get("average_weight"))}</span>'
            + f'<span class="card-meta-item"><span class="card-meta-label">Playtime</span>{_playtime(row)}</span>'
            + (f'<span class="card-meta-item"><span class="card-meta-label">Pr(Yes)</span>{pr_html}</span>' if pr_html else "")
            + "</div></div></div>"
        )
    if not cards:
        return ""
    return '<div class="card-grid">' + "".join(cards) + "</div>"
