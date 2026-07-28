"""Render games as cover-tile cards for the menu report.

Card = image, name+year, description, recommended player-count badges,
complexity heat chip, playtime, and Pr(Yes). Badge/chip styling mirrors the
JS explorer so the two reports read the same.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl

# Downscaled cover images are embedded as data URIs so the report stays
# self-contained without inlining full-res originals (which bloat it to tens of
# MB). Fetched originals are resized to COVER_PX and cached on disk so repeat
# renders don't re-download.
COVER_PX = 400
_COVER_CACHE = Path("/tmp/bgg_cover_cache")


def _cover_data_uri(url) -> str:
    """Fetch `url`, downscale to <=COVER_PX, return a base64 JPEG data URI.
    Falls back to the raw URL on any failure (network, decode, no Pillow)."""
    if not url or (isinstance(url, float) and pd.isna(url)):
        return ""
    url = str(url)
    key = hashlib.sha1(f"{url}@{COVER_PX}".encode()).hexdigest()[:16]
    cached = _COVER_CACHE / f"{key}.txt"
    if cached.exists():
        return cached.read_text()
    try:
        from PIL import Image

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=20).read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((COVER_PX, COVER_PX))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=82)
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        _COVER_CACHE.mkdir(parents=True, exist_ok=True)
        cached.write_text(uri)
        return uri
    except Exception:
        return url  # graceful fallback: link the remote image

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


# Community-convention complexity tiers (no official BGG standard; these
# cutoffs are a chosen mapping for readability).
_WEIGHT_TIERS = [
    (1.9, "Light"),
    (2.4, "Light-Medium"),
    (3.0, "Medium"),
    (3.9, "Medium-Heavy"),
    (5.0, "Heavy"),
]


def complexity_label(weight) -> str:
    if weight is None or (isinstance(weight, float) and pd.isna(weight)) or weight == 0:
        return ""
    w = float(weight)
    for hi, label in _WEIGHT_TIERS:
        if w <= hi:
            return label
    return "Heavy"


def complexity_chip_labeled(weight) -> str:
    """Heat chip showing the tier label instead of the number."""
    label = complexity_label(weight)
    if not label:
        return '<span class="badge-none">—</span>'
    base = complexity_chip(weight)  # carries the --wc/--wi gradient vars
    # swap the numeric text for the label, keep the styling span
    import re
    return re.sub(r">[\d.]+<", f">{label}<", base)


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


def _name_year(gid, name, year) -> str:
    yr = ""
    if year is not None and not (isinstance(year, float) and pd.isna(year)):
        yr = f'<span class="tile-year">{int(year)}</span>'
    return f'{html.escape(str(name or f"Game {gid}"))}{yr}'


def game_cards_html(games: pl.DataFrame, game_ids: list[int], tier: str = "") -> str:
    """Cover-forward tile grid for the Spain menu report.

    Each tile is a link to the game's BGG page (new tab): cover image, name +
    year, a meta strip (recommended-player badges · complexity tier · playtime),
    and a truncated description. `tier` (locks/maybes/others) adds a class so
    the stylesheet can weight the sections differently. Order preserved;
    ids absent from `games` are skipped.
    """
    if not game_ids or games is None or games.height == 0:
        return ""
    by_id = {int(r["game_id"]): r for r in games.iter_rows(named=True)}
    tier_cls = f" tile-{tier}" if tier else ""
    cards = []
    for gid in game_ids:
        row = by_id.get(int(gid))
        if row is None:
            continue
        # Downscale the full-res cover to ~400px and embed it (crisp at tile
        # size, ~35KB each) rather than inlining the multi-MB original or the
        # soft 200px thumbnail.
        src = _cover_data_uri(row.get("image") or row.get("thumbnail"))
        img_html = f'<img src="{src}" loading="lazy" alt=""/>' if src else ""
        cards.append(
            f'<a class="tile{tier_cls}" '
            f'href="https://boardgamegeek.com/boardgame/{int(gid)}" '
            f'target="_blank" rel="noopener">'
            f'<div class="tile-cover">{img_html}</div>'
            f'<div class="tile-body">'
            f'<div class="tile-name">{_name_year(gid, row.get("name") or row.get("game_name"), row.get("year_published"))}</div>'
            f'<div class="tile-desc">{_truncate(row.get("description"), 160)}</div>'
            f'<div class="tile-meta">'
            f'<span class="tile-meta-item"><span class="tile-meta-k">Players</span>{player_badges(row.get("best_player_counts"), row.get("recommended_player_counts"))}</span>'
            f'<span class="tile-meta-item"><span class="tile-meta-k">Complexity</span>{complexity_chip_labeled(row.get("average_weight"))}</span>'
            f'<span class="tile-meta-item"><span class="tile-meta-k">Playtime</span><span class="tile-time">{_playtime(row)}</span></span>'
            f'</div></div></a>'
        )
    if not cards:
        return ""
    return '<div class="tile-grid">' + "".join(cards) + "</div>"


def game_strips_html(
    games: pl.DataFrame, game_ids: list[int], complexity: str = "label"
) -> str:
    """Compact horizontal-strip list: one thin full-width row per game —
    small cover thumbnail on the left, name + year and a meta row (recommended
    player badges · complexity · playtime) beside it. Shows every id in order;
    skips ids absent from `games`. Whole row links to BGG.

    complexity="label" (default) shows the tier chip (Light..Heavy); "number"
    shows the numeric weight chip.
    """
    if not game_ids or games is None or games.height == 0:
        return ""
    cx_fn = complexity_chip if complexity == "number" else complexity_chip_labeled
    by_id = {int(r["game_id"]): r for r in games.iter_rows(named=True)}
    rows = []
    for gid in game_ids:
        row = by_id.get(int(gid))
        if row is None:
            continue
        src = _cover_data_uri(row.get("image") or row.get("thumbnail"))
        img_html = f'<img src="{src}" loading="lazy" alt=""/>' if src else ""
        rows.append(
            f'<a class="strip" '
            f'href="https://boardgamegeek.com/boardgame/{int(gid)}" '
            f'target="_blank" rel="noopener">'
            f'<div class="strip-cover">{img_html}</div>'
            f'<div class="strip-main">'
            f'<div class="strip-name">{_name_year(gid, row.get("name") or row.get("game_name"), row.get("year_published"))}</div>'
            f'<div class="strip-meta">'
            f'<span class="strip-meta-item">{player_badges(row.get("best_player_counts"), row.get("recommended_player_counts"))}</span>'
            f'<span class="strip-meta-item">{cx_fn(row.get("average_weight"))}</span>'
            f'<span class="strip-meta-item strip-time">{_playtime(row)}</span>'
            f'</div></div></a>'
        )
    if not rows:
        return ""
    return '<div class="strip-list">' + "".join(rows) + "</div>"
