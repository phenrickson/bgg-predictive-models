"""Small formatting helpers shared across report tables."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd


_TREE_MODEL_TYPES = {
    "lightgbm",
    "xgboost",
    "random_forest",
    "gradient_boosting",
    "catboost",
}
_LINEAR_MODEL_TYPES = {
    "logistic",
    "logistic_regression",
    "linear",
    "linear_regression",
    "ridge",
    "lasso",
    "elasticnet",
}


def model_kind(registration: dict) -> str:
    """Return ``"tree"`` or ``"linear"`` for a model based on its
    registration.json. Defaults to ``"linear"`` when the type is
    unknown — that's the older behavior and the safer fallback for
    plotting (signed coefficients render fine for nonneg values too).
    """
    spec = registration.get("candidate_spec") or {}
    cfg = spec.get("classification_config") or spec.get("regression_config") or {}
    mtype = (cfg.get("model_type") or "").lower()
    if mtype in _TREE_MODEL_TYPES:
        return "tree"
    if mtype in _LINEAR_MODEL_TYPES:
        return "linear"
    return "linear"


def bgg_link(game_id: int, name: str, year: int | None) -> str:
    year_part = ""
    if year is not None and not (isinstance(year, float) and pd.isna(year)):
        try:
            year_part = f" ({int(year)})"
        except (TypeError, ValueError):
            year_part = ""
    return (
        f'<a href="https://boardgamegeek.com/boardgame/{int(game_id)}" '
        f'target="_blank" rel="noopener">{name}{year_part}</a>'
    )


def img_tag(url: str | None) -> str:
    if not url or (isinstance(url, float) and pd.isna(url)):
        return '<div class="cover-cell"></div>'
    return (
        f'<div class="cover-cell">'
        f'<img class="cover-thumb" src="{url}" loading="lazy" alt="" />'
        f"</div>"
    )


def truncate(text: str | None, max_len: int = 220) -> str:
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("&#10;", " ").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def truncate_name(text: str, max_len: int = 28) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def format_range(lo, hi, suffix: str = "") -> str:
    """Render a min/max pair as 'lo–hi' or 'lo' when they're equal."""
    if lo is None and hi is None:
        return ""
    if lo is None:
        return f"{hi}{suffix}"
    if hi is None or lo == hi:
        return f"{lo}{suffix}"
    return f"{lo}–{hi}{suffix}"


def short_sha(sha: str | None) -> str:
    return (sha or "")[:8] or "—"


def short_dt(value: str | None) -> str:
    if not value:
        return "—"
    return str(value).split(".")[0].replace("T", " ")


def maybe(value: Any, fmt: Callable[[Any], str] = str) -> str:
    if value is None or (isinstance(value, str) and not value):
        return "—"
    try:
        return fmt(value)
    except Exception:  # noqa: BLE001
        return str(value)
