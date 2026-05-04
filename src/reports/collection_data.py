"""Data loader for the collection Quarto report.

Returns a `CollectionReportData` aggregate that the `.qmd` template
consumes. Splits user-level (outcome-agnostic) data from per-outcome
artifacts so multi-outcome reports can iterate `data.outcomes` without
a loader refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class OutcomeArtifacts:
    """All artifacts the report needs for one (user, outcome)."""

    outcome: str
    selected_candidate: str
    selected_version: int

    pipeline: Any
    registration: dict
    threshold: float | None
    feature_importance: pl.DataFrame

    oof_predictions: pl.DataFrame
    val_predictions: pl.DataFrame
    test_predictions: pl.DataFrame

    upcoming_predictions: pl.DataFrame


@dataclass
class CollectionReportData:
    """Full data bundle for a single rendering of the collection report."""

    username: str
    collection: pl.DataFrame
    games: pl.DataFrame
    outcomes: dict[str, OutcomeArtifacts]
