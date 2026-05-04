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


from pathlib import Path

DEFAULT_CANDIDATE = "logistic_row_norm"


def _list_candidate_versions(user_outcome_dir: Path, candidate: str) -> list[int]:
    """Return all integer versions for a candidate dir, ascending."""
    cand_dir = user_outcome_dir / candidate
    if not cand_dir.exists():
        return []
    versions: list[int] = []
    for child in cand_dir.iterdir():
        if not (child.is_dir() and child.name.startswith("v")):
            continue
        try:
            versions.append(int(child.name[1:]))
        except ValueError:
            continue
    return sorted(versions)


def _is_finalized(user_outcome_dir: Path, candidate: str, version: int) -> bool:
    """A candidate version is 'finalized' iff it has finalized.pkl."""
    return (user_outcome_dir / candidate / f"v{version}" / "finalized.pkl").exists()


def _list_finalized_candidates(user_outcome_dir: Path) -> list[tuple[str, int]]:
    """Return (candidate, latest_finalized_version) for every candidate in the
    outcome dir that has at least one finalized version."""
    if not user_outcome_dir.exists():
        return []
    out: list[tuple[str, int]] = []
    for child in user_outcome_dir.iterdir():
        if not child.is_dir() or child.name.startswith("_") or child.name.startswith("v"):
            continue
        cand = child.name
        finalized_versions = [
            v
            for v in _list_candidate_versions(user_outcome_dir, cand)
            if _is_finalized(user_outcome_dir, cand, v)
        ]
        if finalized_versions:
            out.append((cand, max(finalized_versions)))
    return sorted(out)


def select_candidate(
    root: Path,
    username: str,
    outcome: str,
    candidate: str | None = None,
) -> tuple[str, int]:
    """Pick a (candidate, version) for a user/outcome.

    Resolution order:
        1. If `candidate` is given and has a finalized version, use its
           latest finalized version.
        2. Otherwise prefer DEFAULT_CANDIDATE if it has a finalized version.
        3. Otherwise pick any finalized candidate (alphabetically first).
        4. Raise ValueError if nothing is finalized.
    """
    user_outcome_dir = Path(root) / username / outcome
    finalized = dict(_list_finalized_candidates(user_outcome_dir))

    if candidate is not None:
        if candidate not in finalized:
            raise ValueError(
                f"Candidate {candidate!r} is not finalized for "
                f"{username}/{outcome} under {root}"
            )
        return candidate, finalized[candidate]

    if DEFAULT_CANDIDATE in finalized:
        return DEFAULT_CANDIDATE, finalized[DEFAULT_CANDIDATE]

    if not finalized:
        raise ValueError(
            f"No finalized candidate found for {username}/{outcome} under {root}"
        )

    cand = sorted(finalized.keys())[0]
    return cand, finalized[cand]
