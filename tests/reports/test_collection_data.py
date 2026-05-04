"""Tests for src.reports.collection_data."""

from __future__ import annotations

import polars as pl
import pytest

from src.reports.collection_data import CollectionReportData, OutcomeArtifacts


def test_outcome_artifacts_has_required_fields():
    fields = OutcomeArtifacts.__dataclass_fields__
    expected = {
        "outcome",
        "selected_candidate",
        "selected_version",
        "pipeline",
        "registration",
        "threshold",
        "feature_importance",
        "oof_predictions",
        "val_predictions",
        "test_predictions",
        "upcoming_predictions",
    }
    assert set(fields) == expected


def test_collection_report_data_has_required_fields():
    fields = CollectionReportData.__dataclass_fields__
    expected = {"username", "collection", "games", "outcomes"}
    assert set(fields) == expected


def test_collection_report_data_outcomes_is_dict():
    data = CollectionReportData(
        username="phenrickson",
        collection=pl.DataFrame(),
        games=pl.DataFrame(),
        outcomes={},
    )
    assert isinstance(data.outcomes, dict)
