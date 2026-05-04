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


import json
from pathlib import Path

from src.reports.collection_data import select_candidate


def test_select_candidate_prefers_logistic_row_norm(fixture_collection_root: Path):
    cand, version = select_candidate(
        fixture_collection_root, "phenrickson", "own"
    )
    assert cand == "logistic_row_norm"
    assert version == 1


def test_select_candidate_explicit_override(fixture_collection_root: Path):
    other_dir = fixture_collection_root / "phenrickson" / "own" / "lgbm_default" / "v1"
    other_dir.mkdir(parents=True)
    (other_dir / "finalized.pkl").write_bytes(b"x")
    (other_dir / "registration.json").write_text(
        json.dumps(
            {"candidate": "lgbm_default", "version": 1, "splits_version": 1}
        )
    )

    cand, version = select_candidate(
        fixture_collection_root,
        "phenrickson",
        "own",
        candidate="lgbm_default",
    )
    assert cand == "lgbm_default"
    assert version == 1


def test_select_candidate_raises_when_no_finalized(tmp_path: Path):
    user_dir = tmp_path / "phenrickson" / "own"
    user_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="No finalized candidate"):
        select_candidate(tmp_path, "phenrickson", "own")


from src.reports.collection_data import _read_json, _read_parquet, _read_pickle


def test_read_json_local(fixture_collection_root: Path):
    path = (
        fixture_collection_root
        / "phenrickson"
        / "own"
        / "logistic_row_norm"
        / "v1"
        / "registration.json"
    )
    data = _read_json(str(path))
    assert data["candidate"] == "logistic_row_norm"


def test_read_parquet_local(fixture_collection_root: Path):
    path = (
        fixture_collection_root
        / "phenrickson"
        / "own"
        / "logistic_row_norm"
        / "v1"
        / "predictions"
        / "oof.parquet"
    )
    df = _read_parquet(str(path))
    assert df.height == 3


def test_read_pickle_local(fixture_collection_root: Path):
    path = (
        fixture_collection_root
        / "phenrickson"
        / "own"
        / "logistic_row_norm"
        / "v1"
        / "model.pkl"
    )
    pipeline = _read_pickle(str(path))
    assert hasattr(pipeline, "predict")
