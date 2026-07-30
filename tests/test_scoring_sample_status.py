"""Tests for sample_status / training_cutoff_year in services.scoring.main."""

import pandas as pd
import pytest

from services.scoring.sample_status import (
    compute_sample_status,
    resolve_training_cutoff_year,
)


def _registration(test_through):
    return {"original_experiment": {"metadata": {"test_through": test_through}}}


def _all_targets(test_through):
    return {
        target: _registration(test_through)
        for target in ("hurdle", "complexity", "rating", "users_rated", "geek_rating")
    }


def test_cutoff_read_from_registration_when_models_agree():
    assert resolve_training_cutoff_year(_all_targets(2024)) == 2024


def test_cutoff_takes_minimum_when_models_disagree():
    # in_sample must mean "seen by every model", so the narrowest cutoff wins
    registrations = _all_targets(2024)
    registrations["geek_rating"] = _registration(2023)

    assert resolve_training_cutoff_year(registrations) == 2023


def test_cutoff_raises_rather_than_guessing_when_absent():
    # Falling back to config.yaml would let the flag drift from the loaded model
    registrations = _all_targets(2024)
    registrations["rating"] = {"original_experiment": {"metadata": {}}}

    with pytest.raises(ValueError, match="test_through"):
        resolve_training_cutoff_year(registrations)


def test_cutoff_raises_when_registration_has_no_experiment_metadata():
    with pytest.raises(ValueError, match="test_through"):
        resolve_training_cutoff_year({"hurdle": {}})


def test_sample_status_splits_at_the_cutoff_year():
    years = pd.Series([2022, 2023, 2024, 2025, 2026])

    status = compute_sample_status(years, training_cutoff_year=2024)

    assert list(status) == [
        "in_sample",
        "in_sample",
        "in_sample",  # the cutoff year itself was fitted on
        "out_of_sample",
        "out_of_sample",
    ]


def test_sample_status_preserves_index_for_row_alignment():
    years = pd.Series([2020, 2030], index=[7, 11])

    status = compute_sample_status(years, training_cutoff_year=2024)

    assert list(status.index) == [7, 11]
    assert status.loc[7] == "in_sample"
    assert status.loc[11] == "out_of_sample"
