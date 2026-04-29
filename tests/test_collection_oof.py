"""Tests for OOF cross-validation artifact generation in collection runs."""

from __future__ import annotations

import pytest

from src.collection.candidates import CollectionCandidate
from src.collection.collection_model import ClassificationModelConfig


def _classification_candidate(name: str = "cand", **overrides) -> CollectionCandidate:
    kwargs = {
        "name": name,
        "classification_config": ClassificationModelConfig(model_type="logistic"),
        "tuning": "none",
        "fixed_params": {},
    }
    kwargs.update(overrides)
    return CollectionCandidate(**kwargs)


def test_oof_cv_folds_defaults_to_5():
    cand = _classification_candidate()
    assert cand.oof_cv_folds == 5


def test_oof_cv_folds_zero_disables():
    cand = _classification_candidate(oof_cv_folds=0)
    assert cand.oof_cv_folds == 0


def test_oof_cv_folds_negative_rejected():
    with pytest.raises(ValueError, match="oof_cv_folds"):
        _classification_candidate(oof_cv_folds=-1)


def test_oof_cv_folds_one_rejected():
    with pytest.raises(ValueError, match="oof_cv_folds"):
        _classification_candidate(oof_cv_folds=1)


def test_oof_cv_folds_round_trips_through_to_dict():
    cand = _classification_candidate(oof_cv_folds=7)
    d = cand.to_dict()
    assert d["oof_cv_folds"] == 7
    restored = CollectionCandidate.from_dict(d)
    assert restored.oof_cv_folds == 7


def test_oof_cv_folds_omitted_in_dict_uses_default():
    cand = _classification_candidate(oof_cv_folds=5)
    d = cand.to_dict()
    d.pop("oof_cv_folds", None)
    restored = CollectionCandidate.from_dict(d)
    assert restored.oof_cv_folds == 5
