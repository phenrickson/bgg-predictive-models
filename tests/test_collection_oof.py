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


import numpy as np
import pandas as pd
import polars as pl

from src.collection.collection_model import (
    CollectionModel,
    ClassificationModelConfig,
    RegressionModelConfig,
)
from src.collection.outcomes import DirectColumnRule, OutcomeDefinition


def _classification_outcome() -> OutcomeDefinition:
    return OutcomeDefinition(
        name="own",
        task="classification",
        label_rule=DirectColumnRule(column="label"),
        require=None,
    )


_MINIMAL_PREPROCESSOR_KWARGS = {
    "create_category_features": False,
    "create_mechanic_features": False,
    "create_designer_features": False,
    "create_artist_features": False,
    "create_publisher_features": False,
    "create_family_features": False,
    "create_player_dummies": False,
}


def _make_classification_frame(n: int = 200, seed: int = 0) -> pl.DataFrame:
    """Tiny synthetic classification frame with two informative features.

    Uses varied year_published to pass variance threshold in preprocessor, and
    avoids BGG-specific list columns by using preprocessor_kwargs to disable them.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    logits = 1.5 * f1 - 0.5 * f2
    proba = 1.0 / (1.0 + np.exp(-logits))
    label = (rng.uniform(size=n) < proba).astype(int)
    return pl.DataFrame({
        "game_id": np.arange(n, dtype=np.int64),
        "year_published": rng.integers(1990, 2024, size=n).astype(np.int64),
        "users_rated": rng.integers(0, 1000, size=n).astype(np.int64),
        "f1": f1,
        "f2": f2,
        "label": label.astype(bool),
    })


def _fitted_classification_model(
    train_df: pl.DataFrame, val_df: pl.DataFrame
) -> CollectionModel:
    model = CollectionModel(
        username="tester",
        outcome=_classification_outcome(),
        classification_config=ClassificationModelConfig(
            model_type="logistic",
            preprocessor_kwargs=_MINIMAL_PREPROCESSOR_KWARGS,
        ),
    )
    model.train(train_df, params={})  # default params
    model.find_threshold(val_df)
    return model


def test_oof_predict_cv_classification_shape():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train, val)

    oof_preds, per_fold, overall = model.oof_predict_cv(train, n_folds=4)

    # one OOF prediction per training row
    assert oof_preds.height == train.height
    # original columns preserved + fold/proba/pred appended
    expected_extra = {"fold", "proba", "pred"}
    assert expected_extra.issubset(set(oof_preds.columns))
    # folds partition the rows
    folds = sorted(oof_preds["fold"].unique().to_list())
    assert folds == [0, 1, 2, 3]

    # per-fold metrics: one entry per fold, each carries n_rows
    assert len(per_fold) == 4
    fold_indices = sorted(p["fold"] for p in per_fold)
    assert fold_indices == [0, 1, 2, 3]
    assert all("n_rows" in p for p in per_fold)
    # pooled overall has the same metric keys as evaluate()
    eval_keys = set(model.evaluate(train).keys())
    assert eval_keys.issubset(set(overall.keys()))


def test_oof_predict_cv_classification_deterministic():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train, val)

    a, _, _ = model.oof_predict_cv(train, n_folds=3)
    b, _, _ = model.oof_predict_cv(train, n_folds=3)

    # same seed → same fold assignment and same predictions
    assert a["fold"].to_list() == b["fold"].to_list()
    assert a["proba"].to_list() == b["proba"].to_list()


def test_oof_predict_cv_uses_self_threshold_for_pred():
    train = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train, val)
    threshold = model.threshold
    assert threshold is not None

    oof_preds, _, _ = model.oof_predict_cv(train, n_folds=3)
    proba = np.asarray(oof_preds["proba"].to_list())
    pred = np.asarray(oof_preds["pred"].to_list())
    expected = (proba >= threshold).astype(int)
    assert np.array_equal(pred, expected)


def test_oof_predict_cv_requires_fitted():
    model = CollectionModel(
        username="tester",
        outcome=_classification_outcome(),
        classification_config=ClassificationModelConfig(
            model_type="logistic",
            preprocessor_kwargs=_MINIMAL_PREPROCESSOR_KWARGS,
        ),
    )
    train = _make_classification_frame(n=60)
    with pytest.raises(RuntimeError, match="not fit"):
        model.oof_predict_cv(train, n_folds=3)


def test_oof_predict_cv_requires_threshold_for_classification():
    train = _make_classification_frame(n=200, seed=0)
    model = CollectionModel(
        username="tester",
        outcome=_classification_outcome(),
        classification_config=ClassificationModelConfig(
            model_type="logistic",
            preprocessor_kwargs=_MINIMAL_PREPROCESSOR_KWARGS,
        ),
    )
    model.train(train, params={})  # fitted but no threshold
    with pytest.raises(RuntimeError, match="threshold"):
        model.oof_predict_cv(train, n_folds=3)


def test_oof_predict_cv_too_few_rows():
    # Model is fitted on a large enough frame; only the OOF call receives 5 rows.
    train_big = _make_classification_frame(n=200, seed=0)
    val = _make_classification_frame(n=80, seed=1)
    model = _fitted_classification_model(train_big, val)
    too_small = _make_classification_frame(n=5, seed=2)
    with pytest.raises(ValueError, match="too few rows"):
        model.oof_predict_cv(too_small, n_folds=5)


# ---------------------------------------------------------------------------
# Regression OOF tests (Task 3)
# ---------------------------------------------------------------------------


def _regression_outcome() -> OutcomeDefinition:
    return OutcomeDefinition(
        name="rating",
        task="regression",
        label_rule=DirectColumnRule(column="label"),
        require=None,
    )


def _make_regression_frame(n: int = 200, seed: int = 0) -> pl.DataFrame:
    """Tiny synthetic regression frame with two informative features.

    Uses varied year_published to pass variance threshold in preprocessor, and
    avoids BGG-specific list columns via _MINIMAL_PREPROCESSOR_KWARGS.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    target = 0.7 * f1 - 0.3 * f2 + rng.normal(scale=0.1, size=n)
    return pl.DataFrame({
        "game_id": np.arange(n, dtype=np.int64),
        "year_published": rng.integers(1990, 2024, size=n).astype(np.int64),
        "users_rated": rng.integers(0, 1000, size=n).astype(np.int64),
        "f1": f1,
        "f2": f2,
        "label": target.astype(np.float64),
    })


def test_oof_predict_cv_regression_shape():
    train = _make_regression_frame(n=200, seed=0)
    model = CollectionModel(
        username="tester",
        outcome=_regression_outcome(),
        regression_config=RegressionModelConfig(
            model_type="lightgbm",
            preprocessor_kwargs=_MINIMAL_PREPROCESSOR_KWARGS,
        ),
    )
    model.train(train, params={})

    oof_preds, per_fold, overall = model.oof_predict_cv(train, n_folds=4)

    # regression: no proba column, only fold + pred
    assert "fold" in oof_preds.columns
    assert "pred" in oof_preds.columns
    assert "proba" not in oof_preds.columns
    assert oof_preds.height == train.height

    # folds partition the rows
    folds = sorted(oof_preds["fold"].unique().to_list())
    assert folds == [0, 1, 2, 3]

    # pooled regression metrics
    assert {"rmse", "mae", "r2"}.issubset(set(overall.keys()))

    # per-fold list well-formed
    assert len(per_fold) == 4
    for entry in per_fold:
        assert "fold" in entry and "n_rows" in entry
        assert "rmse" in entry
