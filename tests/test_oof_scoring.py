"""Tests for K-fold OOF prediction utility."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.oof import kfold_oof_predict


def _make_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())])


def test_oof_predictions_have_same_length_as_input():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
    y = pd.Series((rng.normal(size=100) > 0).astype(int))

    preds = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=42)
    assert preds.shape == (100,)


def test_oof_is_deterministic_with_seed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    y = pd.Series((rng.normal(size=50) > 0).astype(int))

    p1 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=7)
    p2 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=7)
    np.testing.assert_array_equal(p1, p2)


def test_oof_differs_with_different_seed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
    y = pd.Series((rng.normal(size=50) > 0).astype(int))

    p1 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=1)
    p2 = kfold_oof_predict(_make_pipeline(), X, y, k=5, seed=2)
    assert not np.array_equal(p1, p2)


def test_oof_with_proba_returns_class_one_proba():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=list("abc"))
    y = pd.Series((rng.normal(size=80) > 0).astype(int))

    preds = kfold_oof_predict(_make_pipeline(), X, y, k=4, seed=0, predict_proba=True)
    assert preds.shape == (80,)
    assert preds.min() >= 0 and preds.max() <= 1
