"""Tests for new src.collection.viz helpers used by the report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.collection.viz import extract_finalized_importance


def _passthrough(x):
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def _build_pipeline_with_coef():
    preprocessor = FunctionTransformer(_passthrough, validate=False)
    model = DummyClassifier(strategy="constant", constant=0)
    model.fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))
    model.coef_ = np.array([[0.7, -0.4, 0.1]])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def test_extract_finalized_importance_uses_coef():
    pipeline = _build_pipeline_with_coef()
    train_sample = pd.DataFrame(
        {"feat_a": [0.0, 1.0], "feat_b": [0, 1], "feat_c": [0.5, 0.5]}
    )
    df = extract_finalized_importance(pipeline, train_sample)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"feature", "value", "abs_value"}
    assert df["feature"].tolist() == ["feat_a", "feat_b", "feat_c"]
    assert df["abs_value"].is_monotonic_decreasing


def test_extract_finalized_importance_returns_none_when_unsupported():
    """If the model has neither coef_ nor feature_importances_, return None."""
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        [
            ("preprocessor", FunctionTransformer(validate=False)),
            ("model", StandardScaler()),
        ]
    )
    out = extract_finalized_importance(pipeline, pd.DataFrame({"x": [1, 2]}))
    assert out is None
