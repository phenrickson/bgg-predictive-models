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


from src.collection.viz import metrics_table


def test_metrics_table_returns_wide_dataframe():
    registration = {
        "metrics": {"roc_auc": 0.85, "pr_auc": 0.6},
        "val_metrics": {"roc_auc": 0.82, "pr_auc": 0.55},
        "oof_metrics": {"overall": {"roc_auc": 0.8, "pr_auc": 0.5}},
    }
    df = metrics_table(registration)
    assert "split" in df.columns
    assert "roc_auc" in df.columns
    assert "pr_auc" in df.columns
    splits = set(df["split"].tolist())
    assert {"val", "oof", "test"}.issubset(splits)


def test_metrics_table_handles_missing_metrics():
    df = metrics_table({})
    assert df.height == 0 if hasattr(df, "height") else len(df) == 0
    # pandas DataFrame at least has the split column header
    assert "split" in df.columns


from src.collection.viz import plot_separation


def test_plot_separation_returns_plotly_figure():
    preds = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "proba": [0.9, 0.1, 0.7, 0.3, 0.5],
            "label": [True, False, True, False, False],
        }
    )
    fig = plot_separation(preds, title="Test")
    assert hasattr(fig, "data")
    assert hasattr(fig, "layout")


def test_plot_separation_handles_empty():
    preds = pl.DataFrame({"proba": [], "label": []})
    fig = plot_separation(preds, title="Empty")
    assert hasattr(fig, "data")


from src.collection.viz import top_n_by_year_table


def test_top_n_by_year_table_returns_pivot():
    preds = pl.DataFrame(
        {
            "game_id": list(range(1, 9)),
            "name": [f"G{i}" for i in range(1, 9)],
            "year_published": [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022],
            "proba": [0.9, 0.7, 0.3, 0.8, 0.6, 0.2, 0.95, 0.5],
            "label": [True, False, False, True, True, False, True, False],
        }
    )
    df = top_n_by_year_table(preds, top_n=2)
    assert "rank" in df.columns
    year_cols = [c for c in df.columns if c != "rank"]
    assert {"2020", "2021", "2022"}.issubset(set(year_cols))
    assert df.height == 2


def test_top_n_by_year_table_empty_returns_empty_frame():
    preds = pl.DataFrame(
        {
            "game_id": [],
            "name": [],
            "year_published": [],
            "proba": [],
            "label": [],
        }
    )
    df = top_n_by_year_table(preds, top_n=5)
    assert df.height == 0


from src.collection.viz import predictions_datatable


def test_predictions_datatable_returns_pandas():
    preds = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "year_published": [2020, 2021, 2022],
            "proba": [0.9, 0.4, 0.7],
            "label": [True, False, True],
        }
    )
    games = pl.DataFrame({"game_id": [1, 2, 3]})
    out = predictions_datatable(preds, games, top_n=10)
    assert isinstance(out, pd.DataFrame)
    # Sorted descending by proba
    assert list(out["proba"]) == sorted(out["proba"], reverse=True)


def test_predictions_datatable_filters_min_users_rated():
    preds = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "year_published": [2020, 2021, 2022],
            "proba": [0.9, 0.4, 0.7],
            "users_rated": [100, 0, 50],
            "label": [True, False, True],
        }
    )
    games = pl.DataFrame({"game_id": [1, 2, 3]})
    out = predictions_datatable(preds, games, min_users_rated=10)
    assert set(out["game_id"]) == {1, 3}


from src.collection.viz import (
    collection_datatable,
    plot_collection_by_category,
    plot_collection_by_year,
)


def test_plot_collection_by_year_returns_figure():
    coll = pl.DataFrame(
        {"game_id": [1, 2, 3, 4], "owned": [True, True, False, True]}
    )
    games = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "year_published": [2018, 2019, 2020, 2018],
        }
    )
    fig = plot_collection_by_year(coll, games)
    assert hasattr(fig, "data")


def test_plot_collection_by_category_returns_figure():
    coll = pl.DataFrame({"game_id": [1, 2, 3], "owned": [True, True, True]})
    games = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "category_strategy": [1, 0, 1],
            "category_party": [0, 1, 0],
            "designer_uwe_rosenberg": [1, 1, 0],
        }
    )
    fig = plot_collection_by_category(coll, games, top_n=10)
    assert hasattr(fig, "data")


def test_collection_datatable_returns_pandas():
    coll = pl.DataFrame(
        {
            "game_id": [1, 2],
            "game_name": ["A", "B"],
            "owned": [True, True],
            "user_rating": [9.0, 7.5],
        }
    )
    games = pl.DataFrame({"game_id": [1, 2], "year_published": [2020, 2021]})
    out = collection_datatable(coll, games)
    assert isinstance(out, pd.DataFrame)
    assert "game_id" in out.columns


from src.collection.viz import plot_partial_effects_by_group


def test_plot_partial_effects_by_group_returns_dict():
    fi = pd.DataFrame(
        {
            "feature": [
                "category_strategy",
                "category_party",
                "designer_uwe_rosenberg",
                "publisher_z_man_games",
            ],
            "value": [0.5, -0.2, 0.7, 0.1],
            "abs_value": [0.5, 0.2, 0.7, 0.1],
        }
    )
    plots = plot_partial_effects_by_group(fi)
    assert isinstance(plots, dict)
    # At least one of the known groups returned a plot
    assert "Categories" in plots or "Designers" in plots
