"""Shared fixtures for tests/reports.

Builds a minimal but realistic on-disk artifact tree under a tmp_path
that mirrors `models/collections/{username}/{outcome}/...`. Use the
`fixture_collection_root` fixture in tests that exercise the loader.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _identity_pipeline() -> Pipeline:
    """A trivial Pipeline with a `preprocessor` and `model` step.

    The preprocessor is FunctionTransformer with no func (sklearn default
    is identity), which pickles cleanly without a local-symbol reference.
    The model is a fitted DummyClassifier with a fake `coef_` attached so
    importance extraction has something to read.
    """
    preprocessor = FunctionTransformer(validate=False)
    model = DummyClassifier(strategy="constant", constant=0)
    X = np.zeros((4, 3))
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    model.coef_ = np.array([[0.5, -0.2, 0.1]])
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


@pytest.fixture
def fixture_collection_root(tmp_path: Path) -> Path:
    """Build `tmp_path/collections/phenrickson/own/...` with one finalized
    candidate (`logistic_row_norm`) plus canonical splits."""
    root = tmp_path / "collections"
    user_dir = root / "phenrickson"
    user_dir.mkdir(parents=True)

    coll_dir = user_dir / "collection"
    coll_dir.mkdir()
    pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "game_name": ["A", "B", "C"],
            "user_rating": [8.0, 7.5, 9.0],
            "owned": [True, True, False],
        }
    ).write_parquet(coll_dir / "latest.parquet")

    splits_dir = user_dir / "own" / "_splits" / "v1"
    splits_dir.mkdir(parents=True)
    train_df = pl.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "year_published": [2018, 2019, 2020, 2021, 2022],
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feat_b": [1, 0, 1, 0, 1],
            "feat_c": [0.0, 0.5, 1.0, 0.5, 0.0],
            "label": [1, 0, 1, 0, 1],
        }
    )
    train_df.write_parquet(splits_dir / "train.parquet")
    train_df.head(2).write_parquet(splits_dir / "validation.parquet")
    train_df.tail(2).write_parquet(splits_dir / "test.parquet")

    cand_dir = user_dir / "own" / "logistic_row_norm" / "v1"
    cand_dir.mkdir(parents=True)
    pipeline = _identity_pipeline()
    (cand_dir / "model.pkl").write_bytes(pickle.dumps(pipeline))
    (cand_dir / "finalized.pkl").write_bytes(pickle.dumps(pipeline))
    (cand_dir / "registration.json").write_text(
        json.dumps(
            {
                "candidate": "logistic_row_norm",
                "version": 1,
                "splits_version": 1,
                "finalize_through": 2024,
                "finalized_at": "2026-05-01T00:00:00",
                "task": "classification",
                "metrics": {"roc_auc": 0.85, "pr_auc": 0.6, "log_loss": 0.4},
                "val_metrics": {"roc_auc": 0.82, "pr_auc": 0.55},
                "oof_metrics": {"overall": {"roc_auc": 0.8}},
                "threshold": 0.5,
            }
        )
    )
    (cand_dir / "threshold.json").write_text(json.dumps({"threshold": 0.5}))
    pl.DataFrame(
        {"feature": ["feat_a", "feat_b", "feat_c"], "value": [0.5, -0.2, 0.1]}
    ).with_columns(pl.col("value").abs().alias("abs_value")).write_parquet(
        cand_dir / "feature_importance.parquet"
    )

    preds_dir = cand_dir / "predictions"
    preds_dir.mkdir()
    preds_df = pl.DataFrame(
        {
            "game_id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "year_published": [2020, 2021, 2022],
            "proba": [0.9, 0.4, 0.7],
            "label": [True, False, True],
        }
    )
    preds_df.write_parquet(preds_dir / "oof.parquet")
    preds_df.write_parquet(preds_dir / "val.parquet")
    preds_df.write_parquet(preds_dir / "test.parquet")

    return root
