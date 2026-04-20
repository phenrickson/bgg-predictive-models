"""Train collection models (classification or regression) per outcome.

Dispatches on OutcomeDefinition.task. Reuses existing preprocessing and
tuning infrastructure from src.models.training.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from src.collection.outcomes import OutcomeDefinition
from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    select_X_y,
)
from src.models.outcomes.hurdle import find_optimal_threshold

logger = logging.getLogger(__name__)


@dataclass
class ClassificationModelConfig:
    model_type: str = "lightgbm"  # 'lightgbm' | 'catboost' | 'logistic'
    use_sample_weights: bool = False
    handle_imbalance: str = "scale_pos_weight"  # 'scale_pos_weight' | 'none'
    threshold_optimization_metric: str = "f2"  # 'f1' | 'f2' | 'precision' | 'recall'
    preprocessor_type: str = "auto"
    tuning_metric: str = "log_loss"
    patience: int = 10


@dataclass
class RegressionModelConfig:
    model_type: str = "lightgbm"  # 'lightgbm' | 'catboost'
    preprocessor_type: str = "auto"
    tuning_metric: str = "rmse"  # 'rmse' | 'mae'
    patience: int = 10


CLASSIFIER_MAPPING = {
    "logistic": lambda: LogisticRegression(max_iter=4000),
    "lightgbm": lambda: lgb.LGBMClassifier(objective="binary", verbose=-1),
    "catboost": lambda: CatBoostClassifier(verbose=0),
}

REGRESSOR_MAPPING = {
    "lightgbm": lambda: lgb.LGBMRegressor(objective="regression", verbose=-1),
    "catboost": lambda: CatBoostRegressor(verbose=0),
}

CLASSIFIER_PARAM_GRIDS = {
    "logistic": {"model__C": [0.001, 0.01, 0.1, 1.0], "model__penalty": ["l2"]},
    "lightgbm": {
        "model__n_estimators": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__num_leaves": [15, 31],
        "model__min_child_samples": [20],
        "model__scale_pos_weight": [1, 5, 10],
    },
    "catboost": {
        "model__iterations": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__depth": [4, 6],
        "model__scale_pos_weight": [1, 5, 10],
    },
}

REGRESSOR_PARAM_GRIDS = {
    "lightgbm": {
        "model__n_estimators": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__max_depth": [3, 5, 7],
        "model__num_leaves": [15, 31],
        "model__min_child_samples": [20],
    },
    "catboost": {
        "model__iterations": [500],
        "model__learning_rate": [0.01, 0.05],
        "model__depth": [4, 6],
    },
}


class CollectionModel:
    """Train one model for one outcome for one user.

    Dispatches on OutcomeDefinition.task. Callers pass pre-split dataframes
    (train/val/test) with a 'label' column.
    """

    def __init__(
        self,
        username: str,
        outcome: OutcomeDefinition,
        classification_config: Optional[ClassificationModelConfig] = None,
        regression_config: Optional[RegressionModelConfig] = None,
    ):
        self.username = username
        self.outcome = outcome
        self.classification_config = classification_config or ClassificationModelConfig()
        self.regression_config = regression_config or RegressionModelConfig()
        logger.info(
            f"CollectionModel init: user={username!r} outcome={outcome.name!r} task={outcome.task!r}"
        )

    def train(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        if self.outcome.task == "classification":
            return self._train_classification(train_df, val_df)
        if self.outcome.task == "regression":
            return self._train_regression(train_df, val_df)
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    def evaluate(self, pipeline: Pipeline, df: pl.DataFrame) -> Dict[str, float]:
        if self.outcome.task == "classification":
            return self._evaluate_classification(pipeline, df)
        if self.outcome.task == "regression":
            return self._evaluate_regression(pipeline, df)
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    # --- classification path ---

    def _train_classification(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        cfg = self.classification_config
        if cfg.model_type not in CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unknown classification model_type: {cfg.model_type!r}. "
                f"Choose from {list(CLASSIFIER_MAPPING.keys())}"
            )
        model = CLASSIFIER_MAPPING[cfg.model_type]()
        param_grid = dict(CLASSIFIER_PARAM_GRIDS[cfg.model_type])

        X_train, y_train = self._prepare(train_df)
        X_val, y_val = self._prepare(val_df)

        preprocessor = create_preprocessing_pipeline(model_type=cfg.preprocessor_type)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        best_pipeline, best_params = tune_model(
            pipeline=pipeline,
            train_X=X_train,
            train_y=y_train,
            tune_X=X_val,
            tune_y=y_val,
            param_grid=param_grid,
            metric=cfg.tuning_metric,
            patience=cfg.patience,
        )
        return best_pipeline, best_params

    def _evaluate_classification(self, pipeline: Pipeline, df: pl.DataFrame) -> Dict[str, float]:
        X, y = self._prepare(df)
        proba = pipeline.predict_proba(X)[:, 1]
        preds = pipeline.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "f2": fbeta_score(y, preds, beta=2, zero_division=0),
            "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
            "log_loss": log_loss(y, proba, labels=[0, 1]) if len(set(y)) > 1 else float("nan"),
        }

    def find_threshold(self, pipeline: Pipeline, val_df: pl.DataFrame) -> float:
        if self.outcome.task != "classification":
            raise ValueError("find_threshold is only meaningful for classification outcomes")
        X, y = self._prepare(val_df)
        proba = pipeline.predict_proba(X)[:, 1]
        return find_optimal_threshold(
            y, proba, metric=self.classification_config.threshold_optimization_metric
        )

    # --- regression path ---

    def _train_regression(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        cfg = self.regression_config
        if cfg.model_type not in REGRESSOR_MAPPING:
            raise ValueError(
                f"Unknown regression model_type: {cfg.model_type!r}. "
                f"Choose from {list(REGRESSOR_MAPPING.keys())}"
            )
        model = REGRESSOR_MAPPING[cfg.model_type]()
        param_grid = dict(REGRESSOR_PARAM_GRIDS[cfg.model_type])

        X_train, y_train = self._prepare(train_df)
        X_val, y_val = self._prepare(val_df)

        preprocessor = create_preprocessing_pipeline(model_type=cfg.preprocessor_type)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        best_pipeline, best_params = tune_model(
            pipeline=pipeline,
            train_X=X_train,
            train_y=y_train,
            tune_X=X_val,
            tune_y=y_val,
            param_grid=param_grid,
            metric=cfg.tuning_metric,
            patience=cfg.patience,
        )
        return best_pipeline, best_params

    def _evaluate_regression(self, pipeline: Pipeline, df: pl.DataFrame) -> Dict[str, float]:
        X, y = self._prepare(df)
        preds = pipeline.predict(X)
        mse = mean_squared_error(y, preds)
        return {
            "rmse": mse ** 0.5,
            "mae": mean_absolute_error(y, preds),
            "r2": r2_score(y, preds),
        }

    # --- shared helpers ---

    def _prepare(self, df: pl.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract X, y from a labeled dataframe. Target column is 'label'."""
        return select_X_y(df, y_column="label")
