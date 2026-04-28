"""Train collection models (classification or regression) per outcome.

Dispatches on OutcomeDefinition.task. Reuses existing preprocessing and
tuning infrastructure from src.models.training.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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
    tune_model_cv,
    select_X_y,
)
from src.models.outcomes.hurdle import find_optimal_threshold

logger = logging.getLogger(__name__)


@dataclass
class ClassificationModelConfig:
    model_type: str = "logistic"  # 'lightgbm' | 'catboost' | 'logistic'
    use_sample_weights: bool = False
    handle_imbalance: str = "scale_pos_weight"  # 'scale_pos_weight' | 'none'
    threshold_optimization_metric: str = "f2"  # 'f1' | 'f2' | 'precision' | 'recall'
    preprocessor_type: str = "auto"
    preprocessor_kwargs: Dict[str, Any] = field(default_factory=dict)
    tuning_metric: str = "log_loss"
    patience: int = 10


@dataclass
class RegressionModelConfig:
    model_type: str = "lightgbm"  # 'lightgbm' | 'catboost'
    preprocessor_type: str = "auto"
    preprocessor_kwargs: Dict[str, Any] = field(default_factory=dict)
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

    def build_pipeline(self) -> Pipeline:
        """Build the sklearn Pipeline (preprocessor + unfit model) for inspection.

        Useful for seeing the preprocessor steps and model class before
        committing to a training run. Does not fit anything.
        """
        if self.outcome.task == "classification":
            cfg = self.classification_config
            if cfg.model_type not in CLASSIFIER_MAPPING:
                raise ValueError(
                    f"Unknown classification model_type: {cfg.model_type!r}. "
                    f"Choose from {list(CLASSIFIER_MAPPING.keys())}"
                )
            model = CLASSIFIER_MAPPING[cfg.model_type]()
            preprocessor = create_preprocessing_pipeline(
                model_type=cfg.preprocessor_type,
                model_name=cfg.model_type,
                **cfg.preprocessor_kwargs,
            )
            return Pipeline([("preprocessor", preprocessor), ("model", model)])

        if self.outcome.task == "regression":
            cfg = self.regression_config
            if cfg.model_type not in REGRESSOR_MAPPING:
                raise ValueError(
                    f"Unknown regression model_type: {cfg.model_type!r}. "
                    f"Choose from {list(REGRESSOR_MAPPING.keys())}"
                )
            model = REGRESSOR_MAPPING[cfg.model_type]()
            preprocessor = create_preprocessing_pipeline(
                model_type=cfg.preprocessor_type,
                model_name=cfg.model_type,
                **cfg.preprocessor_kwargs,
            )
            return Pipeline([("preprocessor", preprocessor), ("model", model)])

        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    def tune(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        """Holdout hyperparameter tuning.

        Searches the model's param grid by training each candidate on
        ``train_df`` and scoring on ``val_df``. Uses the existing patience
        early-stopping loop in :func:`tune_model`.

        Returns ``(fitted_pipeline, best_params, tuning_results)`` — the
        results frame has one row per *evaluated* config (early-stopped
        configs are not included), sorted best-first.
        """
        if self.outcome.task == "classification":
            return self._tune_classification(train_df, val_df)
        if self.outcome.task == "regression":
            return self._tune_regression(train_df, val_df)
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    def tune_cv(
        self, train_df: pl.DataFrame, cv_folds: int = 5
    ) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        """K-fold cross-validation hyperparameter tuning.

        Searches the full param grid by k-fold CV on ``train_df`` only.
        Validation/test sets are not used. Slower than :meth:`tune` but
        avoids reusing the val set for both selection and final reporting.

        Returns ``(fitted_pipeline, best_params, tuning_results)``.
        """
        if self.outcome.task == "classification":
            return self._tune_classification_cv(train_df, cv_folds=cv_folds)
        if self.outcome.task == "regression":
            return self._tune_regression_cv(train_df, cv_folds=cv_folds)
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    def train(
        self,
        train_df: pl.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Fit a pipeline on ``train_df`` with the given hyperparameters.

        No tuning. ``params`` uses the same ``model__<name>`` key shape that
        :meth:`tune` / :meth:`tune_cv` return as ``best_params``. Pass
        ``None`` (default) to fit with the model's defaults.
        """
        pipeline = self.build_pipeline()
        if params:
            pipeline.named_steps["model"].set_params(
                **{k.replace("model__", ""): v for k, v in params.items()}
            )
        X, y = self._prepare(train_df)
        pipeline.fit(X, y)
        return pipeline

    def evaluate(
        self,
        pipeline: Pipeline,
        df: pl.DataFrame,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate a fitted pipeline on ``df``.

        For classification, ``threshold`` is the decision threshold for turning
        probabilities into hard predictions. If ``None``, uses 0.5.
        Pass the threshold returned by :meth:`find_threshold` to evaluate at
        the optimized cutoff; pass the same frozen threshold to compare val
        and test fairly.

        For regression, ``threshold`` is ignored.
        """
        if self.outcome.task == "classification":
            return self._evaluate_classification(pipeline, df, threshold=threshold)
        if self.outcome.task == "regression":
            return self._evaluate_regression(pipeline, df)
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    def predict_with_labels(
        self,
        pipeline: Pipeline,
        df: pl.DataFrame,
        threshold: Optional[float] = None,
    ) -> pl.DataFrame:
        """Return ``df`` with prediction columns appended.

        Classification: appends ``proba`` (P(label=1)) and ``pred`` (hard
        prediction at ``threshold``, default 0.5).
        Regression: appends ``pred`` (predicted value); ``threshold`` ignored.

        Useful for inspecting predictions row-by-row alongside the true
        label — sort by ``proba``, group by any column you like, find
        misclassifications, etc.
        """
        X, _ = self._prepare(df)
        if self.outcome.task == "classification":
            proba = pipeline.predict_proba(X)[:, 1]
            t = 0.5 if threshold is None else float(threshold)
            return df.with_columns([
                pl.Series("proba", proba),
                pl.Series("pred", (proba >= t).astype(int)),
            ])
        if self.outcome.task == "regression":
            pred = pipeline.predict(X)
            return df.with_columns(pl.Series("pred", pred))
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    # --- classification path ---

    def _tune_classification(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        cfg = self.classification_config
        param_grid = dict(CLASSIFIER_PARAM_GRIDS[cfg.model_type])
        X_train, y_train = self._prepare(train_df)
        X_val, y_val = self._prepare(val_df)
        return tune_model(
            pipeline=self.build_pipeline(),
            train_X=X_train,
            train_y=y_train,
            tune_X=X_val,
            tune_y=y_val,
            param_grid=param_grid,
            metric=cfg.tuning_metric,
            patience=cfg.patience,
        )

    def _tune_classification_cv(
        self, train_df: pl.DataFrame, cv_folds: int
    ) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        cfg = self.classification_config
        param_grid = dict(CLASSIFIER_PARAM_GRIDS[cfg.model_type])
        X_train, y_train = self._prepare(train_df)
        return tune_model_cv(
            pipeline=self.build_pipeline(),
            X=X_train,
            y=y_train,
            param_grid=param_grid,
            metric=cfg.tuning_metric,
            cv_folds=cv_folds,
            task="classification",
        )

    def _evaluate_classification(
        self,
        pipeline: Pipeline,
        df: pl.DataFrame,
        threshold: Optional[float] = None,
    ) -> Dict[str, float]:
        X, y = self._prepare(df)
        proba = pipeline.predict_proba(X)[:, 1]
        t = 0.5 if threshold is None else float(threshold)
        preds = (proba >= t).astype(int)
        return {
            "threshold": t,
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "f2": fbeta_score(y, preds, beta=2, zero_division=0),
            "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
            "pr_auc": average_precision_score(y, proba) if len(set(y)) > 1 else float("nan"),
            "log_loss": log_loss(y, proba, labels=[0, 1]) if len(set(y)) > 1 else float("nan"),
        }

    def evaluate_stratified(
        self,
        pipeline: Pipeline,
        df: pl.DataFrame,
        threshold: Optional[float] = None,
        bins: Sequence[Tuple[Optional[int], Optional[int]]] = (
            (None, 25),
            (25, 100),
            (100, None),
        ),
        bin_column: str = "users_rated",
    ) -> pd.DataFrame:
        """Evaluate ``df`` stratified by a numeric bin column (default
        ``users_rated``).

        ``bins`` is a sequence of ``(low, high)`` tuples with inclusive-low,
        exclusive-high semantics; ``None`` on either side means unbounded.
        Default bins: <25, 25-99, 100+.

        Returns a pandas DataFrame with one row per bin plus an ``all`` row,
        and columns for each metric in :meth:`evaluate`.
        """
        if self.outcome.task != "classification":
            raise ValueError(
                "evaluate_stratified currently supports classification only"
            )
        if bin_column not in df.columns:
            raise ValueError(
                f"bin_column {bin_column!r} missing from df; "
                f"available columns start with: {df.columns[:10]}"
            )

        rows = []
        for low, high in bins:
            label = self._bin_label(low, high)
            subset = df
            if low is not None:
                subset = subset.filter(pl.col(bin_column) >= low)
            if high is not None:
                subset = subset.filter(pl.col(bin_column) < high)
            row = {"bin": label, "n_rows": subset.height}
            if subset.height == 0:
                rows.append(row)
                continue
            metrics = self._evaluate_classification(
                pipeline, subset, threshold=threshold
            )
            row.update(metrics)
            row["n_pos"] = int(subset.filter(pl.col("label") == True).height)
            rows.append(row)

        # Overall row for reference.
        overall = {"bin": "all", "n_rows": df.height}
        if df.height:
            overall.update(
                self._evaluate_classification(pipeline, df, threshold=threshold)
            )
            overall["n_pos"] = int(df.filter(pl.col("label") == True).height)
        rows.append(overall)
        return pd.DataFrame(rows)

    @staticmethod
    def _bin_label(low: Optional[int], high: Optional[int]) -> str:
        if low is None and high is not None:
            return f"<{high}"
        if low is not None and high is None:
            return f"{low}+"
        return f"{low}-{high - 1}"

    def transform_features(
        self, pipeline: Pipeline, df: pl.DataFrame
    ) -> pd.DataFrame:
        """Run the fitted preprocessor on ``df`` and return a pandas DataFrame.

        Uses whatever column names the preprocessor's output carries; falls back
        to generic ``f0..fN-1`` if the underlying transform returns a bare array.
        """
        preprocessor = pipeline.named_steps["preprocessor"]
        X, _ = self._prepare(df)
        transformed = preprocessor.transform(X)
        if hasattr(transformed, "columns"):
            return pd.DataFrame(transformed)  # already has columns
        names = [f"f{i}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=names, index=X.index)

    def feature_names(self, pipeline: Pipeline, df: pl.DataFrame) -> list:
        """Return the post-preprocessing feature names produced when ``df`` is
        transformed. Recovered by actually running ``transform`` on a small
        slice, since ``Pipeline.get_feature_names_out()`` is unreliable on
        this stack.
        """
        return list(self.transform_features(pipeline, df.head(5)).columns)

    def feature_importance(
        self, pipeline: Pipeline, df: pl.DataFrame
    ) -> pd.DataFrame:
        """Return a feature-importance DataFrame for the fitted pipeline.

        Dispatches on what the underlying model exposes:
        - tree models → ``feature_importances_`` as ``value``
        - linear models → ``coef_`` as ``value`` (flattened)

        Columns: ``feature``, ``value``, ``abs_value`` (sorted by abs_value desc).

        ``df`` is any DataFrame with the same schema as training — used only to
        recover feature names.
        """
        model_step = pipeline.named_steps["model"]
        names = self.feature_names(pipeline, df)

        if hasattr(model_step, "feature_importances_"):
            values = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            values = model_step.coef_.ravel()
        else:
            raise ValueError(
                f"Model {type(model_step).__name__} has neither "
                "feature_importances_ nor coef_"
            )

        if len(values) != len(names):
            raise ValueError(
                f"Length mismatch: {len(values)} importance values vs "
                f"{len(names)} feature names"
            )

        out = pd.DataFrame({
            "feature": names,
            "value": values,
        })
        out["abs_value"] = out["value"].abs()
        return out.sort_values("abs_value", ascending=False).reset_index(drop=True)

    def top_games(
        self,
        pipeline: Pipeline,
        df: pl.DataFrame,
        n: int = 25,
        exclude_game_ids: Optional[Iterable[int]] = None,
        include_columns: Sequence[str] = ("game_id", "name", "year_published"),
    ) -> pl.DataFrame:
        """Score every row in ``df`` and return the top-``n`` by predicted score.

        Classification models return a probability in ``score``; regression
        models return the predicted label.

        Args:
            pipeline: Fitted sklearn Pipeline.
            df: Rows to score. Must have columns matching training features
                plus whatever is listed in ``include_columns``.
            n: Number of rows to keep after sorting.
            exclude_game_ids: game_ids to drop before ranking (e.g., games
                the user already owns).
            include_columns: Columns from ``df`` to surface alongside the
                score. Missing columns are silently skipped.

        Returns:
            Polars DataFrame with the ``include_columns`` present in ``df``
            plus a ``score`` column, sorted by score descending, top ``n`` rows.
        """
        if df.height == 0:
            return df.head(0).with_columns(pl.lit(None).cast(pl.Float64).alias("score"))

        # Don't require a `label` column — this is scoring, not training.
        X = df.drop("label") if "label" in df.columns else df
        X = X.to_pandas()
        if self.outcome.task == "classification":
            score = pipeline.predict_proba(X)[:, 1]
        else:
            score = pipeline.predict(X)

        scored = df.with_columns(pl.Series("score", score))

        if exclude_game_ids is not None:
            excluded = list(set(exclude_game_ids))
            if excluded and "game_id" in scored.columns:
                scored = scored.filter(~pl.col("game_id").is_in(excluded))

        kept = [c for c in include_columns if c in scored.columns]
        return scored.select(kept + ["score"]).sort("score", descending=True).head(n)

    def find_threshold(self, pipeline: Pipeline, val_df: pl.DataFrame) -> float:
        """Return the probability threshold that maximises the configured metric on val.

        `find_optimal_threshold` returns a dict including diagnostic scores; we only
        surface the threshold value here so callers (pipeline, storage, CLI) can
        treat it as a plain float.
        """
        if self.outcome.task != "classification":
            raise ValueError("find_threshold is only meaningful for classification outcomes")
        X, y = self._prepare(val_df)
        proba = pipeline.predict_proba(X)[:, 1]
        result = find_optimal_threshold(
            y, proba, metric=self.classification_config.threshold_optimization_metric
        )
        return float(result["threshold"])

    # --- regression path ---

    def _tune_regression(
        self, train_df: pl.DataFrame, val_df: pl.DataFrame
    ) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        cfg = self.regression_config
        param_grid = dict(REGRESSOR_PARAM_GRIDS[cfg.model_type])
        X_train, y_train = self._prepare(train_df)
        X_val, y_val = self._prepare(val_df)
        return tune_model(
            pipeline=self.build_pipeline(),
            train_X=X_train,
            train_y=y_train,
            tune_X=X_val,
            tune_y=y_val,
            param_grid=param_grid,
            metric=cfg.tuning_metric,
            patience=cfg.patience,
        )

    def _tune_regression_cv(
        self, train_df: pl.DataFrame, cv_folds: int
    ) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
        cfg = self.regression_config
        param_grid = dict(REGRESSOR_PARAM_GRIDS[cfg.model_type])
        X_train, y_train = self._prepare(train_df)
        return tune_model_cv(
            pipeline=self.build_pipeline(),
            X=X_train,
            y=y_train,
            param_grid=param_grid,
            metric=cfg.tuning_metric,
            cv_folds=cv_folds,
            task="regression",
        )

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
