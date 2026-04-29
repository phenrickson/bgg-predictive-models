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

        # Fit-time state. Populated by train / tune / tune_cv (pipeline) and
        # find_threshold (threshold). Helpers read these instead of taking
        # the pipeline as an argument.
        self.fitted_pipeline: Optional[Pipeline] = None
        self.threshold: Optional[float] = None
        self._feature_names: Optional[list] = None

        # Set by finalize() — same hyperparams as fitted_pipeline, refit on
        # the union of train/val/test up through `finalize_through`. This is
        # the model used to score upcoming releases.
        self.finalized_pipeline: Optional[Pipeline] = None
        self.finalize_through: Optional[int] = None

        logger.info(
            f"CollectionModel init: user={username!r} outcome={outcome.name!r} task={outcome.task!r}"
        )

    def _require_fitted(self, use_finalized: bool = False) -> Pipeline:
        if use_finalized:
            if self.finalized_pipeline is None:
                raise RuntimeError(
                    "Model has not been finalized. Call finalize() first."
                )
            return self.finalized_pipeline
        if self.fitted_pipeline is None:
            raise RuntimeError(
                "Model is not fit. Call train(), tune(), or tune_cv() first."
            )
        return self.fitted_pipeline

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
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Holdout hyperparameter tuning.

        Searches the model's param grid by training each candidate on
        ``train_df`` and scoring on ``val_df``. Uses the existing patience
        early-stopping loop in :func:`tune_model`. The best pipeline is
        stashed on ``self.fitted_pipeline``; helpers read it from there.

        Returns ``(best_params, tuning_results)`` — the results frame has one
        row per *evaluated* config (early-stopped configs are not included),
        sorted best-first.
        """
        if self.outcome.task == "classification":
            pipeline, best_params, tuning_results = self._tune_classification(
                train_df, val_df
            )
        elif self.outcome.task == "regression":
            pipeline, best_params, tuning_results = self._tune_regression(
                train_df, val_df
            )
        else:
            raise ValueError(f"Unsupported task: {self.outcome.task!r}")
        self._set_fitted(pipeline, train_df)
        return best_params, tuning_results

    def tune_cv(
        self, train_df: pl.DataFrame, cv_folds: int = 5
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """K-fold cross-validation hyperparameter tuning.

        Searches the full param grid by k-fold CV on ``train_df`` only.
        Validation/test sets are not used. Slower than :meth:`tune` but
        avoids reusing the val set for both selection and final reporting.
        The best pipeline is stashed on ``self.fitted_pipeline``.

        Returns ``(best_params, tuning_results)``.
        """
        if self.outcome.task == "classification":
            pipeline, best_params, tuning_results = self._tune_classification_cv(
                train_df, cv_folds=cv_folds
            )
        elif self.outcome.task == "regression":
            pipeline, best_params, tuning_results = self._tune_regression_cv(
                train_df, cv_folds=cv_folds
            )
        else:
            raise ValueError(f"Unsupported task: {self.outcome.task!r}")
        self._set_fitted(pipeline, train_df)
        return best_params, tuning_results

    def train(
        self,
        train_df: pl.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit a pipeline on ``train_df`` with the given hyperparameters.

        No tuning. ``params`` uses the same ``model__<name>`` key shape that
        :meth:`tune` / :meth:`tune_cv` return as ``best_params``. Pass
        ``None`` (default) to fit with the model's defaults. The fitted
        pipeline is stashed on ``self.fitted_pipeline``.
        """
        pipeline = self.build_pipeline()
        if params:
            pipeline.named_steps["model"].set_params(
                **{k.replace("model__", ""): v for k, v in params.items()}
            )
        X, y = self._prepare(train_df)
        pipeline.fit(X, y)
        self._set_fitted(pipeline, train_df)

    def finalize(
        self,
        df: pl.DataFrame,
        finalize_through: int,
        time_column: str = "year_published",
        downsample_ratio: Optional[int] = None,
        protect_min_ratings: int = 25,
    ) -> None:
        """Refit the tuned model on rows through ``finalize_through``.

        Workflow: tune on train, validate on val, evaluate on test, then call
        ``finalize`` with the union of train/val/test. The refit uses the same
        hyperparameters as ``fitted_pipeline`` (no re-tuning, no threshold
        re-search) and is stashed on ``self.finalized_pipeline``. Helpers
        accept ``use_finalized=True`` to score with it; pass it to
        :meth:`top_games` when ranking upcoming releases.

        Args:
            df: Combined train + val + test (or whatever you trust). Will be
                filtered to ``df[time_column] <= finalize_through``.
            finalize_through: Last year (inclusive) to include.
            time_column: Year column to filter on. Defaults to
                ``year_published``.
            downsample_ratio: If set and the outcome is classification, apply
                the same negative-downsampling used during training so the
                refit class balance matches what hyperparameters were tuned
                for. Pass the candidate's
                ``downsample_negatives_ratio`` here.
            protect_min_ratings: Threshold for the protected pool when
                downsampling — games with at least this many ratings are
                kept regardless. Mirrors the train-time default.
        """
        if self.fitted_pipeline is None:
            raise RuntimeError(
                "Cannot finalize before fitting. Call train(), tune(), or "
                "tune_cv() first."
            )
        if time_column not in df.columns:
            raise ValueError(
                f"time_column {time_column!r} missing from df; "
                f"columns start with: {df.columns[:10]}"
            )

        keep = df.filter(pl.col(time_column) <= finalize_through)
        if keep.height == 0:
            raise ValueError(
                f"No rows with {time_column} <= {finalize_through} in df "
                f"(df has {df.height} rows)"
            )

        if downsample_ratio is not None and self.outcome.task == "classification":
            from src.collection.collection_split import downsample_negatives

            before = keep.height
            keep = downsample_negatives(
                keep, ratio=downsample_ratio, protect_min_ratings=protect_min_ratings
            )
            logger.info(
                f"Finalize downsampled negatives: {before} -> {keep.height} rows "
                f"(ratio={downsample_ratio}, protect_min_ratings={protect_min_ratings})"
            )

        logger.info(
            f"Finalizing through {finalize_through}: refitting on "
            f"{keep.height}/{df.height} rows"
        )

        # Mirror the fitted pipeline's structure with the same model
        # hyperparams. Fresh pipeline so the existing fitted_pipeline keeps
        # its train-only state (callers may still want it for diagnostics).
        params = self.fitted_pipeline.named_steps["model"].get_params(deep=False)
        pipeline = self.build_pipeline()
        pipeline.named_steps["model"].set_params(**params)
        X, y = self._prepare(keep)
        pipeline.fit(X, y)

        self.finalized_pipeline = pipeline
        self.finalize_through = finalize_through

    def _set_fitted(self, pipeline: Pipeline, train_df: pl.DataFrame) -> None:
        """Stash the fitted pipeline and cache feature names. Helpers read
        from ``self.fitted_pipeline`` and ``self._feature_names``.
        """
        self.fitted_pipeline = pipeline
        # Cache feature names by transforming a small slice; recovers names
        # even when sklearn's get_feature_names_out is unreliable on this stack.
        preprocessor = pipeline.named_steps["preprocessor"]
        X, _ = self._prepare(train_df.head(5))
        transformed = preprocessor.transform(X)
        if hasattr(transformed, "columns"):
            self._feature_names = list(transformed.columns)
        else:
            self._feature_names = [f"f{i}" for i in range(transformed.shape[1])]

    def evaluate(
        self,
        df: pl.DataFrame,
        threshold: Optional[float] = None,
        use_finalized: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the fitted model on ``df``.

        Classification: ``threshold`` defaults to ``self.threshold`` (set by
        :meth:`find_threshold`); falls back to 0.5 if neither is set.
        Regression: ``threshold`` is ignored.

        ``use_finalized=True`` evaluates with the finalized pipeline. Note
        that evaluating the finalized model on data it was refit on is not
        a generalization measure — use this only on truly held-out frames.
        """
        pipeline = self._require_fitted(use_finalized=use_finalized)
        if self.outcome.task == "classification":
            t = self.threshold if threshold is None else threshold
            return self._evaluate_classification(pipeline, df, threshold=t)
        if self.outcome.task == "regression":
            return self._evaluate_regression(pipeline, df)
        raise ValueError(f"Unsupported task: {self.outcome.task!r}")

    def predict_with_labels(
        self,
        df: pl.DataFrame,
        threshold: Optional[float] = None,
        use_finalized: bool = False,
    ) -> pl.DataFrame:
        """Return ``df`` with prediction columns appended.

        Classification: appends ``proba`` and ``pred`` (hard prediction at
        ``threshold``; defaults to ``self.threshold``, falling back to 0.5).
        Regression: appends ``pred``; ``threshold`` ignored.

        Pass ``use_finalized=True`` to predict with the finalized pipeline.
        """
        pipeline = self._require_fitted(use_finalized=use_finalized)
        X, _ = self._prepare(df)
        if self.outcome.task == "classification":
            proba = pipeline.predict_proba(X)[:, 1]
            t_pick = threshold if threshold is not None else self.threshold
            t = 0.5 if t_pick is None else float(t_pick)
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
        df: pl.DataFrame,
        threshold: Optional[float] = None,
        bins: Sequence[Tuple[Optional[int], Optional[int]]] = (
            (None, 25),
            (25, 100),
            (100, None),
        ),
        bin_column: str = "users_rated",
        use_finalized: bool = False,
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
        pipeline = self._require_fitted(use_finalized=use_finalized)
        t = self.threshold if threshold is None else threshold

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
                pipeline, subset, threshold=t
            )
            row.update(metrics)
            row["n_pos"] = int(subset.filter(pl.col("label") == True).height)
            rows.append(row)

        # Overall row for reference.
        overall = {"bin": "all", "n_rows": df.height}
        if df.height:
            overall.update(
                self._evaluate_classification(pipeline, df, threshold=t)
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

    def transform_features(self, df: pl.DataFrame) -> pd.DataFrame:
        """Run the fitted preprocessor on ``df`` and return a pandas DataFrame.

        Uses whatever column names the preprocessor's output carries; falls back
        to generic ``f0..fN-1`` if the underlying transform returns a bare array.
        """
        pipeline = self._require_fitted()
        preprocessor = pipeline.named_steps["preprocessor"]
        X, _ = self._prepare(df)
        transformed = preprocessor.transform(X)
        if hasattr(transformed, "columns"):
            return pd.DataFrame(transformed)  # already has columns
        names = [f"f{i}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=names, index=X.index)

    def preview_features(self, df: pl.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor on ``df`` and return the post-preprocessing
        feature matrix. The model is not fit.

        Useful for inspecting how different ``preprocessor_kwargs`` change the
        feature set without paying the cost of training the model.
        """
        preprocessor = self.build_pipeline().named_steps["preprocessor"]
        X, y = self._prepare(df)
        transformed = preprocessor.fit_transform(X, y)
        if hasattr(transformed, "columns"):
            return pd.DataFrame(transformed)
        names = [f"f{i}" for i in range(transformed.shape[1])]
        return pd.DataFrame(transformed, columns=names, index=X.index)

    @property
    def feature_names(self) -> list:
        """Post-preprocessing feature names captured at fit time."""
        if self._feature_names is None:
            raise RuntimeError(
                "Model is not fit. Call train(), tune(), or tune_cv() first."
            )
        return list(self._feature_names)

    def feature_importance(self) -> pd.DataFrame:
        """Return a feature-importance DataFrame for the fitted pipeline.

        Dispatches on what the underlying model exposes:
        - tree models → ``feature_importances_`` as ``value``
        - linear models → ``coef_`` as ``value`` (flattened)

        Columns: ``feature``, ``value``, ``abs_value`` (sorted by abs_value desc).
        """
        pipeline = self._require_fitted()
        model_step = pipeline.named_steps["model"]
        names = self.feature_names

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
        df: pl.DataFrame,
        n: int = 25,
        exclude_game_ids: Optional[Iterable[int]] = None,
        include_columns: Sequence[str] = ("game_id", "name", "year_published"),
        use_finalized: bool = False,
    ) -> pl.DataFrame:
        """Score every row in ``df`` and return the top-``n`` by predicted score.

        Classification models return a probability in ``score``; regression
        models return the predicted label.

        Args:
            df: Rows to score. Must have columns matching training features
                plus whatever is listed in ``include_columns``.
            n: Number of rows to keep after sorting.
            exclude_game_ids: game_ids to drop before ranking (e.g., games
                the user already owns).
            include_columns: Columns from ``df`` to surface alongside the
                score. Missing columns are silently skipped.
            use_finalized: If ``True`` use ``self.finalized_pipeline`` (refit
                on the full pre-finalize-through window). Required for
                ranking upcoming releases since the train-only pipeline
                never saw the most recent year.

        Returns:
            Polars DataFrame with the ``include_columns`` present in ``df``
            plus a ``score`` column, sorted by score descending, top ``n`` rows.
        """
        pipeline = self._require_fitted(use_finalized=use_finalized)
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

        if exclude_game_ids is not None and "game_id" in scored.columns:
            excl_df = pl.DataFrame({"game_id": list(exclude_game_ids)}).unique()
            scored = scored.join(excl_df, on="game_id", how="anti")

        kept = [c for c in include_columns if c in scored.columns]
        return scored.select(kept + ["score"]).sort("score", descending=True).head(n)

    def find_threshold(self, val_df: pl.DataFrame) -> float:
        """Return and stash the probability threshold that maximises the
        configured metric on ``val_df``. After this call, ``self.threshold``
        is set; subsequent ``evaluate`` / ``predict_with_labels`` /
        ``evaluate_stratified`` calls use it by default.
        """
        if self.outcome.task != "classification":
            raise ValueError("find_threshold is only meaningful for classification outcomes")
        pipeline = self._require_fitted()
        X, y = self._prepare(val_df)
        proba = pipeline.predict_proba(X)[:, 1]
        result = find_optimal_threshold(
            y, proba, metric=self.classification_config.threshold_optimization_metric
        )
        threshold = float(result["threshold"])
        self.threshold = threshold
        return threshold

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
