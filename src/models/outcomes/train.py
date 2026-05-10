"""Pure model training functions for outcome models.

This module exposes ``train_one``, a function that takes data frames and
a candidate-config dict, and returns the training artifacts. It does not
parse argv, load data, or write to disk — those are orchestration
concerns owned by ``src/pipeline/train.py``.

The module also hosts the model registry (``get_model_class``,
``MODEL_REGISTRY``, ``register_model``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone

from src.models.outcomes.base import TrainableModel
from src.models.outcomes.data import select_X_y
from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    evaluate_model,
    calculate_sample_weights,
)


logger = logging.getLogger(__name__)


# Registry of available model classes
MODEL_REGISTRY: Dict[str, Type[TrainableModel]] = {}


def register_model(model_class: Type[TrainableModel]) -> Type[TrainableModel]:
    """Register a model class in the registry.

    Args:
        model_class: TrainableModel subclass to register.

    Returns:
        The same class (allows use as decorator).
    """
    MODEL_REGISTRY[model_class.model_type] = model_class
    return model_class


def get_model_class(model_type: str) -> Type[TrainableModel]:
    """Get model class from registry.

    Args:
        model_type: Model type name (e.g., 'hurdle', 'complexity').

    Returns:
        TrainableModel subclass.

    Raises:
        ValueError: If model type not found.
    """
    if not MODEL_REGISTRY:
        _populate_registry()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def _populate_registry() -> None:
    """Import and register all model classes."""
    from src.models.outcomes.hurdle import HurdleModel
    from src.models.outcomes.complexity import ComplexityModel
    from src.models.outcomes.rating import RatingModel
    from src.models.outcomes.users_rated import UsersRatedModel
    from src.models.outcomes.geek_rating import GeekRatingModel

    register_model(HurdleModel)
    register_model(ComplexityModel)
    register_model(RatingModel)
    register_model(UsersRatedModel)
    register_model(GeekRatingModel)


def train_one(
    model_type: str,
    candidate_config: Dict[str, Any],
    train_df: pl.DataFrame,
    tune_df: pl.DataFrame,
    test_df: pl.DataFrame,
    metric: Optional[str] = None,
    patience: int = 15,
    preprocessor_type: str = "auto",
) -> Dict[str, Any]:
    """Train one candidate on one (train, tune, test) triple.

    Inputs are frames (already loaded from snapshot+split, already joined
    with any upstream score columns). Output is a dict of artifacts:
    pipeline, metrics, parameters, tune_predictions, test_predictions,
    and (for classification) optimal_threshold.

    Args:
        model_type: Model type name (e.g., 'hurdle', 'complexity').
        candidate_config: Dict with keys such as 'algorithm',
            'use_embeddings', 'use_sample_weights', 'algorithm_params', etc.
        train_df: Training split as a Polars DataFrame.
        tune_df: Tuning split as a Polars DataFrame.
        test_df: Test split as a Polars DataFrame.
        metric: Tuning metric override. Defaults to 'log_loss' for
            classification, 'rmse' for regression.
        patience: Early-stopping patience for tune_model.
        preprocessor_type: One of 'auto', 'linear', 'tree'.

    Returns:
        Dict with keys: pipeline, metrics, parameters,
        tune_predictions, test_predictions, and optionally
        optimal_threshold.
    """
    model_class = get_model_class(model_type)

    # Build model_kwargs from candidate config (model-specific knobs)
    model_kwargs: Dict[str, Any] = {}
    if "min_ratings" in candidate_config:
        model_kwargs["min_ratings"] = candidate_config["min_ratings"]
    if "min_weights" in candidate_config:
        model_kwargs["min_weights"] = candidate_config["min_weights"]
    if "mode" in candidate_config:
        model_kwargs["mode"] = candidate_config["mode"]
    if "include_predictions" in candidate_config:
        model_kwargs["include_predictions"] = candidate_config["include_predictions"]
    model = model_class(**model_kwargs)

    algorithm = candidate_config.get("algorithm")
    if algorithm is None:
        algorithm = "ridge" if model.model_task == "regression" else "lightgbm"

    logger.info(f"train_one: {model.model_type} / {algorithm}")
    logger.info(
        f"  input frames: train={train_df.shape}, tune={tune_df.shape}, test={test_df.shape}"
    )

    # X / y
    train_X, train_y = select_X_y(train_df, model.target_column)
    tune_X, tune_y = select_X_y(tune_df, model.target_column)
    test_X, test_y = select_X_y(test_df, model.target_column)
    tune_X_original = tune_X.copy()

    # Allow models to prepare features (e.g. geek_rating's stacking).
    # The model's prepare_features signature historically took an ``args``
    # namespace. Build a SimpleNamespace so we don't have to change the
    # model classes.
    prep_args = SimpleNamespace(
        use_embeddings=bool(candidate_config.get("use_embeddings", False)),
        sub_model_experiments=candidate_config.get("sub_model_experiments", {}),
        mode=candidate_config.get("mode"),
        include_predictions=candidate_config.get("include_predictions", True),
    )
    train_X, train_y = model.prepare_features(train_X, train_y, "train", prep_args)
    tune_X, tune_y = model.prepare_features(tune_X, tune_y, "tune", prep_args)
    test_X, test_y = model.prepare_features(test_X, test_y, "test", prep_args)
    logger.info(
        f"  after prepare_features: "
        f"train_X={train_X.shape}, tune_X={tune_X.shape}, test_X={test_X.shape}"
    )

    # Filter polars frames to match if prepare_features dropped rows
    if len(train_X) < len(train_df):
        train_df = train_df[train_X.index.tolist()]
    if len(tune_X) < len(tune_df):
        tune_df = tune_df[tune_X.index.tolist()]
    if len(test_X) < len(test_df):
        test_df = test_df[test_X.index.tolist()]

    # Configure model + estimator
    algorithm_params = candidate_config.get("algorithm_params", {}) or {}
    estimator, param_grid = model.configure_model(algorithm, algorithm_params)

    # Build preprocessor
    preserve_columns = ["year_published"]
    if model.data_config.requires_complexity_predictions:
        preserve_columns.append("predicted_complexity")
    if model_type == "geek_rating" and prep_args.mode == "direct":
        preserve_columns.append("predicted_complexity")
        if prep_args.include_predictions:
            preserve_columns.extend(["predicted_rating", "predicted_users_rated_log"])

    preprocessor_kwargs = dict(candidate_config.get("preprocessor_kwargs", {}) or {})
    preprocessor_kwargs.update(
        preserve_columns=preserve_columns,
        include_description_embeddings=prep_args.use_embeddings,
        include_count_features=bool(candidate_config.get("include_count_features", False)),
    )

    preprocessor = create_preprocessing_pipeline(
        model_type=preprocessor_type,
        model_name=algorithm,
        **preprocessor_kwargs,
    )
    pipeline = model.create_pipeline(estimator, preprocessor, algorithm, prep_args)

    # Sample weights
    sample_weights = None
    use_sample_weights = bool(candidate_config.get("use_sample_weights", False))
    weight_column = candidate_config.get("sample_weight_column")
    if use_sample_weights:
        if weight_column is None:
            weight_column = "num_weights" if model.model_type == "complexity" else "users_rated"
        # calculate_sample_weights expects pandas; convert train_df slice
        sample_weights = calculate_sample_weights(
            train_df.to_pandas(), weight_column=weight_column
        )

    # Tuning metric
    if metric is None:
        metric = "log_loss" if model.model_task == "classification" else "rmse"
    logger.info(f"  tuning metric: {metric}, patience: {patience}")
    if use_sample_weights:
        logger.info(f"  sample_weights: {weight_column}, n={len(sample_weights)}")

    tuned_pipeline, best_params, _ = tune_model(
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        tune_X=tune_X,
        tune_y=tune_y,
        param_grid=param_grid,
        metric=metric,
        patience=patience,
        sample_weights=sample_weights,
    )
    logger.info(f"  best params: {best_params}")

    # Train-set metrics from a clone fit on train only
    train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")
    _log_metrics_summary("train", train_metrics)

    # Optional threshold optimization (classification only)
    optimal_threshold: Optional[float] = None
    if hasattr(model, "find_optimal_threshold") and model.model_task == "classification":
        tune_pred_proba = train_pipeline.predict_proba(tune_X)[:, 1]
        threshold_results = model.find_optimal_threshold(tune_y, tune_pred_proba)
        optimal_threshold = float(threshold_results["threshold"])
        logger.info(f"  optimal threshold: {optimal_threshold:.4f}")

    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")
    _log_metrics_summary("tune", tune_metrics)

    # Refit on train + tune (matches existing behavior)
    if hasattr(model, "filter_for_refit"):
        tune_X_refit, tune_y_refit = model.filter_for_refit(tune_X, tune_y, tune_X_original)
    else:
        tune_X_refit, tune_y_refit = tune_X, tune_y

    X_combined = pd.concat([train_X, tune_X_refit])
    y_combined = pd.concat([train_y, tune_y_refit])

    if use_sample_weights:
        combined_weights = calculate_sample_weights(
            pl.concat([train_df, tune_df]).to_pandas(),
            weight_column=weight_column,
        )
        final_pipeline = clone(tuned_pipeline).fit(
            X_combined, y_combined,
            model__sample_weight=np.asarray(combined_weights),
        )
    else:
        final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)

    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")
    test_pred = final_pipeline.predict(test_X)
    additional = model.compute_additional_metrics(test_y.values, test_pred, "test")
    test_metrics.update(additional)
    _log_metrics_summary("test", test_metrics)

    # Predictions frames (polars, suitable for SnapshotStorage.save_result)
    tune_preds = _build_predictions_frame(
        train_pipeline, tune_X, tune_y, tune_df, model.model_task,
    )
    test_preds = _build_predictions_frame(
        final_pipeline, test_X, test_y, test_df, model.model_task,
    )

    # Feature importance / coefficients
    feature_importance = _extract_feature_importance(final_pipeline)
    if feature_importance is not None:
        _log_top_features(feature_importance, top_n=10)

    out: Dict[str, Any] = {
        "pipeline": final_pipeline,
        "metrics": {"train": train_metrics, "tune": tune_metrics, "test": test_metrics},
        "parameters": best_params,
        "tune_predictions": tune_preds,
        "test_predictions": test_preds,
    }
    if optimal_threshold is not None:
        out["optimal_threshold"] = optimal_threshold
    if feature_importance is not None:
        out["feature_importance"] = feature_importance
    return out


def _log_metrics_summary(fold: str, metrics: Dict[str, Any]) -> None:
    """Log a one-line summary of the scalar numeric metrics for a fold."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    if parts:
        logger.info(f"  {fold} metrics: " + ", ".join(parts))


def _extract_feature_importance(pipeline) -> Optional[pl.DataFrame]:
    """Pull coefficients (linear) or feature_importances_ (tree) from a fitted
    sklearn pipeline. Returns a polars DataFrame ready for save_result, or None."""
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        model = pipeline.named_steps.get("model")
        if preprocessor is None or model is None:
            return None

        feature_names = None
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            if hasattr(preprocessor, "named_steps"):
                for _name, step in reversed(list(preprocessor.named_steps.items())):
                    try:
                        feature_names = step.get_feature_names_out()
                        break
                    except Exception:
                        continue

        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim == 2:
                coef = coef[0]
            if feature_names is None or len(feature_names) != len(coef):
                feature_names = [f"f{i}" for i in range(len(coef))]
            return pl.DataFrame({
                "feature": list(feature_names),
                "coefficient": coef.tolist(),
                "abs_coefficient": np.abs(coef).tolist(),
            }).sort("abs_coefficient", descending=True)

        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_)
            if feature_names is None or len(feature_names) != len(imp):
                feature_names = [f"f{i}" for i in range(len(imp))]
            return pl.DataFrame({
                "feature": list(feature_names),
                "importance": imp.tolist(),
            }).sort("importance", descending=True)
    except Exception as e:
        logger.warning(f"Feature importance extraction failed: {e}")
    return None


def _log_top_features(fi: pl.DataFrame, top_n: int = 10) -> None:
    """Log the top-N features by absolute coefficient or importance."""
    sort_col = "abs_coefficient" if "coefficient" in fi.columns else "importance"
    val_col = "coefficient" if "coefficient" in fi.columns else "importance"
    top = fi.head(top_n).to_pandas()
    logger.info(f"  top {top_n} features by {sort_col}:")
    for _, row in top.iterrows():
        logger.info(f"    {row['feature']:40s}  {val_col}={row[val_col]:+.4f}")


def _build_predictions_frame(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    df: pl.DataFrame,
    model_task: str,
) -> pl.DataFrame:
    """Produce a polars frame matching df's rows + ``prediction``/``actual`` columns."""
    preds = pipeline.predict(X)
    out = df.clone().with_columns([
        pl.Series("prediction", preds),
        pl.Series("actual", y.values),
    ])
    if model_task == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                out = out.with_columns(pl.Series("predicted_proba", proba[:, 1]))
        except Exception:
            pass
    return out
