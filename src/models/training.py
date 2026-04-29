"""
Shared training utilities for machine learning models in the BGG predictive models project.

This module provides generic functions for data loading, preprocessing,
and model training that can be used across different model types.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, ParameterGrid, StratifiedKFold
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import lightgbm as lgb
from catboost import CatBoostRegressor

# Project imports
from src.utils.config import load_config
from src.data.loader import BGGDataLoader
from src.features.preprocessor import create_bgg_preprocessor
from src.models.splitting import time_based_split


def load_data(
    local_data_path: Optional[str] = None,
    end_train_year: Optional[int] = None,
    min_weights: int = 0,
    min_ratings: int = 0,
) -> pl.DataFrame:
    """
    Load training data from local parquet or BigQuery.

    Args:
        local_data_path: Path to local parquet file
        end_train_year: Cutoff year for data loading
        min_weights: Minimum number of weights for a game
        min_ratings: Minimum number of ratings for a game

    Returns:
        Polars DataFrame with loaded and filtered data
    """
    logger = logging.getLogger(__name__)

    if local_data_path:
        try:
            df = pl.read_parquet(local_data_path)
            logger.info(f"Loaded local data from {local_data_path}: {len(df)} rows")

            # Add validation for required columns based on context
            required_columns = [
                "year_published"
            ]  # Add other required columns as needed
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in data: {missing_columns}")

        except Exception as e:
            logger.error(f"Error loading local data from {local_data_path}: {e}")
            raise
    else:
        try:
            config = load_config()
            loader = BGGDataLoader(config.get_bigquery_config())
            df = loader.load_training_data(
                end_train_year=end_train_year + 1 if end_train_year else None,
                min_weights=min_weights,
                min_ratings=min_ratings,
            )
            logger.info(f"Loaded data from BigQuery: {len(df)} total rows")
        except Exception as e:
            logger.error(f"Error loading data from BigQuery: {e}")
            logger.error(
                "If BigQuery access is not configured, use --local-data to specify a local file"
            )
            raise

    logger.info(
        f"Year range: {df['year_published'].min()} - {df['year_published'].max()}"
    )
    return df


def create_data_splits(
    df: pl.DataFrame,
    train_through: int,
    tune_start: int,
    tune_through: int,
    test_start: int,
    test_through: int,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Create time-based data splits for model training and evaluation.

    Args:
        df: Input Polars DataFrame
        train_through: Last year to include in training data (inclusive)
        tune_start: First year for tuning data (inclusive)
        tune_through: Last year for tuning data (inclusive)
        test_start: First year for test data (inclusive)
        test_through: Last year for test data (inclusive)

    Returns:
        Tuple of train, tune, and test DataFrames
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating data splits...")

    validation_window = tune_through - tune_start + 1
    test_window = test_through - test_start + 1

    try:
        train_df, tune_df, test_df = time_based_split(
            df=df,
            train_through=train_through,
            prediction_window=validation_window,
            test_window=test_window,
            time_col="year_published",
            return_dict=False,
        )

        if len(train_df) == 0:
            raise ValueError(
                f"Training set is empty. Check train_through={train_through}"
            )
        if len(tune_df) == 0:
            raise ValueError(
                f"Tuning set is empty. Check tune years: {tune_start}-{tune_through}"
            )
        if len(test_df) == 0:
            raise ValueError(
                f"Test set is empty. Check test years: {test_start}-{test_through}"
            )

    except Exception as e:
        logger.error(f"Error creating data splits: {e}")
        raise

    logger.info(f"Training data: {len(train_df)} rows (years through {train_through})")
    logger.info(
        f"Tuning data: {len(tune_df)} rows (years {tune_start}-{tune_through})"
    )
    logger.info(
        f"Test data: {len(test_df)} rows (years {test_start}-{test_through})"
    )

    return train_df, tune_df, test_df


def select_X_y(
    df: pl.DataFrame, y_column: str, to_pandas: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features (X) and target (y) from a dataframe.

    Args:
        df: Polars DataFrame
        y_column: Name of the target column
        to_pandas: Whether to convert to pandas DataFrame/Series

    Returns:
        X, y as either Polars or Pandas objects

    Raises:
        ValueError: If y_column is not in the dataframe
    """
    if y_column not in df.columns:
        raise ValueError(
            f"Target column '{y_column}' not found in dataframe. Available columns: {df.columns}"
        )

    X = df.drop(y_column)
    y = df.select(y_column)

    if to_pandas:
        X = X.to_pandas()
        y = y.to_pandas().squeeze()
        return X, y
    else:
        return X, y


def create_preprocessing_pipeline(
    model_type: str = "auto", model_name: str = None, **kwargs
) -> Pipeline:
    """
    Create a flexible preprocessing pipeline with configurable parameters.

    Args:
        model_type : str, optional (default='auto')
            Type of preprocessor to create.
            Options:
            - 'auto': Automatically select preprocessor type based on model_name
            - 'linear': Full preprocessing with scaling and transformations
            - 'tree': Minimal preprocessing suitable for tree-based models
        model_name : str, optional
            Name of the model to use for automatic preprocessor selection.
            Required when model_type='auto'.
        **kwargs: Keyword arguments to customize preprocessing

    Returns:
        Scikit-learn Pipeline with preprocessing steps

    Raises:
        ValueError: If an unsupported model_type is provided or if model_name is required but not provided
    """
    # Mapping of model names to their recommended preprocessor types
    MODEL_TO_PREPROCESSOR = {
        # Linear models - benefit from scaling and transformations
        "linear": "linear",
        "ridge": "linear",
        "lasso": "linear",
        "logistic": "linear",
        "svc": "linear",
        "bayesian_ridge": "linear",
        "ard": "linear",
        # Tree-based models - work better with minimal preprocessing
        "lightgbm": "tree",
        "catboost": "tree",
        "rf": "tree",
        "random_forest": "tree",
        "lightgbm_linear": "tree",  # Still tree-based despite linear leaves
    }

    # Determine the actual preprocessor type to use
    if model_type == "auto":
        if model_name is None:
            raise ValueError("model_name must be provided when model_type='auto'")

        # Look up the preprocessor type for this model
        if model_name not in MODEL_TO_PREPROCESSOR:
            raise ValueError(
                f"Unknown model name '{model_name}' for automatic preprocessor selection. "
                f"Supported models: {list(MODEL_TO_PREPROCESSOR.keys())}. "
                f"Use model_type='linear' or 'tree' for manual selection."
            )

        actual_model_type = MODEL_TO_PREPROCESSOR[model_name]
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-selected '{actual_model_type}' preprocessor for model '{model_name}'"
        )
    elif model_type in ["linear", "tree"]:
        actual_model_type = model_type
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Choose from 'auto', 'linear', or 'tree'."
        )

    # Default preprocessing configuration
    default_config = {
        "model_type": actual_model_type,  # Use the determined model type
        "reference_year": 2000,
        "normalization_factor": 25,
        "log_columns": ["min_age", "min_playtime", "max_playtime"],
        "category_min_freq": 0,
        "mechanic_min_freq": 0,
        "designer_min_freq": 10,
        "artist_min_freq": 10,
        "publisher_min_freq": 5,
        "family_min_freq": 10,
        "max_artist_features": 500,
        "max_publisher_features": 250,
        "max_designer_features": 500,
        "max_family_features": 500,
        "max_mechanic_features": 500,
        "max_category_features": 500,
        "create_category_features": True,
        "create_mechanic_features": True,
        "create_designer_features": True,
        "create_artist_features": True,
        "create_publisher_features": True,
        "create_family_features": True,
        "create_player_dummies": True,
        "include_base_numeric": True,
        "include_description_embeddings": False,
        "verbose": False,
        "preserve_columns": ["year_published"],
        "remove_correlated": False,  # Backwards compatible default
        "correlation_threshold": 0.95,
    }

    # Update default config with provided kwargs
    default_config.update(kwargs)

    # Create preprocessing pipeline
    pipeline = create_bgg_preprocessor(**default_config)

    return pipeline


def preprocess_data(
    df: pl.DataFrame,
    preprocessing_pipeline: Pipeline,
    fit: bool = False,
    dataset_name: str = "data",
) -> pd.DataFrame:
    """
    Preprocess data for model training.

    Args:
        df: Input Polars DataFrame
        preprocessing_pipeline: Scikit-learn preprocessing pipeline
        fit: Whether to fit the pipeline or just transform
        dataset_name: Name of the dataset for logging

    Returns:
        Preprocessed features as a pandas DataFrame
    """
    # Convert array columns to lists before converting to pandas
    array_columns = [
        "categories",
        "mechanics",
        "designers",
        "artists",
        "publishers",
        "families",
    ]
    df_converted = df.clone()

    for col in array_columns:
        if col in df.columns:
            # Convert polars array to python list with explicit return type
            df_converted = df_converted.with_columns(
                pl.col(col).map_elements(
                    lambda x: x.to_list() if x is not None else [],
                    return_dtype=pl.List(pl.Utf8),
                )
            )

    # Convert to pandas
    df_pandas = df_converted.to_pandas()

    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing {dataset_name} set: {len(df_pandas)} rows")

    # Process with pipeline
    if fit:
        features = preprocessing_pipeline.fit_transform(df_pandas)
        logger.info(f"  Fitted preprocessing pipeline on {dataset_name} data")
    else:
        features = preprocessing_pipeline.transform(df_pandas)
        logger.info(f"  Applied preprocessing pipeline to {dataset_name} data")

    logger.info(f"  {dataset_name.title()} features shape: {features.shape}")

    # Get feature names from BGG preprocessor
    try:
        feature_names = preprocessing_pipeline.named_steps[
            "bgg_preprocessor"
        ].get_feature_names_out()
    except Exception as e:
        logger.warning(f"Could not get feature names: {e}")
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    # Convert to DataFrame
    features = pd.DataFrame(features, columns=feature_names)
    return features


def tune_model(
    pipeline: Pipeline,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    tune_X: pd.DataFrame,
    tune_y: pd.Series,
    param_grid: Dict[str, Any],
    metric: str = "rmse",
    patience: int = 5,
    min_delta: float = 1e-4,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Tune hyperparameters using a separate tuning set.

    Args:
        pipeline: The pipeline to tune
        train_X: Training features
        train_y: Training target
        tune_X: Tuning features
        tune_y: Tuning target
        param_grid: Grid of parameters to search
        metric: Metric to optimize
        patience: Number of iterations without improvement
        min_delta: Minimum change in score to be considered an improvement

    Returns:
        Tuple of (fitted pipeline, best params, results DataFrame). The
        results frame has one row per evaluated config (early-stopped configs
        are not included), columns: ``params``, ``score``, ``metric``,
        sorted best-first in the metric's natural direction.
    """
    from sklearn.base import clone
    import gc

    logger = logging.getLogger(__name__)

    # Validate inputs
    if train_X is None or train_y is None or tune_X is None or tune_y is None:
        raise ValueError("All input arrays must not be None")

    if not isinstance(train_X, pd.DataFrame) or not isinstance(tune_X, pd.DataFrame):
        raise ValueError("train_X and tune_X must be pandas DataFrames")

    if not isinstance(train_y, (pd.Series, np.ndarray)) or not isinstance(
        tune_y, (pd.Series, np.ndarray)
    ):
        raise ValueError("train_y and tune_y must be pandas Series or numpy arrays")

    if len(train_X) != len(train_y):
        raise ValueError(
            f"train_X and train_y must have same length, got {len(train_X)} and {len(train_y)}"
        )

    if len(tune_X) != len(tune_y):
        raise ValueError(
            f"tune_X and tune_y must have same length, got {len(tune_X)} and {len(tune_y)}"
        )

    if len(train_X) == 0 or len(tune_X) == 0:
        raise ValueError("Training and tuning sets cannot be empty")

    # Validate hyperparameter tuning parameters
    if not isinstance(patience, int):
        raise ValueError("Patience must be an integer")
    if patience < 1:
        raise ValueError("Patience must be at least 1")

    if not isinstance(min_delta, (int, float)):
        raise ValueError("min_delta must be a number")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    # Validate metric
    if not isinstance(metric, str):
        raise ValueError("Metric must be a string")

    # Expanded list of valid metrics
    valid_metrics = [
        "rmse",  # Regression
        "log_loss",
        "f1",
        "accuracy",
        "precision",
        "recall",
        "auc",  # Classification
        "mse",
        "mae",
        "r2",
        "mape",  # Additional regression metrics
    ]
    if metric not in valid_metrics:
        raise ValueError(f"Unsupported metric: {metric}. Choose from {valid_metrics}")

    # Validate target values for specific metrics
    if metric in ["log_loss", "f1", "accuracy", "precision", "recall", "auc"]:
        if not np.all(np.isin(train_y, [0, 1])):
            raise ValueError(
                f"Training target values must be binary (0 or 1) for {metric} metric"
            )
        if not np.all(np.isin(tune_y, [0, 1])):
            raise ValueError(
                f"Tuning target values must be binary (0 or 1) for {metric} metric"
            )

    # Validate pipeline
    if pipeline is None:
        raise ValueError("Pipeline cannot be None")

    if not isinstance(pipeline, Pipeline):
        raise ValueError("Pipeline must be a scikit-learn Pipeline object")

    if (
        "preprocessor" not in pipeline.named_steps
        or "model" not in pipeline.named_steps
    ):
        raise ValueError("Pipeline must have 'preprocessor' and 'model' steps")

    # Get preprocessor from pipeline
    preprocessor = pipeline.named_steps["preprocessor"]

    # Fit and transform the data once with the preprocessor
    logger.info("Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(train_X)
    X_tune_transformed = preprocessor.transform(tune_X)
    logger.info(
        f"Transformed features shape: Train {X_train_transformed.shape}, Tune {X_tune_transformed.shape}"
    )

    # Validate param_grid
    if param_grid is None:
        param_grid = {}

    # Scoring functions using sklearn metrics
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error,
        log_loss,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    scoring_functions = {
        # Regression metrics
        "rmse": lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "r2": r2_score,
        "mape": mean_absolute_percentage_error,
        # Classification metrics
        "log_loss": log_loss,
        "f1": f1_score,
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "auc": roc_auc_score,
    }

    # Note: Metric validation already done above

    score_func = scoring_functions[metric]

    best_score = np.inf  # Initialize to infinity since we want to minimize
    best_params = None
    best_model = None
    tuning_results = []
    patience_counter = 0

    # Initialize models
    base_model = clone(pipeline.named_steps["model"])
    current_best_model = None

    # Fit base model as fallback
    try:
        base_model.fit(X_train_transformed, train_y)

        # Verify base model works with metric
        if metric == "log_loss":
            if not hasattr(base_model, "predict_proba"):
                raise AttributeError(
                    f"Base model {base_model.__class__.__name__} does not support predict_proba"
                )
            test_pred = base_model.predict_proba(X_train_transformed[:1])
            if test_pred.shape[1] != 2:
                raise ValueError(
                    f"Base model returned invalid probability shape: {test_pred.shape}"
                )

        current_best_model = base_model  # Use base model as initial best
    except Exception as e:
        logger.error(f"Failed to initialize base model: {str(e)}")
        raise RuntimeError(
            "Could not initialize base model with required capabilities"
        ) from e

    try:
        # Get total number of parameter combinations for logging
        param_combinations = list(ParameterGrid(param_grid)) if param_grid else [{}]
        n_combinations = len(param_combinations)
        logger.info(f"Testing {n_combinations} parameter combinations")

        # Use tqdm for progress tracking
        for i, params in enumerate(
            tqdm(param_combinations, desc="Tuning hyperparameters")
        ):
            logger.info(f"Evaluating combination {i + 1}/{n_combinations}: {params}")

            try:
                # Create and configure model for this iteration
                model_candidate = clone(pipeline.named_steps["model"])

                # Set parameters if any
                if params:
                    model_candidate.set_params(
                        **{k.replace("model__", ""): v for k, v in params.items()}
                    )

                # Fit the model with optional sample weights
                if sample_weights is not None:
                    sample_weights_array = np.asarray(sample_weights)
                    model_candidate.fit(
                        X_train_transformed, train_y, sample_weight=sample_weights_array
                    )
                else:
                    model_candidate.fit(X_train_transformed, train_y)

                # Predict on tuning set
                try:
                    if metric == "log_loss":
                        if not hasattr(model_candidate, "predict_proba"):
                            raise AttributeError(
                                f"Model {model_candidate.__class__.__name__} does not support predict_proba"
                            )
                        y_tune_pred = model_candidate.predict_proba(X_tune_transformed)
                        if y_tune_pred.shape[1] != 2:
                            raise ValueError(
                                f"Expected binary classification probabilities, got shape {y_tune_pred.shape}"
                            )
                    else:
                        y_tune_pred = model_candidate.predict(X_tune_transformed)

                    # Calculate score
                    score = score_func(tune_y, y_tune_pred)

                    # Validate score is finite
                    if not np.isfinite(score):
                        raise ValueError(f"Invalid score value: {score}")

                except Exception as e:
                    logger.warning(f"Error during prediction/scoring: {str(e)}")
                    continue

                # Store detailed results
                result = {"params": params, "score": score, "metric": metric}
                tuning_results.append(result)

                logger.info(f"Params {params}: {metric} = {score:.4f}")

                # Check if this is the best score
                if score < best_score - min_delta:  # Improvement beyond threshold
                    best_score = score
                    best_params = params.copy()  # Make a copy to be safe
                    best_model = clone(model_candidate)
                    current_best_model = best_model
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping check
                if patience_counter >= patience:
                    logger.info(
                        f"Early stopping triggered after {patience} iterations without improvement"
                    )
                    break

            except Exception as e:
                logger.warning(f"Failed to train with params {params}: {str(e)}")
                continue
            finally:
                # Clean up temporary objects to prevent memory leaks
                gc.collect()

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        if best_params is None:
            raise RuntimeError("Hyperparameter tuning failed completely") from e

    # Log full tuning results
    logger.info("Hyperparameter Tuning Results:")
    for result in sorted(tuning_results, key=lambda x: x["score"]):
        logger.info(f"  Params: {result['params']}, {metric}: {result['score']:.4f}")

    logger.info(f"Best params: {best_params} ({metric} = {best_score:.4f})")

    # Ensure we have valid parameters
    if best_params is None:
        best_params = {}

    # Refit the Pipeline as a whole on raw training data so the returned
    # object is a genuinely fitted sklearn Pipeline. During tuning we fit
    # preprocessor and model separately on already-transformed arrays for
    # speed; that leaves the Pipeline wrapper itself in an unfit state and
    # downstream `pipeline.predict(...)` raises NotFittedError. The refit
    # here is fast relative to the tuning loop and makes the return value
    # a proper drop-in sklearn Pipeline.
    final_model_cls = clone(current_best_model) if current_best_model is not None else clone(base_model)
    tuned_pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", final_model_cls),
    ])
    tuned_pipeline.fit(train_X, train_y)

    # All metrics that come through here are lower-is-better at scoring time,
    # since the loop minimizes; sort ascending.
    results_df = (
        pd.DataFrame(tuning_results)
        .sort_values("score", ascending=True)
        .reset_index(drop=True)
        if tuning_results
        else pd.DataFrame(columns=["params", "score", "metric"])
    )

    return tuned_pipeline, best_params, results_df


def tune_model_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    param_grid: Dict[str, Any],
    metric: str = "rmse",
    cv_folds: int = 5,
    task: str = "classification",
    random_seed: int = 42,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """Tune hyperparameters via k-fold cross-validation on a single dataset.

    Exhaustive grid search: every parameter combination in ``param_grid`` is
    evaluated by ``cv_folds`` CV scoring on (X, y). The best-scoring config
    is then refit on the full (X, y) and returned as a fitted Pipeline.

    Use this when you want CV-based selection on the training set instead of
    holdout-based selection against a separate val set (see :func:`tune_model`
    for that).

    Args:
        pipeline: Pipeline with 'preprocessor' and 'model' steps. The
            preprocessor is refit inside each fold to avoid leakage.
        X: Features (pandas DataFrame).
        y: Target (pandas Series).
        param_grid: Same dict-of-lists shape as :func:`tune_model` —
            keys prefixed with 'model__'.
        metric: Same metric vocabulary as :func:`tune_model`. Lower is
            better for the loss-style metrics (rmse/mse/mae/log_loss);
            higher is better for the rest. The function negates the
            higher-is-better ones internally so the returned best config
            always corresponds to the best score for that metric in its
            natural direction.
        cv_folds: Number of CV folds. Stratified for classification, plain
            KFold for regression.
        task: 'classification' or 'regression'. Controls fold strategy and
            which metrics are valid.
        random_seed: Seed for fold shuffling.
        sample_weights: Optional per-row weights, applied during fold fits.

    Returns:
        (fitted_pipeline, best_params, results_df). ``results_df`` has one
        row per evaluated config with columns ``params``, ``score``,
        ``score_std``, ``n_folds``, ``metric`` — sorted best-first in the
        metric's natural direction.
    """
    from sklearn.base import clone
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error,
        log_loss,
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    logger = logging.getLogger(__name__)

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError("y must be a pandas Series or numpy array")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length, got {len(X)} and {len(y)}"
        )
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    if pipeline is None or not isinstance(pipeline, Pipeline):
        raise ValueError("pipeline must be a scikit-learn Pipeline")
    if (
        "preprocessor" not in pipeline.named_steps
        or "model" not in pipeline.named_steps
    ):
        raise ValueError("Pipeline must have 'preprocessor' and 'model' steps")

    # Lower-is-better; everything else is negated so we can always minimize.
    lower_is_better = {"rmse", "mse", "mae", "mape", "log_loss"}
    scoring_functions = {
        "rmse": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
        "mse": mean_squared_error,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
        "r2": r2_score,
        "log_loss": log_loss,
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "auc": roc_auc_score,
    }
    if metric not in scoring_functions:
        raise ValueError(
            f"Unsupported metric: {metric}. Choose from {list(scoring_functions)}"
        )
    score_func = scoring_functions[metric]
    needs_proba = metric in {"log_loss", "auc"}

    if task == "classification":
        if metric in {"r2", "rmse", "mse", "mae", "mape"}:
            raise ValueError(
                f"metric {metric!r} is a regression metric; not valid for task='classification'"
            )
        if not np.all(np.isin(np.asarray(y), [0, 1])):
            raise ValueError(
                f"Target values must be binary (0 or 1) for classification metric {metric}"
            )
        splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_seed
        )
    else:
        if metric in {"log_loss", "f1", "accuracy", "precision", "recall", "auc"}:
            raise ValueError(
                f"metric {metric!r} is a classification metric; not valid for task='regression'"
            )
        splitter = KFold(
            n_splits=cv_folds, shuffle=True, random_state=random_seed
        )

    param_combinations = list(ParameterGrid(param_grid)) if param_grid else [{}]
    n_combinations = len(param_combinations)
    logger.info(
        f"CV tuning {n_combinations} combinations × {cv_folds} folds = "
        f"{n_combinations * cv_folds} fits"
    )

    base_preprocessor = pipeline.named_steps["preprocessor"]
    base_model = pipeline.named_steps["model"]

    weights_array = np.asarray(sample_weights) if sample_weights is not None else None

    tuning_results = []
    best_score = np.inf
    best_params: Optional[Dict[str, Any]] = None

    for i, params in enumerate(tqdm(param_combinations, desc="CV tuning")):
        logger.info(f"Evaluating combination {i + 1}/{n_combinations}: {params}")
        fold_scores: list = []

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
                X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_vl = (
                    y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx],
                    y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx],
                )

                preproc = clone(base_preprocessor)
                X_tr_t = preproc.fit_transform(X_tr)
                X_vl_t = preproc.transform(X_vl)

                model_candidate = clone(base_model)
                if params:
                    model_candidate.set_params(
                        **{k.replace("model__", ""): v for k, v in params.items()}
                    )

                if weights_array is not None:
                    model_candidate.fit(
                        X_tr_t, y_tr, sample_weight=weights_array[train_idx]
                    )
                else:
                    model_candidate.fit(X_tr_t, y_tr)

                if needs_proba:
                    if not hasattr(model_candidate, "predict_proba"):
                        raise AttributeError(
                            f"Model {type(model_candidate).__name__} does not support "
                            f"predict_proba (required for metric={metric})"
                        )
                    proba = model_candidate.predict_proba(X_vl_t)
                    y_pred = proba[:, 1] if metric == "auc" else proba
                else:
                    y_pred = model_candidate.predict(X_vl_t)

                score = float(score_func(y_vl, y_pred))
                if not np.isfinite(score):
                    raise ValueError(f"Invalid fold score: {score}")
                fold_scores.append(score)

        except Exception as e:
            logger.warning(f"Failed CV for params {params}: {e}")
            continue
        finally:
            import gc
            gc.collect()

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        # Normalize to "lower is better" for selection.
        selection_score = mean_score if metric in lower_is_better else -mean_score

        tuning_results.append({
            "params": params,
            "score": mean_score,
            "score_std": std_score,
            "metric": metric,
            "n_folds": len(fold_scores),
        })
        logger.info(
            f"Params {params}: {metric} mean={mean_score:.4f} std={std_score:.4f}"
        )

        if selection_score < best_score:
            best_score = selection_score
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("CV hyperparameter tuning failed for all combinations")

    logger.info("CV tuning results:")
    for r in sorted(
        tuning_results,
        key=lambda x: x["score"] if metric in lower_is_better else -x["score"],
    ):
        logger.info(
            f"  Params: {r['params']}, {metric}: {r['score']:.4f} ± {r['score_std']:.4f}"
        )
    natural_best = best_score if metric in lower_is_better else -best_score
    logger.info(
        f"Best params: {best_params} ({metric} = {natural_best:.4f})"
    )

    final_model = clone(base_model)
    final_model.set_params(
        **{k.replace("model__", ""): v for k, v in best_params.items()}
    )
    tuned_pipeline = Pipeline([
        ("preprocessor", clone(base_preprocessor)),
        ("model", final_model),
    ])
    if weights_array is not None:
        tuned_pipeline.fit(X, y, model__sample_weight=weights_array)
    else:
        tuned_pipeline.fit(X, y)

    if tuning_results:
        results_df = pd.DataFrame(tuning_results).sort_values(
            "score", ascending=metric in lower_is_better
        ).reset_index(drop=True)
    else:
        results_df = pd.DataFrame(
            columns=["params", "score", "score_std", "n_folds", "metric"]
        )

    return tuned_pipeline, best_params, results_df


def calculate_sample_weights(
    df: pd.DataFrame, weight_column: str = "users_rated"
) -> np.ndarray:
    """
    Calculate sample weights based on a specified column.

    Args:
        df (pd.DataFrame): DataFrame containing the weight column
        weight_column (str, optional): Column to use for calculating weights.
                                       Defaults to 'users_rated'.

    Returns:
        numpy array of normalized weights
    """
    if weight_column not in df.columns:
        raise ValueError(f"Column '{weight_column}' not found in DataFrame")

    weights = np.sqrt(df[weight_column]) / np.sqrt(df[weight_column].max())
    return weights


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "test",
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model performance with standard metrics.

    Args:
        model: Trained model
        X: Features
        y: True target values
        dataset_name: Name of the dataset for logging
        threshold: Classification threshold for binary predictions

    Returns:
        Dictionary of performance metrics
    """
    logger = logging.getLogger(__name__)

    # Validate inputs
    if X is None or y is None:
        raise ValueError("X and y cannot be None")

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame")

    if not isinstance(y, (pd.Series, np.ndarray)):
        raise ValueError("y must be a pandas Series or numpy array")

    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

    if len(X) == 0:
        raise ValueError("X and y cannot be empty")

    # Validate model
    if model is None:
        raise ValueError("Model cannot be None")

    if not hasattr(model, "predict"):
        raise ValueError("Model must implement predict method")

    if not hasattr(model, "fit"):
        raise ValueError("Model must be a fitted estimator with fit method")

    # Validate threshold
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    # Get predictions for both regression and classification
    y_pred = model.predict(X)

    # Check if model supports predict_proba (for classification)
    is_classifier = hasattr(model, "predict_proba")

    # Regression metrics
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error,
    )

    metrics = {}

    # Only compute regression metrics if it's not a classifier
    if not is_classifier:
        metrics = {
            "mse": mean_squared_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
            "mape": mean_absolute_percentage_error(y, y_pred),
        }

    # If it's a classifier, add classification metrics
    if is_classifier:
        try:
            y_pred_proba = model.predict_proba(X)

            # Ensure we have binary classification probabilities
            if y_pred_proba.shape[1] != 2:
                raise ValueError(
                    f"Expected binary classification probabilities, got shape {y_pred_proba.shape}"
                )

            # Validate target values are binary
            if not np.all(np.isin(y, [0, 1])):
                raise ValueError("Target values must be binary (0 or 1)")

            # Get positive class probabilities
            y_pred_prob = y_pred_proba[:, 1]

            # Clip only exact 0s and 1s for numerical stability
            eps = 1e-15
            y_pred_prob = np.where(y_pred_prob == 0, eps, y_pred_prob)
            y_pred_prob = np.where(y_pred_prob == 1, 1 - eps, y_pred_prob)

            # Apply threshold for predictions
            y_pred_class = (y_pred_prob >= threshold).astype(int)

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                log_loss,
                fbeta_score,
                matthews_corrcoef,
                confusion_matrix,
            )

            metrics.update(
                {
                    "accuracy": accuracy_score(y, y_pred_class),
                    "precision": precision_score(y, y_pred_class),
                    "recall": recall_score(y, y_pred_class),
                    "f1": f1_score(y, y_pred_class),
                    "f2": fbeta_score(y, y_pred_class, beta=2.0),
                    "auc": roc_auc_score(y, y_pred_prob),
                    "log_loss": log_loss(y, y_pred_prob),
                    "matthews_corr": matthews_corrcoef(y, y_pred_class),
                    "confusion_matrix": confusion_matrix(y, y_pred_class).tolist(),
                }
            )
        except Exception as e:
            logger.warning(f"Could not compute classification metrics: {str(e)}")

    logger.info(f"{dataset_name.title()} Performance:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
        elif metric == "confusion_matrix":
            logger.info(f"  {metric}: {value}")
        else:
            logger.info(f"  {metric}: {value}")

    return metrics


def configure_model(model_name: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Set up regression model and parameter grid."""
    model_MAPPING = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "catboost": CatBoostRegressor,
        "lightgbm": lgb.LGBMRegressor,
        "lightgbm_linear": lgb.LGBMRegressor,  # New model type
    }

    PARAM_GRIDS = {
        "linear": {},  # Linear Regression has no hyperparameters to tune
        "ridge": {
            "model__alpha": [0.0001, 0.0005, 0.01, 0.1, 1.0, 5],  # Expanded alpha range
            "model__solver": ["auto"],
            "model__fit_intercept": [True],
        },
        "lasso": {
            "model__alpha": [0.1, 1.0, 10.0],
            "model__selection": ["cyclic", "random"],
        },
        "lightgbm": {
            "model__n_estimators": [500],
            "model__learning_rate": [0.01],
            "model__max_depth": [-1],  # -1 means no limit
            "model__num_leaves": [50, 100],
            "model__min_child_samples": [10],
            "model__reg_alpha": [0.1],
        },
        "catboost": {
            "model__iterations": [500],
            "model__learning_rate": [0.01, 0.5],
            "model__depth": [4, 6, 8],
            "model__l2_leaf_reg": [1, 3, 5],
        },
        "lightgbm_linear": {
            "model__n_estimators": [500, 1000],
            "model__learning_rate": [0.01, 0.05],
            "model__max_depth": [
                3,
                5,
                7,
            ],  # Shallower trees work better with linear leaves
            "model__num_leaves": [31, 50],  # Fewer leaves for linear trees
            "model__min_child_samples": [10, 20],
            "model__reg_alpha": [0.1, 1.0],
            "model__linear_tree": [True],  # Key parameter for linear leaves
        },
    }

    model = model_MAPPING[model_name]()
    param_grid = PARAM_GRIDS[model_name]

    return model, param_grid
