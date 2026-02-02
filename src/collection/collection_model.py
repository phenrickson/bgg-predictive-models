"""Train ownership prediction models for user collections."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, clone
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
    confusion_matrix,
)

import lightgbm as lgb
from catboost import CatBoostClassifier

from src.models.training import (
    create_preprocessing_pipeline,
    tune_model,
    select_X_y,
)
from src.models.outcomes.hurdle import find_optimal_threshold

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for collection ownership model."""

    model_type: str = "lightgbm"
    """Model type: 'lightgbm', 'catboost', or 'logistic'."""

    use_sample_weights: bool = False
    """Whether to use sample weights based on game popularity."""

    handle_imbalance: str = "scale_pos_weight"
    """How to handle class imbalance: 'scale_pos_weight', 'none'."""

    threshold_optimization_metric: str = "f2"
    """Metric for threshold optimization: 'f1', 'f2', 'precision', 'recall'."""

    preprocessor_type: str = "auto"
    """Preprocessor type: 'auto', 'linear', or 'tree'."""

    tuning_metric: str = "log_loss"
    """Metric for hyperparameter tuning: 'log_loss', 'f1', 'auc'."""

    patience: int = 10
    """Early stopping patience for hyperparameter tuning."""


# Model and parameter grid configurations
CLASSIFIER_MAPPING = {
    "logistic": lambda: LogisticRegression(max_iter=4000),
    "lightgbm": lambda: lgb.LGBMClassifier(objective="binary", verbose=-1),
    "catboost": lambda: CatBoostClassifier(verbose=0),
}

PARAM_GRIDS = {
    "logistic": {
        "model__C": [0.001, 0.01, 0.1, 1.0],
        "model__penalty": ["l2"],
    },
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


class CollectionModel:
    """Train ownership prediction model for a user's collection.

    Uses the existing preprocessing pipeline from the project and follows
    the same training patterns as the hurdle model. The model predicts
    the probability that a user would want to own a given game.

    Example usage:
        >>> model = CollectionModel("phenrickson")
        >>> pipeline, best_params = model.train(train_df, val_df)
        >>> threshold = model.find_optimal_threshold(pipeline, val_df)
        >>> predictions = model.predict(pipeline, games_df, threshold)
    """

    def __init__(
        self,
        username: str,
        config: Optional[ModelConfig] = None,
    ):
        """Initialize model trainer for a specific user.

        Args:
            username: BGG username (for logging and metadata)
            config: Model configuration
        """
        self.username = username
        self.config = config or ModelConfig()

        logger.info(f"Initialized CollectionModel for user '{username}'")
        logger.info(f"Model type: {self.config.model_type}")

    def _configure_model(self) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Set up classifier and parameter grid based on config."""
        if self.config.model_type not in CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unknown model type: {self.config.model_type}. "
                f"Choose from: {list(CLASSIFIER_MAPPING.keys())}"
            )

        model = CLASSIFIER_MAPPING[self.config.model_type]()
        param_grid = PARAM_GRIDS[self.config.model_type].copy()

        return model, param_grid

    def _prepare_data(
        self, df: pl.DataFrame, target_col: str = "target"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare DataFrame for training by extracting X and y.

        Args:
            df: Polars DataFrame with features and target
            target_col: Name of target column

        Returns:
            Tuple of (X, y) as pandas objects
        """
        return select_X_y(df, y_column=target_col, to_pandas=True)

    def train(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        param_grid: Optional[Dict[str, Any]] = None,
        target_col: str = "target",
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Train model with hyperparameter tuning.

        Uses the existing preprocessing pipeline and tune_model function
        from the project's training utilities.

        Args:
            train_df: Training DataFrame with features and target
            val_df: Validation DataFrame for hyperparameter tuning
            param_grid: Optional custom parameter grid
            target_col: Name of target column

        Returns:
            Tuple of (fitted_pipeline, best_params)
        """
        logger.info(f"Training ownership model for user '{self.username}'")
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

        # Prepare data
        train_X, train_y = self._prepare_data(train_df, target_col)
        val_X, val_y = self._prepare_data(val_df, target_col)

        # Log class balance
        train_pos_rate = train_y.mean()
        val_pos_rate = val_y.mean()
        logger.info(f"Train positive rate: {train_pos_rate:.2%}")
        logger.info(f"Val positive rate: {val_pos_rate:.2%}")

        # Configure model and params
        model, default_param_grid = self._configure_model()
        param_grid = param_grid or default_param_grid

        # Create preprocessing pipeline (reuse existing)
        preprocessor = create_preprocessing_pipeline(
            model_type=self.config.preprocessor_type,
            model_name=self.config.model_type,
        )

        # Build full pipeline
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        logger.info(f"Pipeline: {pipeline}")
        logger.info(f"Parameter grid: {param_grid}")

        # Calculate sample weights if requested
        sample_weights = None
        if self.config.use_sample_weights and "users_rated" in train_df.columns:
            users_rated = train_df["users_rated"].to_numpy()
            sample_weights = np.sqrt(users_rated) / np.sqrt(users_rated.max())
            logger.info("Using sample weights based on users_rated")

        # Tune hyperparameters
        tuned_pipeline, best_params = tune_model(
            pipeline=pipeline,
            train_X=train_X,
            train_y=train_y,
            tune_X=val_X,
            tune_y=val_y,
            param_grid=param_grid,
            metric=self.config.tuning_metric,
            patience=self.config.patience,
            sample_weights=sample_weights,
        )

        logger.info(f"Best parameters: {best_params}")

        # Fit the tuned pipeline on training data
        logger.info("Fitting final pipeline on training data...")
        fitted_pipeline = clone(tuned_pipeline).fit(train_X, train_y)

        return fitted_pipeline, best_params

    def find_optimal_threshold(
        self,
        pipeline: Pipeline,
        val_df: pl.DataFrame,
        metric: Optional[str] = None,
        target_col: str = "target",
    ) -> float:
        """Find optimal classification threshold on validation set.

        Args:
            pipeline: Fitted pipeline
            val_df: Validation DataFrame
            metric: Metric to optimize (default from config)
            target_col: Name of target column

        Returns:
            Optimal threshold value
        """
        metric = metric or self.config.threshold_optimization_metric

        val_X, val_y = self._prepare_data(val_df, target_col)

        # Get predicted probabilities
        val_pred_proba = pipeline.predict_proba(val_X)[:, 1]

        # Use existing function from hurdle.py
        threshold_results = find_optimal_threshold(val_y, val_pred_proba, metric=metric)

        logger.info(
            f"Optimal threshold: {threshold_results['threshold']:.4f} "
            f"({metric} = {threshold_results[f'{metric}_score']:.4f})"
        )

        return threshold_results["threshold"]

    def evaluate(
        self,
        pipeline: Pipeline,
        test_df: pl.DataFrame,
        threshold: float = 0.5,
        target_col: str = "target",
    ) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            pipeline: Fitted pipeline
            test_df: Test DataFrame
            threshold: Classification threshold
            target_col: Name of target column

        Returns:
            Dictionary of metrics
        """
        test_X, test_y = self._prepare_data(test_df, target_col)

        # Get predictions
        test_pred_proba = pipeline.predict_proba(test_X)[:, 1]
        test_pred = (test_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(test_y, test_pred),
            "precision": precision_score(test_y, test_pred, zero_division=0),
            "recall": recall_score(test_y, test_pred, zero_division=0),
            "f1": f1_score(test_y, test_pred, zero_division=0),
            "f2": fbeta_score(test_y, test_pred, beta=2.0, zero_division=0),
            "auc": roc_auc_score(test_y, test_pred_proba),
            "log_loss": log_loss(test_y, test_pred_proba),
            "threshold": threshold,
        }

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)

        logger.info(f"Test Evaluation (threshold={threshold:.4f}):")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  F2: {metrics['f2']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  Confusion: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        return metrics

    def predict(
        self,
        pipeline: Pipeline,
        games_df: pl.DataFrame,
        threshold: float = 0.5,
    ) -> pl.DataFrame:
        """Generate ownership predictions for all games.

        Args:
            pipeline: Fitted pipeline
            games_df: DataFrame with game features
            threshold: Classification threshold

        Returns:
            DataFrame with game_id, ownership_probability, predicted_owned
        """
        logger.info(f"Generating predictions for {len(games_df)} games")

        # Prepare features (no target column)
        games_pandas = games_df.to_pandas()

        # Get predictions
        pred_proba = pipeline.predict_proba(games_pandas)[:, 1]
        pred_class = (pred_proba >= threshold).astype(int)

        # Build result DataFrame
        result = games_df.select(["game_id"]).with_columns(
            [
                pl.Series("ownership_probability", pred_proba),
                pl.Series("predicted_owned", pred_class.astype(bool)),
            ]
        )

        # Add rank
        result = result.with_columns(
            pl.col("ownership_probability")
            .rank(method="ordinal", descending=True)
            .alias("ownership_rank")
        )

        logger.info(f"Generated predictions. Top probability: {pred_proba.max():.4f}")

        return result

    def get_feature_importance(
        self, pipeline: Pipeline
    ) -> Optional[pl.DataFrame]:
        """Extract feature importance from trained model.

        Args:
            pipeline: Fitted pipeline

        Returns:
            DataFrame with feature names and importance values, or None
        """
        try:
            model = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocessor"]

            # Get feature names
            feature_names = preprocessor.named_steps[
                "bgg_preprocessor"
            ].get_feature_names_out()

            # Get importance based on model type
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0])
            else:
                logger.warning("Model does not have feature importance attribute")
                return None

            # Build DataFrame
            fi_df = pl.DataFrame({
                "feature": feature_names,
                "importance": importance,
            }).sort("importance", descending=True)

            return fi_df

        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return None

    def get_model_metadata(
        self,
        pipeline: Pipeline,
        best_params: Dict[str, Any],
        metrics: Dict[str, float],
        threshold: float,
    ) -> Dict[str, Any]:
        """Build metadata dictionary for model registration.

        Args:
            pipeline: Fitted pipeline
            best_params: Best hyperparameters from tuning
            metrics: Evaluation metrics
            threshold: Optimal threshold

        Returns:
            Metadata dictionary
        """
        return {
            "username": self.username,
            "model_type": self.config.model_type,
            "preprocessor_type": self.config.preprocessor_type,
            "threshold_optimization_metric": self.config.threshold_optimization_metric,
            "tuning_metric": self.config.tuning_metric,
            "best_params": best_params,
            "metrics": metrics,
            "threshold": threshold,
        }
