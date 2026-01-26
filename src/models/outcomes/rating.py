"""Rating model for predicting game average rating."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import lightgbm as lgb
import logging

from src.models.outcomes.base import DataConfig, TrainableModel, TrainingConfig


logger = logging.getLogger(__name__)


class RatingModel(TrainableModel):
    """Regression model predicting game average rating on 1-10 scale.

    Uses complexity predictions as an input feature, so complexity
    model must be trained first.
    """

    model_type = "rating"
    target_column = "rating"
    model_task = "regression"

    data_config = DataConfig(
        min_ratings=5,
        requires_complexity_predictions=True,
        supports_embeddings=True,
    )

    def __init__(self, training_config: TrainingConfig = None, **kwargs):
        """Initialize RatingModel.

        Args:
            training_config: Training configuration.
            **kwargs: Additional arguments passed to TrainableModel.
        """
        super().__init__(training_config=training_config, **kwargs)

    def configure_model(self, algorithm: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Configure regressor and parameter grid.

        Args:
            algorithm: Algorithm name.

        Returns:
            Tuple of (regressor_instance, param_grid).
        """
        MODEL_MAPPING = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "catboost": CatBoostRegressor,
            "lightgbm": lgb.LGBMRegressor,
            "lightgbm_linear": lgb.LGBMRegressor,
        }

        PARAM_GRIDS = {
            "linear": {},
            "ridge": {
                "model__alpha": [0.0001, 0.0005, 0.01, 0.1, 1.0, 5],
                "model__solver": ["auto"],
                "model__fit_intercept": [True],
            },
            "lasso": {
                "model__alpha": [0.1, 1.0, 10.0],
                "model__selection": ["cyclic", "random"],
            },
            "catboost": {
                "model__iterations": [500],
                "model__learning_rate": [0.01, 0.5],
                "model__depth": [4, 6, 8],
                "model__l2_leaf_reg": [1, 3, 5],
            },
            "lightgbm": {
                "model__n_estimators": [500],
                "model__learning_rate": [0.01],
                "model__max_depth": [-1],
                "model__num_leaves": [50, 100],
                "model__min_child_samples": [10],
                "model__reg_alpha": [0.1],
            },
            "lightgbm_linear": {
                "model__n_estimators": [500, 1000],
                "model__learning_rate": [0.01, 0.05],
                "model__max_depth": [3, 5, 7],
                "model__num_leaves": [31, 50],
                "model__min_child_samples": [10, 20],
                "model__reg_alpha": [0.1, 1.0],
                "model__linear_tree": [True],
            },
        }

        if algorithm not in MODEL_MAPPING:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {list(MODEL_MAPPING.keys())}"
            )

        model = MODEL_MAPPING[algorithm]()
        param_grid = PARAM_GRIDS[algorithm]

        return model, param_grid

    def post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Clip predictions to valid rating range [1, 10].

        Args:
            predictions: Raw model predictions.

        Returns:
            Predictions clipped to [1, 10].
        """
        return np.clip(predictions, 1, 10)

    def compute_additional_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Compute stratified metrics by rating confidence level.

        Args:
            y_true: True ratings.
            y_pred: Predicted ratings.
            dataset_name: Name of dataset.

        Returns:
            Dictionary with stratified metrics.
        """
        # This requires users_rated to be available for stratification
        # Will be implemented when full training context is available
        return {}

    def stratified_evaluation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        users_rated: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model across different rating count buckets.

        Args:
            X: Features.
            y: True target values.
            users_rated: Number of users who rated each game.

        Returns:
            Dictionary of metrics for each confidence bucket.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded")

        buckets = {
            "high_confidence": users_rated >= 25,
            "medium_confidence": (users_rated >= 15) & (users_rated < 25),
            "low_confidence": (users_rated >= 10) & (users_rated < 15),
        }

        stratified_metrics = {}
        for bucket_name, mask in buckets.items():
            mask_index = mask.to_numpy()

            if mask_index.sum() == 0:
                logger.info(f"No games in {bucket_name} bucket")
                continue

            X_bucket = X.loc[mask_index]
            y_bucket = y.loc[mask_index]
            y_pred = self.pipeline.predict(X_bucket)

            bucket_metrics = {
                "rmse": np.sqrt(mean_squared_error(y_bucket, y_pred)),
                "mae": mean_absolute_error(y_bucket, y_pred),
                "r2": r2_score(y_bucket, y_pred),
                "n_samples": len(y_bucket),
            }

            logger.info(f"{bucket_name}: RMSE={bucket_metrics['rmse']:.4f}, n={bucket_metrics['n_samples']}")
            stratified_metrics[bucket_name] = bucket_metrics

        return stratified_metrics


if __name__ == "__main__":
    from src.models.outcomes.train import train_model

    train_model(RatingModel)
