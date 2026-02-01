"""Complexity model for predicting game weight/complexity."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge, ARDRegression
from catboost import CatBoostRegressor
import lightgbm as lgb

from src.models.outcomes.base import DataConfig, TrainableModel, TrainingConfig


class ComplexityModel(TrainableModel):
    """Regression model predicting game complexity (weight) on 1-5 scale.

    Complexity is trained independently and its predictions are used
    as features for the rating and users_rated models.
    """

    model_type = "complexity"
    target_column = "complexity"
    model_task = "regression"

    data_config = DataConfig(
        min_weights=5,
        requires_complexity_predictions=False,
        supports_embeddings=True,
    )

    def __init__(self, training_config: TrainingConfig = None, **kwargs):
        """Initialize ComplexityModel.

        Args:
            training_config: Training configuration.
            **kwargs: Additional arguments passed to TrainableModel.
        """
        super().__init__(training_config=training_config, **kwargs)

    def configure_model(
        self, algorithm: str, algorithm_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Configure regressor and parameter grid.

        Args:
            algorithm: Algorithm name.
            algorithm_params: Optional algorithm-specific parameters from config.
                For bayesian_ridge, can include: alpha_1, alpha_2, lambda_1, lambda_2.

        Returns:
            Tuple of (regressor_instance, param_grid).
        """
        MODEL_MAPPING = {
            "linear": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "bayesian_ridge": BayesianRidge,
            "ard": ARDRegression,
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
            "bayesian_ridge": {
                # BayesianRidge learns alpha/lambda automatically via EM
                "model__fit_intercept": [True],
            },
            "ard": {
                # ARDRegression uses per-feature relevance determination
                "model__fit_intercept": [True],
                "model__threshold_lambda": [10000, 100000],
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

        # Create model instance with optional config params
        model_class = MODEL_MAPPING[algorithm]
        if algorithm_params:
            model = model_class(**algorithm_params)
        else:
            model = model_class()

        param_grid = PARAM_GRIDS[algorithm]

        return model, param_grid

    def post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Clip predictions to valid complexity range [1, 5].

        Args:
            predictions: Raw model predictions.

        Returns:
            Predictions clipped to [1, 5].
        """
        return np.clip(predictions, 1, 5)

    def post_process_uncertainty(
        self, std: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Reduce uncertainty near complexity bounds [1, 5].

        Args:
            std: Standard deviation from Bayesian model.
            predictions: Raw predictions (before clipping).

        Returns:
            Adjusted standard deviations.
        """
        dist_to_bound = np.minimum(predictions - 1.0, 5.0 - predictions)
        scale_factor = np.clip(dist_to_bound / (2 * std + 1e-8), 0.1, 1.0)
        return std * scale_factor


if __name__ == "__main__":
    from src.models.outcomes.train import train_model

    train_model(ComplexityModel)
