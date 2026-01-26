"""Complexity model for predicting game weight/complexity."""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, LinearRegression
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
        supports_embeddings=False,  # Complexity doesn't use embeddings
    )

    def __init__(self, training_config: TrainingConfig = None, **kwargs):
        """Initialize ComplexityModel.

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
        """Clip predictions to valid complexity range [1, 5].

        Args:
            predictions: Raw model predictions.

        Returns:
            Predictions clipped to [1, 5].
        """
        return np.clip(predictions, 1, 5)


if __name__ == "__main__":
    from src.models.outcomes.train import train_model

    train_model(ComplexityModel)
