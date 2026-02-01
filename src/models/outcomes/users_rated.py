"""Users rated model for predicting number of user ratings."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge, ARDRegression
from catboost import CatBoostRegressor
import lightgbm as lgb

from src.models.outcomes.base import DataConfig, TrainableModel, TrainingConfig


class UsersRatedModel(TrainableModel):
    """Regression model predicting log(users_rated).

    Predicts the log-transformed number of users who will rate a game.
    Uses complexity predictions as an input feature.
    """

    model_type = "users_rated"
    target_column = "log_users_rated"
    model_task = "regression"

    data_config = DataConfig(
        min_ratings=0,
        requires_complexity_predictions=True,
        supports_embeddings=True,
    )

    def __init__(self, training_config: TrainingConfig = None, **kwargs):
        """Initialize UsersRatedModel.

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
                "model__alpha": [0.0001, 0.0005, 0.01, 0.1, 5],
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
        """Transform log predictions back to count scale, rounded to nearest 50.

        Target is log1p(users_rated), so inverse is expm1.

        Args:
            predictions: Log-scale predictions (from log1p transform).

        Returns:
            User count predictions, minimum 25, rounded to nearest 50.
        """
        # Inverse of log1p is expm1
        raw_counts = np.expm1(predictions)

        # Round to nearest 50 and ensure minimum of 25
        rounded = np.maximum(np.round(raw_counts / 50) * 50, 25)

        return rounded

    def predict_log_scale(self, features) -> np.ndarray:
        """Get predictions on log scale (without post-processing).

        Args:
            features: Input features.

        Returns:
            Log-scale predictions.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded")
        return self.pipeline.predict(features)

    def post_process_uncertainty(
        self, std: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Transform uncertainty from log scale to count scale.

        Uses delta method: since target is log1p(users_rated),
        std in count space is approximately std_log * exp(prediction).

        Args:
            std: Standard deviation in log scale from Bayesian model.
            predictions: Raw log-scale predictions.

        Returns:
            Standard deviation in count scale.
        """
        # Delta method: d/dx(expm1(x)) = exp(x)
        count_std = std * np.exp(predictions)
        return np.maximum(count_std, 25)


if __name__ == "__main__":
    from src.models.outcomes.train import train_model

    train_model(UsersRatedModel)
