"""Users rated model for predicting number of user ratings."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from src.models.outcomes.base import (
    DataConfig,
    TrainableModel,
    TrainingConfig,
    configure_regressor,
)


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

        Returns:
            Tuple of (regressor_instance, param_grid).
        """
        return configure_regressor(algorithm, algorithm_params)

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
