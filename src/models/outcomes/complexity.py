"""Complexity model for predicting game weight/complexity."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator

from src.models.outcomes.base import (
    DataConfig,
    TrainableModel,
    TrainingConfig,
    configure_regressor,
)


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

        Returns:
            Tuple of (regressor_instance, param_grid).
        """
        return configure_regressor(algorithm, algorithm_params)

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
