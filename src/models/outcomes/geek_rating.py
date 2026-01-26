"""Geek rating model - composite model combining predictions from sub-models."""

from typing import Any, Dict, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import logging

from src.models.outcomes.base import CompositeModel, Predictor


logger = logging.getLogger(__name__)


class GeekRatingModel(CompositeModel):
    """Composite model that combines hurdle, complexity, rating, and users_rated.

    Currently uses a Bayesian average formula to compute geek rating.
    Architected to support future enhancements:
    - Trained combiner (stacking/ensemble)
    - End-to-end training directly on geek_rating
    """

    model_type = "geek_rating"

    def __init__(
        self,
        sub_models: Optional[Dict[str, Predictor]] = None,
        prior_rating: float = 5.5,
        prior_weight: float = 2000,
        hurdle_threshold: Optional[float] = None,
        version: Optional[str] = None,
    ):
        """Initialize GeekRatingModel.

        Args:
            sub_models: Dictionary of sub-models (hurdle, complexity, rating, users_rated).
            prior_rating: Prior mean rating for Bayesian average.
            prior_weight: Weight given to prior rating.
            hurdle_threshold: Threshold for hurdle model classification.
                If None, uses the hurdle model's optimal_threshold.
            version: Model version string.
        """
        super().__init__(sub_models=sub_models, version=version)
        self.prior_rating = prior_rating
        self.prior_weight = prior_weight
        self.hurdle_threshold = hurdle_threshold

    def combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine sub-model predictions using Bayesian average.

        Args:
            predictions: Dictionary with keys 'rating' and 'users_rated'.

        Returns:
            Geek rating predictions.
        """
        rating = predictions["rating"]
        users_rated = predictions["users_rated"]

        geek_rating = (
            (rating * users_rated) + (self.prior_rating * self.prior_weight)
        ) / (users_rated + self.prior_weight)

        return geek_rating

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate all predictions including intermediate values.

        Args:
            features: Input features.

        Returns:
            DataFrame with columns:
            - game_id, name, year_published (preserved)
            - predicted_hurdle_prob
            - predicted_complexity
            - predicted_rating
            - predicted_users_rated
            - predicted_geek_rating
        """
        self._validate_sub_models()

        # Get hurdle threshold
        threshold = self.hurdle_threshold
        if threshold is None and hasattr(self.sub_models["hurdle"], "optimal_threshold"):
            threshold = self.sub_models["hurdle"].optimal_threshold
        if threshold is None:
            threshold = 0.5

        # Initialize results
        results = pd.DataFrame(index=features.index)

        # Preserve identifying fields
        for col in ["game_id", "name", "year_published"]:
            if col in features.columns:
                results[col] = features[col]

        # Step 1: Predict hurdle probability
        hurdle_model = self.sub_models["hurdle"]
        if hasattr(hurdle_model, "predict_proba"):
            hurdle_proba = hurdle_model.predict_proba(features)[:, 1]
        else:
            hurdle_proba = hurdle_model.predict(features)

        results["predicted_hurdle_prob"] = hurdle_proba

        # Identify likely games (above threshold)
        likely_mask = hurdle_proba >= threshold

        # Step 2: For likely games, predict complexity
        complexity_model = self.sub_models["complexity"]
        results["predicted_complexity"] = 1.0  # Default

        if likely_mask.any():
            likely_features = features[likely_mask]
            results.loc[likely_mask, "predicted_complexity"] = complexity_model.predict(
                likely_features
            )

        # Step 3: Add complexity to features for rating/users_rated
        features_with_complexity = features.copy()
        features_with_complexity["predicted_complexity"] = results["predicted_complexity"]

        # Step 4: Predict rating and users_rated for likely games
        rating_model = self.sub_models["rating"]
        users_rated_model = self.sub_models["users_rated"]

        # Defaults for unlikely games
        results["predicted_rating"] = 5.5
        results["predicted_users_rated"] = 25

        if likely_mask.any():
            likely_features_with_complexity = features_with_complexity[likely_mask]

            results.loc[likely_mask, "predicted_rating"] = rating_model.predict(
                likely_features_with_complexity
            )
            results.loc[likely_mask, "predicted_users_rated"] = users_rated_model.predict(
                likely_features_with_complexity
            )

        # Step 5: Calculate geek rating
        predictions_dict = {
            "rating": results["predicted_rating"].values,
            "users_rated": results["predicted_users_rated"].values,
        }
        results["predicted_geek_rating"] = self.combine(predictions_dict)

        return results

    def predict_simple(self, features: pd.DataFrame) -> np.ndarray:
        """Generate just geek rating predictions (implements Predictor interface).

        Args:
            features: Input features.

        Returns:
            Array of geek rating predictions.
        """
        results = self.predict(features)
        return results["predicted_geek_rating"].values

    def _validate_sub_models(self) -> None:
        """Validate that all required sub-models are loaded."""
        required = ["hurdle", "complexity", "rating", "users_rated"]
        missing = [m for m in required if m not in self.sub_models]

        if missing:
            raise ValueError(f"Missing required sub-models: {missing}")

    def load(self, path: Union[str, Path]) -> None:
        """Load composite model from experiment directory.

        Args:
            path: Path to geek_rating experiment directory.
        """
        from src.models.experiments import ExperimentTracker

        tracker = ExperimentTracker("geek_rating")
        experiment = tracker.load_experiment(str(path))

        # Load sub-model references from metadata
        sub_model_experiments = experiment.metadata.get("model_experiments", {})
        self.load_sub_models(sub_model_experiments)

        # Load prediction parameters
        params = experiment.metadata.get("prediction_parameters", {})
        self.prior_rating = params.get("prior_rating", 5.5)
        self.prior_weight = params.get("prior_weight", 2000)
        self.hurdle_threshold = params.get("threshold")

        self._metadata = experiment.metadata
        self.version = experiment.version

    def load_sub_models(self, experiments: Dict[str, str]) -> None:
        """Load sub-models from experiment names and extract hurdle threshold.

        Overrides base class to also load optimal threshold from hurdle experiment.

        Args:
            experiments: Dictionary mapping model types to experiment names.
                e.g., {"hurdle": "lightgbm-hurdle", "complexity": "catboost-complexity"}
        """
        from src.models.score import load_model, extract_threshold

        logger.info("Loading sub-models:")
        for model_type, experiment_name in experiments.items():
            logger.info(f"  {model_type}: {experiment_name}")
            self.sub_models[model_type] = load_model(experiment_name, model_type)

        # Extract optimal threshold from hurdle experiment if not already set
        if self.hurdle_threshold is None and "hurdle" in experiments:
            loaded_threshold = extract_threshold(experiments["hurdle"], "hurdle")
            if loaded_threshold is not None:
                self.hurdle_threshold = loaded_threshold
                logger.info(f"Using threshold from hurdle experiment: {self.hurdle_threshold}")

        self._metadata["model_experiments"] = experiments

    @classmethod
    def from_experiments(
        cls,
        hurdle_experiment: str,
        complexity_experiment: str,
        rating_experiment: str,
        users_rated_experiment: str,
        prior_rating: float = 5.5,
        prior_weight: float = 2000,
        hurdle_threshold: Optional[float] = None,
    ) -> "GeekRatingModel":
        """Create GeekRatingModel from experiment names.

        Args:
            hurdle_experiment: Experiment name for hurdle model.
            complexity_experiment: Experiment name for complexity model.
            rating_experiment: Experiment name for rating model.
            users_rated_experiment: Experiment name for users_rated model.
            prior_rating: Prior mean rating for Bayesian average.
            prior_weight: Weight given to prior rating.
            hurdle_threshold: Optional threshold override. If None, loads from
                hurdle experiment metadata.

        Returns:
            Initialized GeekRatingModel with loaded sub-models.
        """
        model = cls(
            prior_rating=prior_rating,
            prior_weight=prior_weight,
            hurdle_threshold=hurdle_threshold,
        )

        experiments = {
            "hurdle": hurdle_experiment,
            "complexity": complexity_experiment,
            "rating": rating_experiment,
            "users_rated": users_rated_experiment,
        }

        model.load_sub_models(experiments)

        return model

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata including configuration."""
        base_metadata = super().metadata
        base_metadata.update({
            "prior_rating": self.prior_rating,
            "prior_weight": self.prior_weight,
            "hurdle_threshold": self.hurdle_threshold,
        })
        return base_metadata
