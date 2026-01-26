"""Base classes for outcome prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class DataConfig:
    """Configuration for model data requirements.

    Defines what data a model needs for training, including filters
    and optional feature joins.
    """

    # Filtering thresholds
    min_ratings: Optional[int] = None
    min_weights: Optional[int] = None

    # Feature dependencies
    requires_complexity_predictions: bool = False
    supports_embeddings: bool = False

    def __post_init__(self):
        """Validate that at least one filter is specified."""
        if self.min_ratings is None and self.min_weights is None:
            raise ValueError("Must specify either min_ratings or min_weights")


@dataclass
class TrainingConfig:
    """Configuration for model training parameters.

    Loaded from config.yaml, contains tunables like algorithm choice,
    hyperparameter grids, and training options.
    """

    algorithm: str = "ridge"
    experiment_name: str = "experiment"
    use_sample_weights: bool = False
    sample_weight_column: Optional[str] = None
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)

    # Year-based splits
    train_end_year: int = 2022
    tune_start_year: int = 2022
    tune_end_year: int = 2023
    test_start_year: int = 2024
    test_end_year: int = 2025


class Predictor(Protocol):
    """Protocol defining the interface for all predictive models.

    Both TrainableModel and CompositeModel implement this interface,
    allowing downstream code (scoring service, viewer app) to work
    with either type interchangeably.
    """

    model_type: str
    version: Optional[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given features."""
        ...

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        ...

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata (experiment info, metrics, etc.)."""
        ...


class TrainableModel(ABC):
    """Base class for models that are trained on data.

    Subclasses define their data requirements and target column.
    The training flow is standardized in the train() method.
    """

    # Subclasses must define these
    model_type: str  # e.g., "hurdle", "complexity", "rating", "users_rated"
    target_column: str  # e.g., "hurdle", "complexity", "rating", "log_users_rated"
    model_task: Literal["classification", "regression"]
    data_config: DataConfig

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        pipeline: Optional[Pipeline] = None,
        version: Optional[str] = None,
    ):
        """Initialize the model.

        Args:
            training_config: Configuration for training parameters.
            pipeline: Fitted sklearn pipeline (preprocessor + model).
            version: Model version string.
        """
        self.training_config = training_config
        self.pipeline = pipeline
        self.version = version
        self._metadata: Dict[str, Any] = {}

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the given features.

        Args:
            features: DataFrame with input features.

        Returns:
            Array of predictions.

        Raises:
            ValueError: If model has not been trained/loaded.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded")
        predictions = self.pipeline.predict(features)
        return self.post_process_predictions(predictions)

    def post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Post-process predictions (e.g., clipping, transformations).

        Override in subclasses for model-specific post-processing.

        Args:
            predictions: Raw model predictions.

        Returns:
            Post-processed predictions.
        """
        return predictions

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk.

        Args:
            path: Path to the experiment directory.
        """
        from src.models.experiments import ExperimentTracker

        tracker = ExperimentTracker(self.model_type)
        experiment = tracker.load_experiment(str(path))
        self.pipeline = experiment.pipeline
        self._metadata = experiment.metadata
        self.version = experiment.version

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata."""
        return self._metadata

    @abstractmethod
    def configure_model(self, algorithm: str) -> tuple:
        """Configure the model and parameter grid for the given algorithm.

        Args:
            algorithm: Algorithm name (e.g., "ridge", "catboost", "lightgbm").

        Returns:
            Tuple of (model_instance, param_grid).
        """
        ...

    def compute_additional_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Compute model-specific metrics beyond standard ones.

        Override in subclasses for custom metrics (e.g., threshold
        optimization for hurdle, stratified evaluation for rating).

        Args:
            y_true: True target values.
            y_pred: Predicted values.
            dataset_name: Name of the dataset (train/tune/test).

        Returns:
            Dictionary of additional metrics.
        """
        return {}


class CompositeModel(ABC):
    """Base class for models that combine predictions from other models.

    Currently used for GeekRatingModel which orchestrates hurdle,
    complexity, rating, and users_rated models.
    """

    model_type: str = "composite"

    def __init__(
        self,
        sub_models: Optional[Dict[str, Predictor]] = None,
        version: Optional[str] = None,
    ):
        """Initialize the composite model.

        Args:
            sub_models: Dictionary mapping model names to Predictor instances.
            version: Model version string.
        """
        self.sub_models = sub_models or {}
        self.version = version
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    def combine(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine predictions from sub-models.

        Args:
            predictions: Dictionary mapping model names to their predictions.

        Returns:
            Combined predictions.
        """
        ...

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions by running sub-models and combining.

        Args:
            features: DataFrame with input features.

        Returns:
            Combined predictions.
        """
        predictions = {}
        for name, model in self.sub_models.items():
            predictions[name] = model.predict(features)
        return self.combine(predictions)

    def load(self, path: Union[str, Path]) -> None:
        """Load composite model configuration.

        Args:
            path: Path to the composite model configuration.
        """
        # Implementation depends on how composite models are stored
        raise NotImplementedError("Subclasses must implement load()")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata including sub-model info."""
        return {
            "model_type": self.model_type,
            "version": self.version,
            "sub_models": {
                name: model.metadata for name, model in self.sub_models.items()
            },
            **self._metadata,
        }

    def load_sub_models(self, experiments: Dict[str, str]) -> None:
        """Load sub-models from experiment names.

        Args:
            experiments: Dictionary mapping model types to experiment names.
                e.g., {"hurdle": "lightgbm-hurdle", "complexity": "catboost-complexity"}
        """
        from src.models.score import load_model

        for model_type, experiment_name in experiments.items():
            self.sub_models[model_type] = load_model(experiment_name, model_type)
