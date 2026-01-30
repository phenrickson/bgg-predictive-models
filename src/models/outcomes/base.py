"""Base classes for outcome prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Union
import logging

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


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

    # Year-based splits (all years are inclusive)
    train_through: int = 2021
    tune_start: int = 2022
    tune_through: int = 2023
    test_start: int = 2024
    test_through: int = 2024


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

    def finalize(
        self,
        experiment_name: str,
        end_year: Optional[int] = None,
        description: Optional[str] = None,
        complexity_predictions_path: Optional[Union[str, Path]] = None,
        use_embeddings: Optional[bool] = None,
        local_data_path: Optional[Union[str, Path]] = None,
        recent_year_threshold: int = 2,
        version: Optional[int] = None,
    ) -> Path:
        """Finalize a model by refitting on full dataset for production.

        Loads the trained experiment and refits the pipeline on all available
        data up to end_year. Uses the model's data_config to ensure correct
        data loading (filters, joins, embeddings).

        Args:
            experiment_name: Name of the experiment to finalize.
            end_year: End year for training data. If None, defaults to
                current year minus recent_year_threshold.
            description: Optional description for the finalized model.
            complexity_predictions_path: Path to complexity predictions.
                Required if model requires complexity predictions.
            use_embeddings: Whether to include embeddings in features.
                If None, reads from experiment metadata.
            local_data_path: Optional path to local data file.
            recent_year_threshold: Years to exclude from current year
                when end_year is None. Default 2.
            version: Optional specific experiment version to finalize.

        Returns:
            Path to the finalized model directory.

        Raises:
            ValueError: If required complexity predictions not provided.
        """
        from src.models.experiments import ExperimentTracker
        from src.models.outcomes.data import load_training_data, select_X_y
        from src.models.training import calculate_sample_weights

        # Load experiment
        tracker = ExperimentTracker(self.model_type)

        # Handle experiment names that include version
        base_experiment_name = experiment_name
        if "_v" in experiment_name:
            base_experiment_name = experiment_name.split("_v")[0]

        experiment = tracker.load_experiment(base_experiment_name, version)
        logger.info(f"Loaded experiment: {experiment.name}")

        # Determine end year
        current_year = datetime.now().year
        if end_year is None:
            end_year = current_year - recent_year_threshold
        elif current_year - end_year <= recent_year_threshold:
            logger.info(
                f"End year {end_year} is within {recent_year_threshold} years "
                f"of current year. Adjusting to {current_year - recent_year_threshold}"
            )
            end_year = current_year - recent_year_threshold

        logger.info(f"Finalizing model with data through {end_year}")

        # Get complexity predictions path from experiment metadata if not provided
        if self.data_config.requires_complexity_predictions:
            if complexity_predictions_path is None:
                # Try to get from experiment metadata
                complexity_predictions_path = experiment.metadata.get(
                    "complexity_predictions_path"
                ) or experiment.metadata.get("config", {}).get(
                    "complexity_predictions_path"
                )
                # Also check for complexity_experiment and construct path
                complexity_experiment = experiment.metadata.get(
                    "complexity_experiment"
                ) or experiment.metadata.get("config", {}).get("complexity_experiment")
                if complexity_experiment and not complexity_predictions_path:
                    from src.utils.config import load_config
                    config = load_config()
                    complexity_predictions_path = (
                        f"{config.predictions_dir}/{complexity_experiment}.parquet"
                    )

            if complexity_predictions_path is None:
                raise ValueError(
                    f"Model type '{self.model_type}' requires complexity predictions "
                    "but no path provided and none found in experiment metadata."
                )
            logger.info(f"Using complexity predictions: {complexity_predictions_path}")

        # Determine use_embeddings from experiment metadata if not specified
        trained_with_embeddings = experiment.metadata.get("use_embeddings", False)
        if use_embeddings is None:
            use_embeddings = trained_with_embeddings
            logger.info(f"Using embeddings setting from experiment: {use_embeddings}")
        elif use_embeddings and not trained_with_embeddings:
            logger.warning(
                "use_embeddings=True but experiment was not trained with embeddings. "
                "Using embeddings anyway for finalization."
            )
        elif trained_with_embeddings and not use_embeddings:
            logger.warning(
                "Experiment was trained with embeddings but use_embeddings=False. "
                "Finalized model will NOT include embeddings."
            )

        # Load data using model's data config
        df = load_training_data(
            data_config=self.data_config,
            end_year=end_year,
            use_embeddings=use_embeddings,
            complexity_predictions_path=complexity_predictions_path,
            local_data_path=local_data_path,
        )

        logger.info(f"Loaded {len(df)} rows for finalization")
        logger.info(
            f"Year range: {df['year_published'].min()} - {df['year_published'].max()}"
        )

        # Prepare X and y
        X, y = select_X_y(df, self.target_column)
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

        # Extract sample weights if used during training
        sample_weights = None
        sample_weights_info = experiment.metadata.get("sample_weights", {})
        weight_column = sample_weights_info.get("column")

        if weight_column and weight_column in df.columns:
            logger.info(f"Calculating sample weights from column: {weight_column}")
            sample_weights = calculate_sample_weights(
                df.to_pandas(), weight_column=weight_column
            )
            logger.info(
                f"Sample weights: min={sample_weights.min():.4f}, "
                f"max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}"
            )

        # Build description
        if description is None:
            description = f"Production {self.model_type} model trained through {end_year}"

        # Add metadata to description
        description_parts = [description]
        threshold = experiment.metadata.get("model_info", {}).get("threshold")
        if threshold is not None:
            description_parts.append(f"Optimal threshold: {threshold:.4f}")
        if self.data_config.min_ratings is not None:
            description_parts.append(f"Min ratings: {self.data_config.min_ratings}")
        if self.data_config.min_weights is not None:
            description_parts.append(f"Min weights: {self.data_config.min_weights}")

        final_description = ". ".join(description_parts)

        # Finalize the model
        logger.info("Fitting pipeline on full dataset...")
        finalized_dir = experiment.finalize_model(
            X=X,
            y=y,
            description=final_description,
            final_end_year=end_year,
            sample_weight=np.asarray(sample_weights) if sample_weights is not None else None,
        )

        logger.info(f"Model finalized and saved to {finalized_dir}")
        return finalized_dir


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
