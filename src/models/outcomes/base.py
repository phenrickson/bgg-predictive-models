"""Base classes for outcome prediction models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Protocol, Union
import logging

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """Container for predictions with optional uncertainty estimates."""

    values: np.ndarray
    std: Optional[np.ndarray] = None

    def confidence_interval(
        self, level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence interval at the given level.

        Args:
            level: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound) arrays.

        Raises:
            ValueError: If no uncertainty estimates available.
        """
        if self.std is None:
            raise ValueError("No uncertainty estimates available")
        from scipy import stats

        z = stats.norm.ppf((1 + level) / 2)
        return self.values - z * self.std, self.values + z * self.std


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

    def post_process_uncertainty(
        self, std: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """Post-process uncertainty estimates.

        Override in subclasses for model-specific adjustments (e.g.,
        reducing std near prediction bounds, transforming from log scale).

        Args:
            std: Standard deviation from Bayesian model.
            predictions: Raw predictions (before post-processing).

        Returns:
            Adjusted standard deviations.
        """
        return std

    def predict_with_uncertainty(
        self, features: pd.DataFrame
    ) -> PredictionResult:
        """Generate predictions with uncertainty estimates.

        Only works with Bayesian models (e.g., BayesianRidge) that support
        the return_std parameter.

        Args:
            features: DataFrame with input features.

        Returns:
            PredictionResult with values and optional std.

        Raises:
            ValueError: If model has not been trained/loaded.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded")

        preprocessor = self.pipeline.named_steps.get("preprocessor")
        model = self.pipeline.named_steps.get("model")

        # Transform features through preprocessor
        X_transformed = preprocessor.transform(features) if preprocessor else features

        # Check if model supports return_std (BayesianRidge does)
        if hasattr(model, "predict"):
            import inspect

            sig = inspect.signature(model.predict)
            if "return_std" in sig.parameters:
                predictions, std = model.predict(X_transformed, return_std=True)
                processed_values = self.post_process_predictions(predictions)
                processed_std = self.post_process_uncertainty(std, predictions)
                return PredictionResult(values=processed_values, std=processed_std)

        # Fall back to point predictions
        predictions = self.pipeline.predict(features)
        return PredictionResult(values=self.post_process_predictions(predictions))

    def sample_posterior_predictive(
        self,
        features: pd.DataFrame,
        n_samples: int = 1000,
        include_noise: bool = True,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Sample from the posterior predictive distribution.

        Uses the weight posterior (coef_, sigma_) from BayesianRidge to draw
        samples. Each sample represents a plausible prediction given the
        uncertainty in the model parameters.

        Args:
            features: DataFrame with input features.
            n_samples: Number of posterior samples to draw.
            include_noise: If True, add observation noise (alpha_) to samples.
                If False, only sample from the mean function posterior.
            random_state: Random seed for reproducibility.

        Returns:
            Array of shape (n_observations, n_samples) with posterior samples.

        Raises:
            ValueError: If model doesn't support posterior sampling.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded")

        preprocessor = self.pipeline.named_steps.get("preprocessor")
        model = self.pipeline.named_steps.get("model")

        # Check for required attributes (BayesianRidge stores these)
        if not (hasattr(model, "coef_") and hasattr(model, "sigma_")):
            raise ValueError(
                f"Model {type(model).__name__} does not support posterior sampling. "
                "Requires coef_ and sigma_ attributes (e.g., BayesianRidge)."
            )

        # Transform features
        X_transformed = preprocessor.transform(features) if preprocessor else features
        if hasattr(X_transformed, "values"):
            X_transformed = X_transformed.values

        rng = np.random.default_rng(random_state)

        # Sample weight vectors from posterior: w ~ N(coef_, sigma_)
        # Handle ARDRegression which prunes features - sigma_ may be smaller than coef_
        coef = model.coef_
        sigma = model.sigma_

        if sigma.shape[0] != len(coef):
            # ARDRegression: sigma_ only covers active (non-pruned) features
            # Build full covariance matrix with zeros for pruned features
            n_features = len(coef)
            full_sigma = np.zeros((n_features, n_features))
            if hasattr(model, "lambda_"):
                active_mask = model.lambda_ < getattr(model, "threshold_lambda", np.inf)
                active_indices = np.where(active_mask)[0]
                for i, ai in enumerate(active_indices):
                    for j, aj in enumerate(active_indices):
                        full_sigma[ai, aj] = sigma[i, j]
            else:
                # Fallback: assume first features are active
                n_active = sigma.shape[0]
                full_sigma[:n_active, :n_active] = sigma
            sigma = full_sigma

        weight_samples = rng.multivariate_normal(
            coef, sigma, size=n_samples
        )  # shape: (n_samples, n_features)

        # Compute predictions for each weight sample
        # y = X @ w.T + intercept
        predictions = X_transformed @ weight_samples.T  # shape: (n_obs, n_samples)
        if hasattr(model, "intercept_"):
            predictions += model.intercept_

        # Optionally add observation noise
        if include_noise and hasattr(model, "alpha_"):
            noise_std = 1.0 / np.sqrt(model.alpha_)
            predictions += rng.normal(0, noise_std, size=predictions.shape)

        # Post-process each sample (e.g., clipping)
        for i in range(n_samples):
            predictions[:, i] = self.post_process_predictions(predictions[:, i])

        return predictions

    @property
    def supports_uncertainty(self) -> bool:
        """Check if the fitted model supports uncertainty estimation."""
        if self.pipeline is None:
            return False
        model = self.pipeline.named_steps.get("model")
        if model is None:
            return False
        # BayesianRidge has return_std in predict signature
        import inspect

        if hasattr(model, "predict"):
            sig = inspect.signature(model.predict)
            return "return_std" in sig.parameters
        return False

    @property
    def supports_coefficient_uncertainty(self) -> bool:
        """Check if the fitted model has coefficient uncertainty (coef_ and sigma_)."""
        if self.pipeline is None:
            return False
        model = self.pipeline.named_steps.get("model")
        return hasattr(model, "coef_") and hasattr(model, "sigma_")

    def _get_feature_names_from_preprocessor(self) -> List[str]:
        """Extract feature names from the preprocessor using the same approach as experiments.py."""
        preprocessor = self.pipeline.named_steps.get("preprocessor")
        model = self.pipeline.named_steps.get("model")
        n_features = len(model.coef_)

        if preprocessor is None:
            return [f"feature_{i}" for i in range(n_features)]

        # Iterate through steps in reverse order (same as extract_feature_importance)
        feature_names = None
        steps = list(preprocessor.named_steps.items())
        for name, step in reversed(steps):
            try:
                feature_names = list(step.get_feature_names_out())
                break
            except (AttributeError, TypeError):
                continue

        # Fallback: try getting from entire preprocessor
        if feature_names is None:
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except (AttributeError, TypeError):
                if hasattr(preprocessor, "feature_names_"):
                    feature_names = list(preprocessor.feature_names_)

        # Final fallback: use indices
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        return feature_names

    def get_coefficient_estimates(
        self,
        confidence_levels: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """Extract coefficient estimates with uncertainty from Bayesian models.

        Args:
            confidence_levels: List of confidence levels for intervals (default: [0.80, 0.90, 0.95]).

        Returns:
            DataFrame with columns: feature, coefficient, std, and confidence bounds.

        Raises:
            ValueError: If model doesn't support coefficient uncertainty.
        """
        if not self.supports_coefficient_uncertainty:
            raise ValueError(
                "Model does not support coefficient uncertainty. "
                "Requires coef_ and sigma_ attributes (e.g., BayesianRidge)."
            )

        if confidence_levels is None:
            confidence_levels = [0.80, 0.90, 0.95]

        model = self.pipeline.named_steps.get("model")
        coef = model.coef_
        sigma_diag = np.diag(model.sigma_)
        feature_names = self._get_feature_names_from_preprocessor()

        # Handle ARDRegression which prunes features - sigma_ may be smaller than coef_
        # ARD sets pruned feature coefficients to 0 but sigma_ only contains active features
        if len(sigma_diag) != len(coef):
            # ARDRegression: need to map sigma back to full feature space
            # Pruned features have effectively zero variance (infinite precision)
            coef_std = np.zeros_like(coef)
            if hasattr(model, "lambda_"):
                # Features with lambda < threshold_lambda are active
                active_mask = model.lambda_ < getattr(model, "threshold_lambda", np.inf)
                coef_std[active_mask] = np.sqrt(sigma_diag)
            else:
                # Fallback: assume first len(sigma_diag) features are active
                coef_std[:len(sigma_diag)] = np.sqrt(sigma_diag)
        else:
            coef_std = np.sqrt(sigma_diag)

        # Build DataFrame
        data = {
            "feature": feature_names,
            "coefficient": coef,
            "std": coef_std,
            "abs_coefficient": np.abs(coef),
        }

        # Add confidence intervals
        from scipy import stats

        for level in confidence_levels:
            z = stats.norm.ppf((1 + level) / 2)
            level_pct = int(level * 100)
            data[f"lower_{level_pct}"] = coef - z * coef_std
            data[f"upper_{level_pct}"] = coef + z * coef_std

        # Add significance indicator (95% CI doesn't include zero)
        z_95 = stats.norm.ppf(0.975)
        data["significant_95"] = np.abs(coef) > z_95 * coef_std

        df = pd.DataFrame(data)
        return df.sort_values("abs_coefficient", ascending=False)

    def save_coefficient_estimates(
        self,
        output_path: Union[str, Path],
        confidence_levels: Optional[List[float]] = None,
    ) -> Path:
        """Save coefficient estimates to a CSV file.

        Args:
            output_path: Path to save the estimates (should be .csv).
            confidence_levels: List of confidence levels for intervals.

        Returns:
            Path to saved file.
        """
        df = self.get_coefficient_estimates(confidence_levels)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved coefficient estimates to {output_path}")
        return output_path

    def plot_top_coefficients(
        self,
        output_path: Union[str, Path],
        top_n: int = 100,
        confidence_level: float = 0.95,
        figsize: tuple = (10, 20),
    ) -> Path:
        """Plot top N coefficients by absolute value as dot plot with CI.

        Args:
            output_path: Path to save the plot.
            top_n: Number of top features to plot (default: 100).
            confidence_level: Confidence level for error bars (default: 0.95).
            figsize: Figure size (default: (10, 20)).

        Returns:
            Path to saved plot.
        """
        import matplotlib.pyplot as plt

        df = self.get_coefficient_estimates([confidence_level])
        df_top = df.head(top_n)

        level_pct = int(confidence_level * 100)
        lower_col = f"lower_{level_pct}"
        upper_col = f"upper_{level_pct}"

        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(df_top))
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in df_top["coefficient"]]

        # Plot horizontal error bars (CI) then point estimates
        for i, (_, row) in enumerate(df_top.iterrows()):
            color = "#2ecc71" if row["coefficient"] > 0 else "#e74c3c"
            # CI line
            ax.hlines(
                y=i,
                xmin=row[lower_col],
                xmax=row[upper_col],
                color=color,
                alpha=0.6,
                linewidth=2,
            )
            # Point estimate
            marker = "o" if row["significant_95"] else "o"
            ax.scatter(
                row["coefficient"],
                i,
                color=color,
                s=30,
                zorder=5,
                edgecolors="black" if row["significant_95"] else "none",
                linewidths=1,
            )

        # Vertical line at zero
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_top["feature"], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient Value")
        ax.set_title(
            f"Top {len(df_top)} Features by Absolute Effect\n"
            f"({level_pct}% CI, black edge = significant at 95%)"
        )

        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved coefficient plot to {output_path}")
        return output_path

    def plot_residuals(
        self,
        output_path: Union[str, Path],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: tuple = (12, 10),
    ) -> Path:
        """Plot residuals diagnostic: residuals vs predictions.

        Creates a 2x2 panel with:
        - Residuals vs predicted values (check for heteroscedasticity)
        - Histogram of residuals (check for normality)
        - Q-Q plot of residuals (check for normality)
        - Residuals vs observation index (check for patterns)

        Args:
            output_path: Path to save the plot.
            y_true: True target values.
            y_pred: Predicted values.
            figsize: Figure size (default: (12, 10)).

        Returns:
            Path to saved plot.
        """
        import matplotlib.pyplot as plt
        from scipy import stats

        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Residuals vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.3, s=10)
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Predicted")

        # Add lowess smoother for trend
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(residuals, y_pred, frac=0.2)
            ax1.plot(smoothed[:, 0], smoothed[:, 1], color="orange", linewidth=2)
        except ImportError:
            pass  # statsmodels not available

        # 2. Histogram of residuals
        ax2 = axes[0, 1]
        ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7, density=True)
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal")
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Residual Distribution (μ={mu:.3f}, σ={sigma:.3f})")
        ax2.legend()

        # 3. Q-Q plot
        ax3 = axes[1, 0]
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title("Q-Q Plot (Normal)")

        # 4. Residuals vs observation index
        ax4 = axes[1, 1]
        ax4.scatter(range(len(residuals)), residuals, alpha=0.3, s=10)
        ax4.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax4.set_xlabel("Observation Index")
        ax4.set_ylabel("Residuals")
        ax4.set_title("Residuals vs Index")

        # Add summary statistics as text
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        fig.suptitle(
            f"Residual Diagnostics (RMSE={rmse:.4f}, MAE={mae:.4f})",
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved residual plot to {output_path}")
        return output_path

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
    def configure_model(
        self, algorithm: str, algorithm_params: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """Configure the model and parameter grid for the given algorithm.

        Args:
            algorithm: Algorithm name (e.g., "ridge", "catboost", "lightgbm").
            algorithm_params: Optional algorithm-specific parameters from config.
                For bayesian_ridge, can include: alpha_1, alpha_2, lambda_1, lambda_2.

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
