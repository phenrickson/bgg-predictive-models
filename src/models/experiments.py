"""Experiment tracking and management for model training."""

import json
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# Import visualization modules
from src.visualizations import regression_diagnostics, classification_diagnostics

from sklearn.base import clone
from sklearn.pipeline import Pipeline


def compute_hash(data: Dict[str, Any]) -> str:
    """Compute a hash of the experiment configuration and data.

    Args:
        data: Dictionary containing experiment configuration

    Returns:
        Hash string
    """
    # Convert data to a stable string representation
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()[:8]


class ExperimentTracker:
    """Tracks and manages machine learning experiments."""

    def __init__(
        self, model_type: str, base_dir: Union[str, Path] = "models/experiments"
    ):
        """Initialize experiment tracker.

        Args:
            model_type: Type of model being tracked (e.g., 'hurdle', 'rating')
            base_dir: Base directory for storing experiments
        """
        self.base_dir = Path(base_dir)
        self.model_type = model_type
        self.model_dir = self.base_dir / model_type
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "Experiment":
        """Create a new experiment.

        Args:
            name: Name of the experiment
            description: Optional description
            metadata: Optional metadata dictionary
            version: Optional version number. If None, auto-increments from existing versions.
            config: Optional configuration dictionary for tracking experiment details

        Returns:
            Experiment object
        """
        # Compute hash of experiment configuration
        exp_config = {
            "name": name,
            "description": description,
            "metadata": metadata or {},
            "config": config or {},
            "model_type": self.model_type,
        }
        exp_hash = compute_hash(exp_config)

        # Find existing experiment directories
        experiment_dir = self.model_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Find existing versions and determine next version
        existing_versions = [
            int(p.name[1:])
            for p in experiment_dir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]

        if version is None:
            version = max(existing_versions, default=0) + 1
        elif version in existing_versions:
            raise ValueError(f"Version {version} already exists for experiment {name}")

        # Create version subdirectory
        versioned_dir = experiment_dir / f"v{version}"
        versioned_dir.mkdir(parents=True, exist_ok=True)

        # Add hash to metadata
        if metadata is None:
            metadata = {}
        metadata["hash"] = exp_hash
        metadata["config_hash"] = exp_hash  # For backwards compatibility
        metadata["config"] = config or {}

        return Experiment(name, versioned_dir, description, metadata)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments for this model type with their versions."""
        experiments = []
        for exp_dir in self.model_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # Get all version subdirectories
            versions = [
                (int(v.name[1:]), v)
                for v in exp_dir.iterdir()
                if v.is_dir() and v.name.startswith("v") and v.name[1:].isdigit()
            ]

            # If no versions, skip
            if not versions:
                continue

            # Sort versions
            versions.sort(key=lambda x: x[0])

            for version, version_dir in versions:
                # Load metadata if available
                metadata_file = version_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                experiments.append(
                    {
                        "name": exp_dir.name,
                        "version": version,
                        "full_name": f"{exp_dir.name}/v{version}",
                        "description": metadata.get("description"),
                        "timestamp": metadata.get("timestamp"),
                    }
                )

        return sorted(experiments, key=lambda x: (x["name"], x["version"]))

    def load_experiment(self, name: str, version: Optional[int] = None) -> "Experiment":
        """Load an existing experiment.

        Args:
            name: Name of the experiment to load
            version: Optional specific version to load. If None, loads latest version.

        Returns:
            Experiment object
        """
        # Find the experiment directory
        experiment_dir = self.model_dir / name

        if not experiment_dir.exists():
            raise ValueError(f"No experiment found matching '{name}'")

        # Find version subdirectories
        versions = [
            int(v.name[1:])
            for v in experiment_dir.iterdir()
            if v.is_dir() and v.name.startswith("v") and v.name[1:].isdigit()
        ]

        if not versions:
            raise ValueError(f"No versions found for experiment '{name}'")

        # If version is not specified, find the latest
        if version is None:
            version = max(versions)
        elif version not in versions:
            raise ValueError(f"Version {version} not found for experiment '{name}'")

        # Select the version directory
        version_dir = experiment_dir / f"v{version}"

        # Load metadata
        metadata_file = version_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        # Create and return experiment
        return Experiment(
            name=name,  # Use base experiment name
            base_dir=version_dir,  # Use specific version directory
            description=metadata.get("description"),
            metadata=metadata.get("metadata", {}),
            timestamp=metadata.get("timestamp"),  # Preserve original timestamp
        )


class Experiment:
    """Represents a single experiment run."""

    def __init__(
        self,
        name: str,
        base_dir: Path,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ):
        """Initialize experiment.

        Args:
            name: Name of the experiment
            base_dir: Base directory for all experiments
            description: Optional description
            metadata: Optional metadata dictionary
            timestamp: Optional timestamp (if not provided, current time will be used)
        """
        self.name = name
        self.exp_dir = base_dir
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.description = description
        self.metadata = metadata or {}

        # Use provided timestamp or generate a new one
        self.timestamp = timestamp or datetime.now().isoformat()

        # Save initial metadata
        self._save_metadata()

    def _save_metadata(self):
        """Save experiment metadata to file."""
        metadata = {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

        with open(self.exp_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def log_metrics(self, metrics: Dict[str, Dict[str, float]], dataset: str):
        """Log metrics for a specific dataset.

        Args:
            metrics: Dictionary of metrics
            dataset: Dataset name (e.g., 'train', 'tune', 'test')
        """
        metrics_file = self.exp_dir / f"{dataset}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    def log_parameters(self, params: Dict[str, Any]):
        """Log model parameters.

        Args:
            params: Dictionary of parameters
        """
        params_file = self.exp_dir / "parameters.json"
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)

    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information.

        Args:
            model_info: Dictionary containing model information
        """
        info_file = self.exp_dir / "model_info.json"
        with open(info_file, "w") as f:
            json.dump(model_info, f, indent=2)

    def save_pipeline(self, pipeline: Pipeline) -> None:
        """Save the complete sklearn pipeline.

        Args:
            pipeline: Complete sklearn pipeline including preprocessing and model
        """
        pipeline_path = self.exp_dir / "pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)

    def load_pipeline(self) -> Pipeline:
        """Load the complete sklearn pipeline.

        Returns:
            Complete sklearn pipeline including preprocessing and model
        """
        pipeline_path = self.exp_dir / "pipeline.pkl"
        if not pipeline_path.exists():
            raise ValueError(f"No pipeline found for experiment {self.name}")

        with open(pipeline_path, "rb") as f:
            return pickle.load(f)

    def finalize_model(
        self,
        X: Any,
        y: Any,
        description: str = "Finalized model for production use",
        final_end_year: Optional[int] = None,
        sample_weight: Optional[Any] = None,
    ) -> Path:
        """Create production version by fitting pipeline on full dataset.

        Args:
            X: Features to fit on
            y: Target to fit on
            description: Description of finalized model
            final_end_year: Final year of data used in model training
            sample_weight: Optional sample weights for fitting

        Returns:
            Path to finalized model directory
        """
        import logging

        logger = logging.getLogger(__name__)

        # Load and clone pipeline
        original_pipeline = self.load_pipeline()
        pipeline = clone(original_pipeline)

        # Diagnostic logging for original pipeline
        logger.info("Original Pipeline Steps:")
        for name, step in original_pipeline.named_steps.items():
            logger.info(f"  Step: {name}, Type: {type(step)}")

            # Try to get feature names and scaling details
            try:
                feature_names = step.get_feature_names_out()
                logger.info(f"    Feature Names: {feature_names[:10]}")
                logger.info(f"    Total Feature Names Count: {len(feature_names)}")
            except Exception as e:
                logger.info(f"    Could not extract feature names: {e}")

            # Check for scaling-related attributes
            try:
                if hasattr(step, "scale_"):
                    logger.info(f"    Scale: {step.scale_}")
                if hasattr(step, "mean_"):
                    logger.info(f"    Mean: {step.mean_}")
                if hasattr(step, "var_"):
                    logger.info(f"    Variance: {step.var_}")
            except Exception as e:
                logger.info(f"    Could not extract scaling details: {e}")

            # Additional diagnostic for preprocessor steps
            if hasattr(step, "named_steps"):
                logger.debug("    Preprocessor Sub-Steps:")
                for sub_name, sub_step in step.named_steps.items():
                    logger.debug(f"      Sub-Step: {sub_name}, Type: {type(sub_step)}")
                    try:
                        sub_feature_names = sub_step.get_feature_names_out()
                        logger.debug(
                            f"        Sub-Step Feature Names: {sub_feature_names[:10]}"
                        )
                        logger.debug(
                            f"        Sub-Step Total Feature Names Count: {len(sub_feature_names)}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"        Could not extract sub-step feature names: {e}"
                        )

        # Detailed input data diagnostics
        logger.info("\nInput Data Diagnostics:")
        logger.info(f"  Input Features Shape: {X.shape}")
        logger.info(f"  Target Shape: {y.shape}")
        logger.info(f"  Target Type: {type(y)}")
        logger.info(f"  Target Range: min={y.min()}, max={y.max()}")
        logger.info(f"  Target Mean: {y.mean()}")
        logger.info(f"  Target Std Dev: {y.std()}")

        # Check if sample weights were used during training
        use_sample_weights = self.metadata.get("use_sample_weights", False)

        # Fit on full dataset with optional sample weights
        if sample_weight is not None and use_sample_weights:
            logger.info("\nFitting with Sample Weights:")
            logger.info(f"  Sample Weight Shape: {sample_weight.shape}")
            logger.info(
                f"  Sample Weight Range: min={sample_weight.min()}, max={sample_weight.max()}"
            )
            logger.info(f"  Sample Weight Mean: {sample_weight.mean()}")

            # Fit with sample weights using model__sample_weight
            pipeline.fit(
                X,
                y,
                model__sample_weight=(
                    np.asarray(sample_weight) if sample_weight is not None else None
                ),
            )
        else:
            pipeline.fit(X, y)

        # Diagnostic logging for fitted pipeline
        logger.info("\nFitted Pipeline Steps:")
        for name, step in pipeline.named_steps.items():
            logger.info(f"  Step: {name}, Type: {type(step)}")

            # Try to get feature names and scaling details
            try:
                if hasattr(step, "get_feature_names_out"):
                    feature_names = step.get_feature_names_out()
                    logger.info(f"    Feature Names: {feature_names[:10]}")

                # Check for scaling-related attributes
                if hasattr(step, "scale_"):
                    logger.info(f"    Scale: {step.scale_}")
                if hasattr(step, "mean_"):
                    logger.info(f"    Mean: {step.mean_}")
                if hasattr(step, "var_"):
                    logger.info(f"    Variance: {step.var_}")

                # Additional diagnostic for preprocessor steps
                if hasattr(step, "named_steps"):
                    logger.info("    Preprocessor Sub-Steps:")
                    for sub_name, sub_step in step.named_steps.items():
                        logger.info(
                            f"      Sub-Step: {sub_name}, Type: {type(sub_step)}"
                        )
                        try:
                            if hasattr(sub_step, "get_feature_names_out"):
                                sub_feature_names = sub_step.get_feature_names_out()
                                logger.info(
                                    f"        Sub-Step Feature Names: {sub_feature_names[:10]}"
                                )
                        except Exception as e:
                            logger.info(
                                f"        Could not extract sub-step feature names: {e}"
                            )
            except Exception as e:
                logger.info(f"    Could not extract details: {e}")

        # Save to finalized directory
        finalized_dir = self.exp_dir / "finalized"
        finalized_dir.mkdir(exist_ok=True)

        # Save fitted pipeline
        pipeline_path = finalized_dir / "pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)

        # Save info
        info = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "experiment_name": self.name,
            "experiment_hash": self.metadata.get("hash"),
            "model_type": self.metadata.get("model_type"),
            "target": self.metadata.get("target"),
            "final_end_year": final_end_year,
        }

        info_path = finalized_dir / "info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        return finalized_dir

    def load_finalized_model(self) -> Pipeline:
        """Load the finalized pipeline.

        Returns:
            Complete sklearn pipeline ready for predictions

        Raises:
            ValueError: If no finalized model exists
        """
        finalized_dir = self.exp_dir / "finalized"
        if not finalized_dir.exists():
            raise ValueError(f"No finalized model found for experiment {self.name}")

        pipeline_path = finalized_dir / "pipeline.pkl"
        if not pipeline_path.exists():
            raise ValueError("Pipeline not found in finalized directory")

        with open(pipeline_path, "rb") as f:
            return pickle.load(f)

    def log_coefficients(self, coefficients_df: pl.DataFrame):
        """Log model coefficients or feature importance.

        Args:
            coefficients_df: DataFrame containing coefficient or feature importance information
        """
        # Determine the file type based on column names
        if "coefficient" in coefficients_df.columns:
            coef_file = self.exp_dir / "coefficients.csv"
            coefficients_df.write_csv(coef_file)
        elif "feature_importance" in coefficients_df.columns:
            coef_file = self.exp_dir / "feature_importance.csv"
            coefficients_df.write_csv(coef_file)
        else:
            raise ValueError(
                "DataFrame must contain either 'coefficient' or 'feature_importance' column"
            )

    def log_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        df: pl.DataFrame,
        dataset: str,
        predicted_proba: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ):
        """Log model predictions along with actual values and identifying information.

        Args:
            predictions: Array of model predictions
            actuals: Array of actual values
            df: DataFrame containing identifying information (e.g., game IDs, names)
            dataset: Dataset name (e.g., 'train', 'tune', 'test')
            predicted_proba: Optional array of predicted probabilities for classification models
            threshold: Optional threshold used for generating predictions
        """
        # Create predictions DataFrame
        predictions_df = df.clone()
        predictions_df = predictions_df.with_columns(
            [pl.Series("prediction", predictions), pl.Series("actual", actuals)]
        )

        # Add threshold column if provided
        if threshold is not None:
            predictions_df = predictions_df.with_columns(
                pl.Series("threshold", [threshold] * len(predictions_df))
            )

        # Add predicted probabilities if provided
        if predicted_proba is not None:
            # For binary classification, add probability of positive class
            if predicted_proba.ndim == 1:
                predictions_df = predictions_df.with_columns(
                    pl.Series("predicted_proba", predicted_proba)
                )
            # For multiclass, add columns for each class probability
            elif predicted_proba.ndim == 2:
                for i in range(predicted_proba.shape[1]):
                    predictions_df = predictions_df.with_columns(
                        pl.Series(f"predicted_proba_class_{i}", predicted_proba[:, i])
                    )

        # Save as parquet for efficient storage and reading
        predictions_file = self.exp_dir / f"{dataset}_predictions.parquet"
        predictions_df.write_parquet(predictions_file)

    def log_data_split(
        self,
        df: pl.DataFrame,
        dataset: str,
    ):
        """Save a data split (train/tune/test) as a parquet artifact.

        Args:
            df: DataFrame containing the data split
            dataset: Dataset name (e.g., 'train', 'tune', 'test')
        """
        data_dir = self.exp_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_file = data_dir / f"{dataset}.parquet"
        df.write_parquet(data_file)

    def log_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ):
        """Compute and save metrics across different classification thresholds.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            thresholds: Optional array of thresholds to evaluate (default: 0.01 to 0.99)
        """
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            fbeta_score,
            accuracy_score,
        )

        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)

        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Skip if all predictions are same class
            if len(np.unique(y_pred)) == 1:
                continue

            results.append({
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "predicted_positive_rate": y_pred.mean(),
                "n_predicted_positive": int(y_pred.sum()),
                "n_predicted_negative": int((1 - y_pred).sum()),
            })

        # Save as CSV
        results_df = pl.DataFrame(results)
        results_file = self.exp_dir / "threshold_analysis.csv"
        results_df.write_csv(results_file)

        # Create and save plot
        self._create_threshold_plot(results_df)

        return results_df

    def _create_threshold_plot(self, results_df: pl.DataFrame):
        """Create and save threshold analysis plot.

        Args:
            results_df: DataFrame with threshold analysis results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Convert to pandas for plotting
        df = results_df.to_pandas()

        # Plot 1: Precision, Recall, F1, F2 vs Threshold
        ax1 = axes[0]
        ax1.plot(df["threshold"], df["precision"], label="Precision", linewidth=2)
        ax1.plot(df["threshold"], df["recall"], label="Recall", linewidth=2)
        ax1.plot(df["threshold"], df["f1"], label="F1", linewidth=2)
        ax1.plot(df["threshold"], df["f2"], label="F2", linewidth=2, linestyle="--")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Score")
        ax1.set_title("Classification Metrics vs Threshold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Plot 2: Predicted Positive Rate vs Threshold
        ax2 = axes[1]
        ax2.plot(df["threshold"], df["predicted_positive_rate"],
                 label="Predicted Positive Rate", linewidth=2, color="purple")
        ax2.axhline(y=df["recall"].iloc[0] if len(df) > 0 else 0.5,
                    color="gray", linestyle="--", alpha=0.5, label="Actual Positive Rate")
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Rate")
        ax2.set_title("Predicted Positive Rate vs Threshold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        plot_file = self.exp_dir / "threshold_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def log_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ):
        """Compute and save calibration curve data and plot.

        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            n_bins: Number of bins for calibration curve
        """
        from sklearn.calibration import calibration_curve

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

        # Save calibration data
        calibration_df = pl.DataFrame({
            "mean_predicted_probability": prob_pred,
            "fraction_of_positives": prob_true,
        })
        calibration_file = self.exp_dir / "calibration_curve.csv"
        calibration_df.write_csv(calibration_file)

        # Create and save plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        # Model calibration curve
        ax.plot(prob_pred, prob_true, "s-", label="Model", linewidth=2, markersize=8)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        plot_file = self.exp_dir / "calibration_curve.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return calibration_df

    def log_correlation_matrix(
        self,
        X_processed: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        max_features: int = 50,
    ):
        """Compute and save correlation matrix of processed features.

        Args:
            X_processed: DataFrame of processed/transformed features
            feature_names: Optional list of feature names (uses column names if not provided)
            max_features: Maximum number of features to include in plot (for readability)
        """
        import seaborn as sns

        # Use provided feature names or column names
        if feature_names is None:
            feature_names = list(X_processed.columns)

        # Compute correlation matrix
        corr_matrix = X_processed.corr()

        # Save full correlation matrix as CSV
        corr_file = self.exp_dir / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_file)

        # For the plot, limit to top features by variance if too many
        if len(feature_names) > max_features:
            # Select features with highest variance
            variances = X_processed.var().sort_values(ascending=False)
            top_features = variances.head(max_features).index.tolist()
            plot_corr = corr_matrix.loc[top_features, top_features]
            title_suffix = f" (top {max_features} by variance)"
        else:
            plot_corr = corr_matrix
            title_suffix = ""

        # Create correlation heatmap
        n_features = len(plot_corr)
        fig_size = max(10, min(20, n_features * 0.3))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Use diverging colormap centered at 0
        sns.heatmap(
            plot_corr,
            annot=n_features <= 20,  # Only show annotations for smaller matrices
            fmt=".2f" if n_features <= 20 else "",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(f"Feature Correlation Matrix{title_suffix}")

        # Rotate labels for readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        plot_file = self.exp_dir / "correlation_matrix.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Also save summary statistics about correlations
        # Find highly correlated feature pairs (excluding diagonal)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    high_corr_pairs.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": corr_val,
                    })

        if high_corr_pairs:
            high_corr_df = pl.DataFrame(high_corr_pairs).sort("correlation", descending=True)
            high_corr_file = self.exp_dir / "high_correlations.csv"
            high_corr_df.write_csv(high_corr_file)

        return corr_matrix

    def get_predictions(self, dataset: str) -> pl.DataFrame:
        """Get model predictions, actual values, and identifying information.

        Args:
            dataset: Dataset name (e.g., 'train', 'tune', 'test')

        Returns:
            DataFrame containing predictions, actual values, and identifying information

        Raises:
            ValueError: If no predictions found for the dataset
        """
        predictions_file = self.exp_dir / f"{dataset}_predictions.parquet"
        if not predictions_file.exists():
            raise ValueError(f"No predictions found for dataset '{dataset}'")

        return pl.read_parquet(predictions_file)

    def get_metrics(self, dataset: str) -> Dict[str, float]:
        """Get metrics for a specific dataset.

        Args:
            dataset: Dataset name (e.g., 'train', 'tune', 'test')

        Returns:
            Dictionary of metrics
        """
        metrics_file = self.exp_dir / f"{dataset}_metrics.json"
        if not metrics_file.exists():
            raise ValueError(f"No metrics found for dataset '{dataset}'")

        with open(metrics_file) as f:
            return json.load(f)

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of parameters
        """
        params_file = self.exp_dir / "parameters.json"
        if not params_file.exists():
            raise ValueError("No parameters found")

        with open(params_file) as f:
            return json.load(f)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary containing model information
        """
        info_file = self.exp_dir / "model_info.json"
        if not info_file.exists():
            raise ValueError("No model info found")

        with open(info_file) as f:
            return json.load(f)

    def get_coefficients(self) -> pl.DataFrame:
        """Get model coefficients or feature importance.

        Returns:
            DataFrame containing coefficient or feature importance information
        """
        coef_file = self.exp_dir / "coefficients.csv"
        importance_file = self.exp_dir / "feature_importance.csv"

        if coef_file.exists():
            return pl.read_csv(coef_file)
        elif importance_file.exists():
            return pl.read_csv(importance_file)
        else:
            raise ValueError("No coefficients or feature importance found")


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        MAPE value
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def extract_feature_importance(
    fitted_pipeline: Pipeline, model_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract feature importance or coefficients from a fitted pipeline.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted scikit-learn pipeline containing a preprocessor and model
    model_type : Optional[str]
        Type of model to handle specific feature importance extraction

    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names, importance values, and absolute importance,
        sorted by absolute importance value in descending order.
    """
    # Get the preprocessor and model from pipeline
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]

    # Find feature names
    steps = list(preprocessor.named_steps.items())
    feature_names = None

    # Iterate through steps in reverse order
    for name, step in reversed(steps):
        try:
            # Try to get feature names from the step
            feature_names = step.get_feature_names_out()
            break
        except (AttributeError, TypeError):
            continue

    # If no step with feature names found, try fallback methods
    if feature_names is None:
        try:
            # Try getting from the entire preprocessor
            feature_names = preprocessor.get_feature_names_out()
        except (AttributeError, TypeError) as e:
            # Last resort: check for feature_names_ attribute
            if hasattr(preprocessor, "feature_names_"):
                feature_names = preprocessor.feature_names_
            else:
                raise ValueError(
                    f"Could not get feature names from any preprocessing step: {e}"
                )

    # Handle different model types for feature importance extraction
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso

    # Determine feature importance extraction method based on model type
    if isinstance(model, (LogisticRegression, LinearRegression, Ridge, Lasso)):
        # For linear models, use coef_ attribute
        if not hasattr(model, "coef_"):
            raise ValueError("Model does not have coefficients")

        # Handle binary classification (logistic regression)
        importance_values = (
            model.coef_[0] if isinstance(model, LogisticRegression) else model.coef_
        )
        importance_type = "coefficient"
    else:
        # For non-linear models, use feature importances if available
        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
            importance_type = "feature_importance"
        else:
            raise ValueError(
                f"Cannot extract feature importance for model type: {type(model)}"
            )

    # Validate lengths match
    if len(feature_names) != len(importance_values):
        raise ValueError(
            f"Mismatch between number of features ({len(feature_names)}) "
            f"and {importance_type} values ({len(importance_values)})"
        )

    # Create DataFrame with importance values
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            f"{importance_type}": importance_values,
            f"abs_{importance_type}": np.abs(importance_values),
        }
    )

    # Sort by absolute importance value
    importance_df = importance_df.sort_values(f"abs_{importance_type}", ascending=False)

    # Add rank
    importance_df["rank"] = range(1, len(importance_df) + 1)

    return importance_df


# Learning curve function removed as per user request
def create_diagnostic_plots(
    experiment: "Experiment",
    predictions_df: pl.DataFrame,
    model_type: str = "regression",
) -> None:
    """
    Create and save diagnostic plots for model predictions.

    Args:
        experiment: Experiment tracking object
        predictions_df: DataFrame containing predictions and actual values
        model_type: Type of model ('regression' or 'classification')
    """
    logger = logging.getLogger(__name__)

    try:
        # Create plots directory if it doesn't exist
        plots_dir = experiment.exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Create plots based on model type
        if model_type == "regression":
            # Use regression_diagnostics module
            fig, _ = regression_diagnostics.plot_regression_diagnostics(predictions_df)
            plot_path = plots_dir / "regression_diagnostics.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved regression diagnostic plots to {plot_path}")

        elif model_type == "classification":
            # Use classification_diagnostics module
            # First, check if the required columns exist
            required_columns = ["actual", "prediction", "predicted_proba_class_1"]
            if all(col in predictions_df.columns for col in required_columns):
                fig, _ = classification_diagnostics.plot_classification_diagnostics(
                    predictions_df
                )
                plot_path = plots_dir / "classification_diagnostics.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Saved classification diagnostic plots to {plot_path}")
            else:
                logger.warning(
                    "Missing required columns for classification diagnostic plots"
                )

        else:
            logger.warning(f"Unsupported model type for diagnostic plots: {model_type}")

    except Exception as e:
        logger.error(f"Error creating diagnostic plots: {e}")


def log_experiment(
    experiment: "Experiment",
    pipeline: Pipeline,
    train_metrics: Dict[str, float],
    tune_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_params: Dict[str, Any],
    args: Any,
    train_df: Optional[pl.DataFrame] = None,
    tune_df: Optional[pl.DataFrame] = None,
    test_df: Optional[pl.DataFrame] = None,
    train_X: Optional[pd.DataFrame] = None,
    tune_X: Optional[pd.DataFrame] = None,
    test_X: Optional[pd.DataFrame] = None,
    train_y: Optional[pd.Series] = None,
    tune_y: Optional[pd.Series] = None,
    test_y: Optional[pd.Series] = None,
    model_type: str = "regression",
    stratified_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """
    Log all experiment results and artifacts with flexibility for different model types.

    Args:
        experiment: Experiment tracking object
        pipeline: Fitted model pipeline
        train_metrics: Metrics for training set
        tune_metrics: Metrics for tuning set
        test_metrics: Metrics for test set
        best_params: Best hyperparameters
        args: Argument namespace
        train_df: Optional training dataframe
        tune_df: Optional tuning dataframe
        test_df: Optional test dataframe
        train_X: Optional training features
        tune_X: Optional tuning features
        test_X: Optional test features
        train_y: Optional training target
        tune_y: Optional tuning target
        test_y: Optional test target
        model_type: Type of model ('regression' or 'classification')
    """
    logger = logging.getLogger(__name__)

    # For rating models, add complexity experiment to metadata
    if model_type in ["rating", "regression"]:
        # Check if complexity experiment is provided in args
        complexity_experiment = getattr(args, "complexity_experiment", None)

        if complexity_experiment:
            # Add complexity experiment to experiment metadata
            experiment.metadata["complexity_experiment"] = complexity_experiment
            experiment._save_metadata()
            logger.info(
                f"Added complexity experiment '{complexity_experiment}' to metadata"
            )

    # Determine if the model is a classifier
    is_classifier = model_type == "classification"

    # Filter metrics based on model type
    def filter_metrics(metrics):
        if is_classifier:
            # Keep only classification metrics
            return {
                k: v
                for k, v in metrics.items()
                if k
                in [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "auc",
                    "log_loss",
                    "matthews_corr",
                    "confusion_matrix",
                    "true_negatives",
                    "false_positives",
                    "false_negatives",
                    "true_positives",
                ]
            }
        else:
            # Keep only regression metrics
            return {
                k: v
                for k, v in metrics.items()
                if k in ["mse", "rmse", "mae", "r2", "mape"]
            }

    # Log filtered metrics
    experiment.log_metrics(filter_metrics(train_metrics), "train")
    experiment.log_metrics(filter_metrics(tune_metrics), "tune")
    experiment.log_metrics(filter_metrics(test_metrics), "test")
    experiment.log_parameters(best_params)

    # For classification models, provide a more descriptive confusion matrix log
    if model_type == "classification":
        for dataset, metrics in [
            ("train", train_metrics),
            ("tune", tune_metrics),
            ("test", test_metrics),
        ]:
            if "confusion_matrix" in metrics:
                cm = metrics["confusion_matrix"]

                # Handle dict format from HurdleModel.compute_additional_metrics
                if isinstance(cm, dict):
                    true_negatives = cm.get("true_negatives", 0)
                    false_positives = cm.get("false_positives", 0)
                    false_negatives = cm.get("false_negatives", 0)
                    true_positives = cm.get("true_positives", 0)
                else:
                    # Ensure the confusion matrix is a 2x2 matrix
                    if isinstance(cm, list):
                        cm = np.array(cm)

                    # Validate confusion matrix shape
                    if cm.shape != (2, 2):
                        logger.warning(
                            f"Unexpected confusion matrix shape for {dataset} set: {cm.shape}"
                        )
                        continue

                    # Extract key metrics from confusion matrix
                    true_negatives = int(cm[0, 0])
                    false_positives = int(cm[0, 1])
                    false_negatives = int(cm[1, 0])
                    true_positives = int(cm[1, 1])

                # Update metrics to store only key metrics
                metrics.update(
                    {
                        "true_negatives": true_negatives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                        "true_positives": true_positives,
                    }
                )
                del metrics["confusion_matrix"]

                logger.info(f"Confusion Matrix Metrics ({dataset.upper()} set):")
                logger.info(
                    f"  True Negatives:     {true_negatives} (Correctly predicted non-events)"
                )
                logger.info(
                    f"  False Positives:    {false_positives} (Incorrectly predicted as events)"
                )
                logger.info(f"  False Negatives:    {false_negatives} (Missed events)")
                logger.info(
                    f"  True Positives:     {true_positives} (Correctly predicted events)"
                )

                # Calculate and log additional insights
                total = (
                    true_negatives + false_positives + false_negatives + true_positives
                )
                logger.info("  Prediction Breakdown:")
                logger.info(
                    f"    Negative Predictions: {true_negatives + false_positives} ({(true_negatives + false_positives) / total * 100:.2f}%)"
                )
                logger.info(
                    f"    Positive Predictions: {false_negatives + true_positives} ({(false_negatives + true_positives) / total * 100:.2f}%)"
                )
                logger.info(
                    f"    Accuracy: {(true_negatives + true_positives) / total * 100:.2f}%"
                )

    # Save train/tune/test data splits as artifacts
    try:
        if train_df is not None:
            experiment.log_data_split(train_df, "train")
            logger.info(f"Saved training data: {len(train_df)} rows")
        if tune_df is not None:
            experiment.log_data_split(tune_df, "tune")
            logger.info(f"Saved tuning data: {len(tune_df)} rows")
        if test_df is not None:
            experiment.log_data_split(test_df, "test")
            logger.info(f"Saved test data: {len(test_df)} rows")
    except Exception as e:
        logger.warning(f"Could not save data splits: {e}")

    # For classification models, compute threshold analysis and calibration
    if is_classifier:
        try:
            # Use test set for threshold analysis, fallback to tune set
            if test_X is not None and test_y is not None:
                test_proba = pipeline.predict_proba(test_X)[:, 1]
                experiment.log_threshold_analysis(test_y.values, test_proba)
                experiment.log_calibration_curve(test_y.values, test_proba)
                logger.info("Saved threshold analysis and calibration curve (test set)")
            elif tune_X is not None and tune_y is not None:
                tune_proba = pipeline.predict_proba(tune_X)[:, 1]
                experiment.log_threshold_analysis(tune_y.values, tune_proba)
                experiment.log_calibration_curve(tune_y.values, tune_proba)
                logger.info("Saved threshold analysis and calibration curve (tune set)")
        except Exception as e:
            logger.warning(f"Could not compute threshold analysis: {e}")

    # Extract and save feature importance
    try:
        # Extract feature importance
        importance_df = extract_feature_importance(pipeline, model_type=model_type)
        importance_pl = pl.from_pandas(importance_df)
        experiment.log_coefficients(importance_pl)

        # Log top features
        importance_column = (
            "coefficient"
            if "coefficient" in importance_df.columns
            else "feature_importance"
        )

        logger.info(
            f"Top 10 most important features (by absolute {importance_column}):"
        )
        for _, row in importance_df.head(10).iterrows():
            logger.info(
                f"  {row['rank']:2d}. {row['feature']:30s} = {row[importance_column]:8.4f}"
            )

        # Save model info
        model_info = {"n_features": len(importance_df), "best_params": best_params}

        # Add model-specific details
        if hasattr(pipeline.named_steps["model"], "intercept_"):
            model_info["intercept"] = float(pipeline.named_steps["model"].intercept_)

        # Extract preprocessor parameters
        try:
            preprocessor = pipeline.named_steps["preprocessor"]

            # Look for the BGG preprocessor step
            bgg_preprocessor = None
            if hasattr(preprocessor, "named_steps"):
                for step_name, step in preprocessor.named_steps.items():
                    if (
                        step_name == "bgg_preprocessor"
                        or "bgg_preprocessor" in step_name
                    ):
                        bgg_preprocessor = step
                        break

            # If BGG preprocessor found, extract its parameters
            if bgg_preprocessor is not None:
                # Try to get the original configuration parameters
                if hasattr(bgg_preprocessor, "config"):
                    model_info["preprocessor_params"] = bgg_preprocessor.config
                elif hasattr(bgg_preprocessor, "_config"):
                    model_info["preprocessor_params"] = bgg_preprocessor._config
                elif hasattr(bgg_preprocessor, "get_params"):
                    # Fallback to get_params method
                    model_info["preprocessor_params"] = {
                        k: v
                        for k, v in bgg_preprocessor.get_params().items()
                        if not k.startswith("__") and not callable(v)
                    }
        except Exception as e:
            logger.warning(f"Could not extract preprocessor parameters: {e}")

        experiment.log_model_info(model_info)

    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        logger.error("Continuing without saving feature importance")

    # Get and log predictions for validation and test sets
    # Retrieve optimal threshold from experiment metadata
    optimal_threshold = experiment.metadata.get("optimal_threshold", 0.5)

    if tune_X is not None and tune_y is not None and tune_df is not None:
        # For classification models, add predicted probabilities
        if is_classifier:
            try:
                tune_predicted_proba = pipeline.predict_proba(tune_X)
                # Use optimal threshold to generate predictions
                tune_predictions = (
                    tune_predicted_proba[:, 1] >= optimal_threshold
                ).astype(int)

                experiment.log_predictions(
                    predictions=tune_predictions,
                    actuals=tune_y.values,
                    df=tune_df,
                    dataset="tune",
                    predicted_proba=tune_predicted_proba,
                )
            except AttributeError:
                # Fallback if predict_proba is not available
                tune_predictions = pipeline.predict(tune_X)
                experiment.log_predictions(
                    predictions=tune_predictions,
                    actuals=tune_y.values,
                    df=tune_df,
                    dataset="tune",
                )
        else:
            tune_predictions = pipeline.predict(tune_X)
            experiment.log_predictions(
                predictions=tune_predictions,
                actuals=tune_y.values,
                df=tune_df,
                dataset="tune",
            )

    if test_X is not None and test_y is not None and test_df is not None:
        # For classification models, add predicted probabilities
        if is_classifier:
            try:
                test_predicted_proba = pipeline.predict_proba(test_X)
                # Use optimal threshold to generate predictions
                test_predictions = (
                    test_predicted_proba[:, 1] >= optimal_threshold
                ).astype(int)

                experiment.log_predictions(
                    predictions=test_predictions,
                    actuals=test_y.values,
                    df=test_df,
                    dataset="test",
                    predicted_proba=test_predicted_proba,
                )
            except AttributeError:
                # Fallback if predict_proba is not available
                test_predictions = pipeline.predict(test_X)
                experiment.log_predictions(
                    predictions=test_predictions,
                    actuals=test_y.values,
                    df=test_df,
                    dataset="test",
                )
        else:
            test_predictions = pipeline.predict(test_X)
            experiment.log_predictions(
                predictions=test_predictions,
                actuals=test_y.values,
                df=test_df,
                dataset="test",
            )

    # Save pipeline
    experiment.save_pipeline(pipeline)

    # Create diagnostic plots
    try:
        # Determine which dataset to use for plots (prefer test, then tune)
        if test_df is not None and test_X is not None and test_y is not None:
            plot_df = test_df.clone()
            plot_df = plot_df.with_columns(
                [
                    pl.Series("prediction", test_predictions),
                    pl.Series("actual", test_y.values),
                ]
            )

            # Add predicted probabilities for classification models
            if is_classifier and hasattr(pipeline, "predict_proba"):
                test_predicted_proba = pipeline.predict_proba(test_X)
                if test_predicted_proba.ndim == 2:
                    plot_df = plot_df.with_columns(
                        pl.Series("predicted_proba_class_1", test_predicted_proba[:, 1])
                    )

            create_diagnostic_plots(experiment, plot_df, model_type)

        elif tune_df is not None and tune_X is not None and tune_y is not None:
            plot_df = tune_df.clone()
            plot_df = plot_df.with_columns(
                [
                    pl.Series("prediction", tune_predictions),
                    pl.Series("actual", tune_y.values),
                ]
            )

            # Add predicted probabilities for classification models
            if is_classifier and hasattr(pipeline, "predict_proba"):
                tune_predicted_proba = pipeline.predict_proba(tune_X)
                if tune_predicted_proba.ndim == 2:
                    plot_df = plot_df.with_columns(
                        pl.Series("predicted_proba_class_1", tune_predicted_proba[:, 1])
                    )

            create_diagnostic_plots(experiment, plot_df, model_type)
    except Exception as e:
        logger.error(f"Error creating diagnostic plots: {e}")

    # Log additional metadata about the experiment
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_sizes": {
            "train": len(train_y) if train_y is not None else None,
            "tune": len(tune_y) if tune_y is not None else None,
            "test": len(test_y) if test_y is not None else None,
        },
        "model_type": model_type,
        "pipeline_steps": list(pipeline.named_steps.keys()),
        "feature_count": train_X.shape[1] if train_X is not None else None,
    }

    # Add stratified metrics to metadata if provided
    if stratified_metrics is not None:
        metadata["stratified_metrics"] = stratified_metrics

    # Add to existing metadata
    experiment.metadata.update(metadata)
    experiment._save_metadata()

    # Log stratified metrics if available
    if stratified_metrics is not None:
        logger.info("Stratified Metrics:")
        for bucket, metrics in stratified_metrics.items():
            logger.info(f"  {bucket}:")
            for metric, value in metrics.items():
                # Round numeric metrics to 3 decimal places
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.3f}")
                else:
                    # For non-numeric metrics, log as-is
                    logger.info(f"    {metric}: {value}")

    logger.info("Experiment logging complete")
