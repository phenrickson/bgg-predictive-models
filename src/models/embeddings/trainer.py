"""Embedding trainer for orchestrating the embedding training pipeline."""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.features.preprocessor import create_bgg_preprocessor
from src.models.experiments import ExperimentTracker
from src.models.training import create_data_splits
from src.utils.config import Config, load_config

from .algorithms import BaseEmbeddingAlgorithm, create_embedding_algorithm
from .data import EmbeddingDataLoader

logger = logging.getLogger(__name__)


# Feature columns to use for embeddings (subset of games_features)
EMBEDDING_FEATURE_COLUMNS = [
    # Base numeric features
    "min_age",
    "min_playtime",
    "max_playtime",
    "time_per_player",
    "description_word_count",
    # Player count dummies
    "player_count_1",
    "player_count_2",
    "player_count_3",
    "player_count_4",
    "player_count_5",
    "player_count_6",
    "player_count_7",
    "player_count_8",
    "player_count_9",
    "player_count_10",
]


class EmbeddingTrainer:
    """Orchestrates the embedding training pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        output_dir: str = "./models/experiments",
    ):
        """Initialize the embedding trainer.

        Args:
            config: Configuration object. If None, loads from config.yaml.
            output_dir: Directory for storing experiment artifacts.
        """
        self.config = config or load_config()
        self.output_dir = Path(output_dir)
        self.tracker = ExperimentTracker("embeddings", str(self.output_dir))

    def load_embedding_data(
        self,
        end_year: Optional[int] = None,
        min_ratings: int = 0,
    ) -> pl.DataFrame:
        """Load game features joined with complexity predictions from BigQuery.

        Args:
            end_year: End year for data filtering. If None, uses config.
            min_ratings: Minimum ratings filter.

        Returns:
            DataFrame with features and predicted_complexity.
        """
        loader = EmbeddingDataLoader(config=self.config)
        return loader.load_embedding_data(
            end_year=end_year,
            min_ratings=min_ratings,
        )

    def prepare_features(
        self,
        df: pl.DataFrame,
        preprocessor: Optional[Pipeline] = None,
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Pipeline]:
        """Prepare features for embedding training.

        Uses a subset of features excluding designers, publishers, and artists
        to focus on game characteristics rather than creator identity.

        Args:
            df: Input DataFrame with game features.
            preprocessor: Optional existing preprocessor pipeline.
            fit: Whether to fit the preprocessor.

        Returns:
            Tuple of (preprocessed features DataFrame, fitted preprocessor).
        """
        if preprocessor is None:
            # Create preprocessor with subset of features for embeddings
            # Exclude designers, publishers, and artists - focus on game mechanics/categories
            preprocessor = create_bgg_preprocessor(
                model_type="linear",
                create_designer_features=False,
                create_artist_features=False,
                create_publisher_features=False,
            )

        df_pandas = df.to_pandas()

        if fit:
            X = preprocessor.fit_transform(df_pandas)
        else:
            X = preprocessor.transform(df_pandas)

        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)

        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")

        return X, preprocessor

    def evaluate_embeddings(
        self,
        embeddings: np.ndarray,
        X_original: pd.DataFrame,
        algorithm: BaseEmbeddingAlgorithm,
    ) -> Dict[str, Any]:
        """Evaluate embedding quality.

        Args:
            embeddings: Generated embeddings array.
            X_original: Original feature DataFrame.
            algorithm: Fitted embedding algorithm.

        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}

        # Get algorithm-specific metrics
        algo_metrics = algorithm.get_metrics()
        metrics.update(algo_metrics)

        # Compute reconstruction error for linear methods
        if hasattr(algorithm, "model") and hasattr(algorithm.model, "inverse_transform"):
            try:
                X_scaled = algorithm.scaler.transform(X_original)
                X_reconstructed = algorithm.model.inverse_transform(embeddings)
                reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
                metrics["reconstruction_mse"] = float(reconstruction_error)
            except Exception as e:
                logger.warning(f"Could not compute reconstruction error: {e}")

        # Embedding statistics
        metrics["embedding_mean"] = float(np.mean(embeddings))
        metrics["embedding_std"] = float(np.std(embeddings))
        metrics["embedding_min"] = float(np.min(embeddings))
        metrics["embedding_max"] = float(np.max(embeddings))

        return metrics

    def train(
        self,
        algorithm: str,
        embedding_dim: int,
        experiment_name: str,
        algorithm_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Tuple[BaseEmbeddingAlgorithm, Pipeline, Dict[str, Any]]:
        """Train an embedding model.

        Args:
            algorithm: Algorithm name ('pca', 'svd', 'umap', 'autoencoder').
            embedding_dim: Target embedding dimension.
            experiment_name: Name for the experiment.
            algorithm_params: Algorithm-specific parameters.
            description: Experiment description.

        Returns:
            Tuple of (fitted algorithm, preprocessor, metrics dict).
        """
        years = self.config.years
        algorithm_params = algorithm_params or {}

        # Load data from BigQuery (games_features + complexity_predictions)
        logger.info("Loading embedding data from BigQuery...")
        df = self.load_embedding_data(end_year=years.test_end)

        # Create time-based splits
        logger.info("Creating data splits...")
        train_df, tune_df, test_df = create_data_splits(
            df,
            train_end_year=years.train_end,
            tune_start_year=years.train_end,
            tune_end_year=years.tune_end,
            test_start_year=years.test_start,
            test_end_year=years.test_end,
        )

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Tune: {len(tune_df)}, Test: {len(test_df)}"
        )

        # Prepare features
        logger.info("Preparing features...")
        train_X, preprocessor = self.prepare_features(train_df, fit=True)
        tune_X, _ = self.prepare_features(tune_df, preprocessor=preprocessor, fit=False)
        test_X, _ = self.prepare_features(test_df, preprocessor=preprocessor, fit=False)

        # Create and train embedding algorithm
        logger.info(f"Training {algorithm} with {embedding_dim} dimensions...")
        embedding_model = create_embedding_algorithm(
            algorithm=algorithm,
            embedding_dim=embedding_dim,
            **algorithm_params,
        )
        embedding_model.fit(train_X)

        # Generate embeddings for evaluation
        train_embeddings = embedding_model.transform(train_X)
        tune_embeddings = embedding_model.transform(tune_X)
        test_embeddings = embedding_model.transform(test_X)

        # Evaluate
        logger.info("Evaluating embeddings...")
        train_metrics = self.evaluate_embeddings(train_embeddings, train_X, embedding_model)
        tune_metrics = self.evaluate_embeddings(tune_embeddings, tune_X, embedding_model)
        test_metrics = self.evaluate_embeddings(test_embeddings, test_X, embedding_model)

        # Create experiment
        experiment = self.tracker.create_experiment(
            name=experiment_name,
            description=description or f"{algorithm.upper()} embeddings with {embedding_dim} dimensions",
            metadata={
                "algorithm": algorithm,
                "embedding_dim": embedding_dim,
                "train_end_year": years.train_end,
                "tune_end_year": years.tune_end,
                "test_start_year": years.test_start,
                "test_end_year": years.test_end,
                "model_type": "embeddings",
                "target": "game_embedding",
            },
            config={
                "algorithm_params": algorithm_params,
                "data_splits": {
                    "train_end_year": years.train_end,
                    "tune_start_year": years.train_end,
                    "tune_end_year": years.tune_end,
                    "test_start_year": years.test_start,
                    "test_end_year": years.test_end,
                },
                "feature_config": {
                    "create_designer_features": False,
                    "create_artist_features": False,
                    "create_publisher_features": False,
                    "create_category_features": True,
                    "create_mechanic_features": True,
                    "create_family_features": True,
                    "create_player_dummies": True,
                    "include_base_numeric": True,
                    "includes_predicted_complexity": True,
                },
            },
        )

        # Log metrics
        experiment.log_metrics(train_metrics, "train")
        experiment.log_metrics(tune_metrics, "tune")
        experiment.log_metrics(test_metrics, "test")

        # Log parameters
        experiment.log_parameters({
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
            **algorithm_params,
        })

        # Save the complete pipeline (preprocessor + embedding model)
        # We'll pickle both together for easy loading
        pipeline_data = {
            "preprocessor": preprocessor,
            "embedding_model": embedding_model,
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
        }
        pipeline_path = experiment.exp_dir / "embedding_pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline_data, f)

        # Also save standard pipeline.pkl for compatibility
        experiment.save_pipeline(preprocessor)

        # Log model info
        experiment.log_model_info({
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
            "n_features_in": train_X.shape[1],
            "n_samples_trained": len(train_X),
            **embedding_model.get_metrics(),
        })

        # Save embeddings for train/tune/test
        for name, emb_df, embeddings in [
            ("train", train_df, train_embeddings),
            ("tune", tune_df, tune_embeddings),
            ("test", test_df, test_embeddings),
        ]:
            emb_output = emb_df.select(["game_id", "name"]).with_columns(
                pl.Series("embedding", [e.tolist() for e in embeddings])
            )
            emb_path = experiment.exp_dir / f"{name}_embeddings.parquet"
            emb_output.write_parquet(emb_path)

        logger.info(f"Experiment saved to {experiment.exp_dir}")

        all_metrics = {
            "train": train_metrics,
            "tune": tune_metrics,
            "test": test_metrics,
        }

        return embedding_model, preprocessor, all_metrics
