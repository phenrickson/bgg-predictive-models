"""Embedding generation for new games."""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import polars as pl

from src.models.experiments import ExperimentTracker
from src.models.training import load_data
from src.utils.config import Config, load_config

from .algorithms import BaseEmbeddingAlgorithm

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings for games using a trained model."""

    def __init__(
        self,
        experiment_name: str,
        version: Optional[int] = None,
        config: Optional[Config] = None,
        experiments_dir: str = "./models/experiments",
    ):
        """Initialize embedding generator.

        Args:
            experiment_name: Name of the experiment to load.
            version: Specific version to load. If None, loads latest.
            config: Configuration object.
            experiments_dir: Directory containing experiments.
        """
        self.config = config or load_config()
        self.tracker = ExperimentTracker("embeddings", experiments_dir)
        self.experiment = self.tracker.load_experiment(experiment_name, version)

        # Load the embedding pipeline
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the embedding pipeline from experiment."""
        pipeline_path = self.experiment.exp_dir / "embedding_pipeline.pkl"

        if not pipeline_path.exists():
            raise ValueError(
                f"No embedding pipeline found at {pipeline_path}. "
                "Make sure the experiment was trained with the embeddings module."
            )

        with open(pipeline_path, "rb") as f:
            pipeline_data = pickle.load(f)

        self.preprocessor = pipeline_data["preprocessor"]
        self.embedding_model: BaseEmbeddingAlgorithm = pipeline_data["embedding_model"]
        self.algorithm = pipeline_data["algorithm"]
        self.embedding_dim = pipeline_data["embedding_dim"]

        logger.info(
            f"Loaded {self.algorithm} embedding model with {self.embedding_dim} dimensions"
        )

    def generate_embeddings(
        self,
        df: Optional[pl.DataFrame] = None,
        game_ids: Optional[List[int]] = None,
        complexity_predictions_path: Optional[str] = None,
        local_data_path: Optional[str] = None,
    ) -> pl.DataFrame:
        """Generate embeddings for games.

        Args:
            df: Optional DataFrame with game features. If None, loads from source.
            game_ids: Specific game IDs to generate embeddings for.
            complexity_predictions_path: Path to complexity predictions.
            local_data_path: Path to local data file.

        Returns:
            DataFrame with columns: game_id, name, embedding (as list).
        """
        if df is None:
            df = self._load_games_data(
                game_ids=game_ids,
                complexity_predictions_path=complexity_predictions_path,
                local_data_path=local_data_path,
            )

        # Filter by game_ids if specified
        if game_ids and "game_id" in df.columns:
            df = df.filter(pl.col("game_id").is_in(game_ids))

        logger.info(f"Generating embeddings for {len(df)} games")

        # Preprocess
        df_pandas = df.to_pandas()
        X = self.preprocessor.transform(df_pandas)

        # Generate embeddings
        embeddings = self.embedding_model.transform(X)

        # Create output DataFrame
        result = df.select(["game_id", "name"]).with_columns(
            pl.Series("embedding", [emb.tolist() for emb in embeddings])
        )

        logger.info(f"Generated {len(result)} embeddings")
        return result

    def _load_games_data(
        self,
        game_ids: Optional[List[int]] = None,
        complexity_predictions_path: Optional[str] = None,
        local_data_path: Optional[str] = None,
    ) -> pl.DataFrame:
        """Load games data for embedding generation.

        Args:
            game_ids: Specific game IDs to load.
            complexity_predictions_path: Path to complexity predictions.
            local_data_path: Path to local data file.

        Returns:
            DataFrame with game features and complexity predictions.
        """
        years = self.config.years

        # Load base features
        df = load_data(
            local_data_path=local_data_path,
            end_train_year=years.score.end,
        )

        # Load complexity predictions
        if complexity_predictions_path:
            complexity_df = pl.read_parquet(complexity_predictions_path)
        else:
            default_path = self.config.models.get("complexity")
            if default_path and default_path.predictions_path:
                complexity_df = pl.read_parquet(default_path.predictions_path)
            else:
                raise ValueError(
                    "complexity_predictions_path must be provided or "
                    "configured in config.yaml"
                )

        # Join
        df = df.join(
            complexity_df.select(["game_id", "predicted_complexity"]),
            on="game_id",
            how="inner",
        )

        return df

    def get_embedding(self, game_id: int) -> Optional[np.ndarray]:
        """Get embedding for a single game.

        Args:
            game_id: Game ID to get embedding for.

        Returns:
            Embedding array or None if game not found.
        """
        result = self.generate_embeddings(game_ids=[game_id])

        if len(result) == 0:
            return None

        return np.array(result["embedding"][0])

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "experiment_name": self.experiment.name,
            "algorithm": self.algorithm,
            "embedding_dim": self.embedding_dim,
            "experiment_dir": str(self.experiment.exp_dir),
        }
