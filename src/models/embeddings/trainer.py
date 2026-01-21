"""Embedding trainer for orchestrating the embedding training pipeline."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.models.experiments import ExperimentTracker
from src.models.training import create_data_splits
from src.utils.config import Config, load_config

from .algorithms import BaseEmbeddingAlgorithm, create_embedding_algorithm
from .data import EmbeddingDataLoader
from .transformer import create_embedding_preprocessor

logger = logging.getLogger(__name__)


# Feature columns to use for embeddings (subset of games_features)
EMBEDDING_FEATURE_COLUMNS = [
    # Base numeric features
    "min_age",
    "min_playtime",
    "max_playtime",
    "time_per_player",
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
            # Create embedding-specific preprocessor with appropriate defaults
            # (excludes designer/artist/publisher, focuses on game characteristics)
            preprocessor = create_embedding_preprocessor(
                model_type="linear",
                preserve_columns=["year_published", "predicted_complexity"],
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

        # Drop year_published columns - year should only be an ID variable, not a feature
        year_cols = [col for col in X.columns if col.startswith("year_published")]
        if year_cols:
            X = X.drop(columns=year_cols)
            logger.info(f"Dropped year columns from features: {year_cols}")

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

        # Compute reconstruction error for linear methods (PCA, SVD)
        # Skip for UMAP - its inverse_transform is unstable for high-dimensional targets
        # and not meaningful for non-linear manifold methods
        # Note: X_original is already scaled by the preprocessor
        is_umap = type(algorithm).__name__ == "UMAPEmbedding"
        if (
            not is_umap
            and hasattr(algorithm, "model")
            and hasattr(algorithm.model, "inverse_transform")
        ):
            try:
                X_reconstructed = algorithm.model.inverse_transform(embeddings)
                reconstruction_error = np.mean((X_original.values - X_reconstructed) ** 2)
                metrics["reconstruction_mse"] = float(reconstruction_error)
            except Exception as e:
                logger.warning(f"Could not compute reconstruction error: {e}")

        # Embedding statistics
        metrics["embedding_mean"] = float(np.mean(embeddings))
        metrics["embedding_std"] = float(np.std(embeddings))
        metrics["embedding_min"] = float(np.min(embeddings))
        metrics["embedding_max"] = float(np.max(embeddings))

        return metrics

    def _compute_and_save_umap_coordinates(
        self,
        exp_dir: Path,
        game_ids: list,
        fit_embeddings: np.ndarray,
        all_embeddings: np.ndarray,
        dataset_name: str,
        complexity_values: Optional[np.ndarray] = None,
        n_neighbors: int = 100,
        min_dist: float = 0.5,
    ) -> None:
        """Compute UMAP 2D projection and save coordinates and visualization.

        Fits UMAP on fit_embeddings (train+tune), then transforms all embeddings.

        Args:
            exp_dir: Experiment directory to save to.
            game_ids: List of game IDs corresponding to all_embeddings.
            fit_embeddings: Embeddings to fit UMAP on (typically train+tune filtered).
            all_embeddings: All embeddings (train + tune + test) to transform.
            dataset_name: Name of dataset (train/tune/test/all).
            complexity_values: Optional array of complexity values for coloring.
            n_neighbors: UMAP n_neighbors parameter.
            min_dist: UMAP min_dist parameter.
        """
        try:
            from umap import UMAP
        except ImportError:
            logger.warning("umap-learn not installed, skipping UMAP projection")
            return

        logger.info(f"Fitting UMAP on {len(fit_embeddings)} samples (train + tune filtered)...")

        umap_2d = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=42,
        )
        umap_2d.fit(fit_embeddings)

        # Save fitted UMAP model for later use (e.g., scoring service)
        umap_model_path = exp_dir / "umap_2d_model.pkl"
        with open(umap_model_path, "wb") as f:
            pickle.dump(umap_2d, f)
        logger.info(f"Saved fitted UMAP model to {umap_model_path}")

        logger.info(f"Transforming all embeddings ({len(all_embeddings)} samples)...")
        projection_2d = umap_2d.transform(all_embeddings)

        # Save as parquet with game_id
        umap_df = pl.DataFrame({
            "game_id": game_ids,
            "umap_1": projection_2d[:, 0],
            "umap_2": projection_2d[:, 1],
        })
        umap_path = exp_dir / f"{dataset_name}_umap_coords.parquet"
        umap_df.write_parquet(umap_path)
        logger.info(f"Saved UMAP 2D coordinates to {umap_path}")

        # Save 2D PNG visualization
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            projection_2d[:, 0],
            projection_2d[:, 1],
            c=complexity_values,
            cmap="viridis",
            s=2,
            alpha=0.5,
        )
        if complexity_values is not None:
            plt.colorbar(scatter, ax=ax, label="Predicted Complexity")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP 2D Projection - {dataset_name} ({len(projection_2d)} games)")

        png_path = exp_dir / f"{dataset_name}_umap_2d.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved UMAP 2D plot to {png_path}")

    def _save_visualization_data(
        self,
        exp_dir: Path,
        df: pl.DataFrame,
        embeddings: np.ndarray,
        dataset_name: str,
        algorithm: str = "",
        sample_size: int = 50000,
    ) -> None:
        """Save 2D projection data for visualization.

        Args:
            exp_dir: Experiment directory to save to.
            df: DataFrame with game metadata.
            embeddings: Full-dimensional embeddings.
            dataset_name: Name of dataset (train/tune/test).
            algorithm: Algorithm name (pca, svd, umap, autoencoder).
            sample_size: Max samples for visualization (for performance).
        """
        # Sample if dataset is large
        n_samples = len(embeddings)
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            embeddings_sample = embeddings[indices]
            df_sample = df[indices.tolist()]
        else:
            embeddings_sample = embeddings
            df_sample = df

        # Create 2D projection
        if embeddings_sample.shape[1] == 2:
            projection_2d = embeddings_sample
        elif embeddings_sample.shape[1] > 2:
            # For PCA/SVD, columns are already ordered by variance - just take first 2
            if algorithm in ("pca", "svd"):
                projection_2d = embeddings_sample[:, :2]
            else:
                # For UMAP/autoencoder, use PCA to reduce to 2D for visualization
                from sklearn.decomposition import PCA
                pca_2d = PCA(n_components=2)
                projection_2d = pca_2d.fit_transform(embeddings_sample)
        else:
            logger.warning("Embeddings have less than 2 dimensions, skipping 2D projection")
            return

        # Get complexity values for coloring
        if "predicted_complexity" in df_sample.columns:
            colors = df_sample["predicted_complexity"].to_numpy()
        else:
            colors = None

        # Create static plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(
            projection_2d[:, 0],
            projection_2d[:, 1],
            c=colors,
            cmap="viridis",
            s=2,
            alpha=0.5,
        )
        if colors is not None:
            plt.colorbar(scatter, ax=ax, label="Predicted Complexity")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"Game Embeddings - {dataset_name} ({len(projection_2d)} games)")

        viz_path = exp_dir / f"{dataset_name}_visualization_2d.png"
        fig.savefig(viz_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved 2D visualization plot ({len(projection_2d)} points) to {viz_path}")

    def _save_component_loading_plots(
        self,
        exp_dir: Path,
        artifacts: Dict[str, Any],
        n_components: int = 10,
        n_features: int = 15,
    ) -> None:
        """Save bar plots showing top feature loadings for each component.

        Args:
            exp_dir: Experiment directory to save to.
            artifacts: Algorithm artifacts containing components and feature names.
            n_components: Number of top components to plot (default 10).
            n_features: Number of top features to show per component (default 15).
        """
        if "components" not in artifacts or "feature_names" not in artifacts:
            logger.warning("No components found in artifacts, skipping loading plots")
            return

        import matplotlib.pyplot as plt

        components = np.array(artifacts["components"])
        feature_names = artifacts["feature_names"]
        n_actual_components = min(n_components, components.shape[0])

        # Create a figure with subplots for each component
        fig, axes = plt.subplots(
            n_actual_components, 1,
            figsize=(12, 4 * n_actual_components),
            squeeze=False
        )

        for i in range(n_actual_components):
            ax = axes[i, 0]
            component = components[i]

            # Get top features by absolute loading, then sort by value (positive to negative)
            abs_loadings = np.abs(component)
            top_indices = np.argsort(abs_loadings)[-n_features:]
            # Sort these indices by actual loading value (descending: positive first)
            top_indices = sorted(top_indices, key=lambda idx: component[idx], reverse=True)

            top_feature_names = [feature_names[idx] for idx in top_indices]
            top_loadings = [component[idx] for idx in top_indices]

            # Create horizontal bar plot (blue for positive, red for negative)
            colors = ['#3498db' if v > 0 else '#e74c3c' for v in top_loadings]
            y_pos = np.arange(len(top_feature_names))

            ax.barh(y_pos, top_loadings, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_feature_names, fontsize=9)
            ax.invert_yaxis()  # Top feature at top
            ax.axvline(x=0, color='black', linewidth=0.5)

            # Add explained variance if available
            if "explained_variance_ratio" in artifacts:
                var_ratio = artifacts["explained_variance_ratio"][i]
                ax.set_title(f"PC{i+1} - Top {n_features} Features (Explained Var: {var_ratio:.2%})")
            else:
                ax.set_title(f"PC{i+1} - Top {n_features} Features")

            ax.set_xlabel("Loading")

        plt.tight_layout()
        plot_path = exp_dir / "component_loadings.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved component loading plots to {plot_path}")

    def _save_scree_plot(
        self,
        exp_dir: Path,
        artifacts: Dict[str, Any],
    ) -> None:
        """Save scree plot showing variance explained by each component.

        Args:
            exp_dir: Experiment directory to save to.
            artifacts: Algorithm artifacts containing explained_variance_ratio.
        """
        if "explained_variance_ratio" not in artifacts:
            logger.warning("No explained_variance_ratio in artifacts, skipping scree plot")
            return

        import matplotlib.pyplot as plt

        var_ratio = np.array(artifacts["explained_variance_ratio"])
        cumulative_var = np.cumsum(var_ratio)
        n_components = len(var_ratio)
        x = np.arange(1, n_components + 1)

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar plot for individual variance
        ax1.bar(x, var_ratio * 100, alpha=0.7, color='#3498db', label='Individual')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#3498db')

        # Line plot for cumulative variance on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, cumulative_var * 100, 'o-', color='#e74c3c', linewidth=2, markersize=4, label='Cumulative')
        ax2.set_ylabel('Cumulative Variance Explained (%)', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax2.set_ylim(0, 105)

        # Add horizontal lines at key thresholds
        for thresh in [80, 90, 95]:
            ax2.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            ax2.text(n_components + 0.5, thresh, f'{thresh}%', va='center', fontsize=8, color='gray')

        ax1.set_title(f'Scree Plot - Variance Explained by Component (Total: {cumulative_var[-1]:.1%})')
        ax1.set_xticks(x[::max(1, n_components // 20)])  # Show ~20 tick labels max

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.tight_layout()
        plot_path = exp_dir / "scree_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved scree plot to {plot_path}")

    def train(
        self,
        algorithm: str,
        embedding_dim: int,
        experiment_name: str,
        algorithm_params: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        min_ratings: int = 25,
    ) -> Tuple[BaseEmbeddingAlgorithm, Pipeline, Dict[str, Any]]:
        """Train an embedding model.

        Args:
            algorithm: Algorithm name ('pca', 'svd', 'umap', 'autoencoder').
            embedding_dim: Target embedding dimension.
            experiment_name: Name for the experiment.
            algorithm_params: Algorithm-specific parameters.
            description: Experiment description.
            min_ratings: Minimum users_rated for training data only.
                         Games with fewer ratings are excluded from training
                         but still included in tune/test evaluation.

        Returns:
            Tuple of (fitted algorithm, preprocessor, metrics dict).
        """
        years = self.config.years
        algorithm_params = algorithm_params or {}

        # Load ALL data from BigQuery (no min_ratings filter)
        # This allows us to embed any game, even those with few ratings
        logger.info("Loading embedding data from BigQuery...")
        df = self.load_embedding_data(end_year=years.test_end, min_ratings=0)

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

        # Filter training and tune data by min_ratings
        # Only train on games with sufficient ratings (more meaningful data)
        train_df_unfiltered = train_df
        tune_df_unfiltered = tune_df
        if min_ratings > 0:
            train_df = train_df.filter(pl.col("users_rated") >= min_ratings)
            tune_df_filtered = tune_df.filter(pl.col("users_rated") >= min_ratings)
            logger.info(
                f"Filtered training data: {len(train_df_unfiltered)} -> {len(train_df)} "
                f"games with users_rated >= {min_ratings}"
            )
            logger.info(
                f"Filtered tune data: {len(tune_df_unfiltered)} -> {len(tune_df_filtered)} "
                f"games with users_rated >= {min_ratings}"
            )
        else:
            tune_df_filtered = tune_df

        logger.info(
            f"Split sizes - Train: {len(train_df)} (filtered), "
            f"Tune: {len(tune_df)}, Test: {len(test_df)}"
        )

        # Prepare features
        logger.info("Preparing features...")
        train_X, preprocessor = self.prepare_features(train_df, fit=True)
        tune_X, _ = self.prepare_features(tune_df, preprocessor=preprocessor, fit=False)
        test_X, _ = self.prepare_features(test_df, preprocessor=preprocessor, fit=False)

        # Create and train embedding algorithm on train data only for evaluation
        logger.info(f"Training {algorithm} with {embedding_dim} dimensions on train data...")
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

        # Refit on combined train + tune data for final model
        # Use filtered tune data (same min_ratings filter as train)
        logger.info("Refitting model on combined train + tune data...")
        combined_df = pl.concat([train_df, tune_df_filtered])
        combined_X, _ = self.prepare_features(combined_df, preprocessor=preprocessor, fit=False)

        # Create fresh model for final fit
        final_model = create_embedding_algorithm(
            algorithm=algorithm,
            embedding_dim=embedding_dim,
            **algorithm_params,
        )
        final_model.fit(combined_X)
        logger.info(f"Final model trained on {len(combined_X)} samples (train + tune)")

        # Generate embeddings for the combined training data (for UMAP fitting)
        combined_embeddings = final_model.transform(combined_X)

        # Use final model for generating embeddings going forward
        embedding_model = final_model

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
                "min_ratings": min_ratings,
                "train_samples_before_filter": len(train_df_unfiltered) if min_ratings > 0 else len(train_df),
                "train_samples_after_filter": len(train_df),
                "tune_samples": len(tune_df),
                "test_samples": len(test_df),
                "final_model_samples": len(combined_df),
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
                    "transformer": "EmbeddingTransformer",
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

        # Save raw data used for training as artifact
        id_cols = ["game_id", "name", "year_published", "users_rated", "average_rating", "predicted_complexity"]
        available_id_cols = [c for c in id_cols if c in train_df.columns]
        train_df.select(available_id_cols).write_parquet(experiment.exp_dir / "train_data.parquet")
        tune_df.select(available_id_cols).write_parquet(experiment.exp_dir / "tune_data.parquet")
        test_df.select(available_id_cols).write_parquet(experiment.exp_dir / "test_data.parquet")
        logger.info(f"Saved raw data splits to {experiment.exp_dir}")

        # Log metrics
        experiment.log_metrics(train_metrics, "train")
        experiment.log_metrics(tune_metrics, "tune")
        experiment.log_metrics(test_metrics, "test")

        # Log parameters
        experiment.log_parameters({
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
            "min_ratings": min_ratings,
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

        # Log model info (final model trained on train + tune)
        experiment.log_model_info({
            "algorithm": algorithm,
            "embedding_dim": embedding_dim,
            "n_features_in": train_X.shape[1],
            "n_samples_trained": len(combined_X),
            "n_train_samples": len(train_X),
            "n_tune_samples": len(tune_X),
            **embedding_model.get_metrics(),
        })

        # Save algorithm-specific artifacts (loadings, components, etc.)
        feature_names = list(train_X.columns)
        artifacts = embedding_model.get_artifacts(feature_names=feature_names)
        if artifacts:
            artifacts_path = experiment.exp_dir / "artifacts.json"
            with open(artifacts_path, "w") as f:
                json.dump(artifacts, f, indent=2)
            logger.info(f"Saved algorithm artifacts to {artifacts_path}")

            # For PCA/SVD, save a readable loadings CSV (features x components)
            if "components" in artifacts and "feature_names" in artifacts:
                components = np.array(artifacts["components"])
                loadings_df = pd.DataFrame(
                    components.T,  # Transpose: rows=features, cols=components
                    index=artifacts["feature_names"],
                    columns=[f"PC{i+1}" for i in range(components.shape[0])],
                )
                loadings_df.index.name = "feature"
                loadings_path = experiment.exp_dir / "component_loadings.csv"
                loadings_df.to_csv(loadings_path)
                logger.info(f"Saved component loadings to {loadings_path}")

                # Generate bar plots of top features per component
                self._save_component_loading_plots(experiment.exp_dir, artifacts)

                # Generate scree plot
                self._save_scree_plot(experiment.exp_dir, artifacts)

        # Generate 2D projection for visualization
        self._save_visualization_data(
            experiment.exp_dir,
            train_df,
            train_embeddings,
            "train",
            algorithm=algorithm,
        )

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

        # Compute and save UMAP 2D projections for visualization
        # Fit UMAP on training data only, then transform all datasets
        all_game_ids = (
            train_df["game_id"].to_list()
            + tune_df["game_id"].to_list()
            + test_df["game_id"].to_list()
        )
        all_embeddings = np.vstack([train_embeddings, tune_embeddings, test_embeddings])

        # Get complexity values for coloring the UMAP plot
        all_complexity = None
        if "predicted_complexity" in train_df.columns:
            all_complexity = np.concatenate([
                train_df["predicted_complexity"].to_numpy(),
                tune_df["predicted_complexity"].to_numpy(),
                test_df["predicted_complexity"].to_numpy(),
            ])

        self._compute_and_save_umap_coordinates(
            experiment.exp_dir,
            all_game_ids,
            fit_embeddings=combined_embeddings,
            all_embeddings=all_embeddings,
            dataset_name="all",
            complexity_values=all_complexity,
        )

        logger.info(f"Experiment saved to {experiment.exp_dir}")

        all_metrics = {
            "train": train_metrics,
            "tune": tune_metrics,
            "test": test_metrics,
        }

        return embedding_model, preprocessor, all_metrics
