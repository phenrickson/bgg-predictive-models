"""Explain predictions from linear models.

Provides feature contribution breakdowns for any linear model
(Ridge, ARD, BayesianRidge, Lasso, LogisticRegression, etc.)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class LinearExplainer:
    """Explain predictions from linear models.

    For any linear model, the prediction is:
        y = intercept + sum(coefficient_i * feature_value_i)

    This class computes the contribution of each feature to the prediction.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        experiment_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize explainer with a trained pipeline.

        Args:
            pipeline: Trained sklearn Pipeline with 'preprocessor' and 'model' steps.
            experiment_dir: Optional path to experiment directory containing
                coefficients.csv and data/ folder. If provided, feature names
                are loaded from coefficients.csv for accuracy.
        """
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Expected sklearn Pipeline")

        if "preprocessor" not in pipeline.named_steps:
            raise ValueError("Pipeline must have 'preprocessor' step")
        if "model" not in pipeline.named_steps:
            raise ValueError("Pipeline must have 'model' step")

        self.pipeline = pipeline
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["model"]
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None

        # Validate model has coefficients
        if not hasattr(self.model, "coef_"):
            raise ValueError(
                f"Model {type(self.model).__name__} does not have coef_ attribute. "
                "LinearExplainer only works with linear models."
            )

        self.coef_ = self.model.coef_
        self.intercept_ = getattr(self.model, "intercept_", 0.0)
        self.feature_names_ = self._get_feature_names()

        # Validate dimensions match
        if len(self.feature_names_) != len(self.coef_):
            logger.warning(
                f"Feature names ({len(self.feature_names_)}) and coefficients "
                f"({len(self.coef_)}) have different lengths. Using indices."
            )
            self.feature_names_ = [f"feature_{i}" for i in range(len(self.coef_))]

    def _get_feature_names(self) -> list:
        """Extract feature names by transforming sample data through preprocessor.

        The preprocessor outputs a DataFrame with column names that match the
        order of model coefficients. This is more reliable than coefficients.csv
        which may be sorted differently.
        """
        # Try to get feature names from experiment data by transforming a sample
        if self.experiment_dir is not None:
            data_dir = self.experiment_dir / "data"
            for split in ["test", "train", "tune"]:
                parquet_path = data_dir / f"{split}.parquet"
                if parquet_path.exists():
                    try:
                        df = pl.read_parquet(parquet_path)
                        sample = df.head(1).to_pandas()
                        X_transformed = self.preprocessor.transform(sample)

                        if isinstance(X_transformed, pd.DataFrame):
                            names = X_transformed.columns.tolist()
                            if len(names) == len(self.coef_):
                                logger.info(f"Got {len(names)} feature names from preprocessor output")
                                return names
                            else:
                                logger.warning(
                                    f"Preprocessor output has {len(names)} columns but model has "
                                    f"{len(self.coef_)} coefficients"
                                )
                        break
                    except Exception as e:
                        logger.warning(f"Could not get feature names from {split} data: {e}")

        # Fallback to preprocessor methods
        try:
            if hasattr(self.preprocessor, "named_steps"):
                bgg_prep = self.preprocessor.named_steps.get("bgg_preprocessor")
                if bgg_prep and hasattr(bgg_prep, "get_feature_names_out"):
                    names = list(bgg_prep.get_feature_names_out())
                    if len(names) == len(self.coef_):
                        return names

            if hasattr(self.preprocessor, "get_feature_names_out"):
                names = list(self.preprocessor.get_feature_names_out())
                if len(names) == len(self.coef_):
                    return names

            return [f"feature_{i}" for i in range(len(self.coef_))]
        except Exception as e:
            logger.warning(f"Could not get feature names: {e}")
            return [f"feature_{i}" for i in range(len(self.coef_))]

    def explain(self, X: pd.DataFrame, top_n: Optional[int] = None) -> pd.DataFrame:
        """Explain prediction for a single sample.

        Args:
            X: DataFrame with raw features (single row or first row used).
            top_n: If provided, return only top N contributors by absolute value.

        Returns:
            DataFrame with columns:
                - feature: Feature name
                - raw_value: Original value from input data (if available)
                - value: Transformed feature value
                - coefficient: Model coefficient
                - contribution: value * coefficient
            Sorted by absolute contribution (descending).
        """
        if len(X) == 0:
            raise ValueError("Empty DataFrame provided")

        # Use first row if multiple provided
        if len(X) > 1:
            logger.warning(f"Multiple rows provided ({len(X)}), using first row only")
            X = X.iloc[[0]]

        # Get pre-standardized values from BGGPreprocessor (if available)
        pre_standardized = None
        if hasattr(self.preprocessor, "named_steps"):
            bgg_prep = self.preprocessor.named_steps.get("bgg_preprocessor")
            if bgg_prep:
                pre_standardized = bgg_prep.transform(X)

        # Transform features through full pipeline
        X_transformed = self.preprocessor.transform(X)

        # Handle sparse matrices
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        # Handle DataFrame output from preprocessor
        if isinstance(X_transformed, pd.DataFrame):
            X_transformed = X_transformed.values

        values = X_transformed[0]
        contributions = values * self.coef_

        # Get raw (pre-standardized) values for each feature
        raw_values = []
        for feature_name in self.feature_names_:
            raw_val = None
            if pre_standardized is not None and feature_name in pre_standardized.columns:
                raw_val = pre_standardized[feature_name].iloc[0]
            raw_values.append(raw_val)

        result = pd.DataFrame({
            "feature": self.feature_names_,
            "raw_value": raw_values,
            "value": values,
            "coefficient": self.coef_,
            "contribution": contributions,
        })

        # Sort by absolute contribution
        result["abs_contribution"] = np.abs(result["contribution"])
        result = result.sort_values("abs_contribution", ascending=False)
        result = result.drop(columns=["abs_contribution"])

        if top_n is not None:
            result = result.head(top_n)

        return result.reset_index(drop=True)

    def explain_with_prediction(
        self, X: pd.DataFrame, top_n: Optional[int] = None
    ) -> dict:
        """Explain prediction and return prediction value.

        Args:
            X: DataFrame with raw features.
            top_n: If provided, return only top N contributors.

        Returns:
            Dictionary with:
                - prediction: The model's prediction
                - intercept: Model intercept
                - contributions: DataFrame of feature contributions
        """
        explanation = self.explain(X, top_n=top_n)
        prediction = self.pipeline.predict(X)[0]

        return {
            "prediction": prediction,
            "intercept": self.intercept_,
            "contributions": explanation,
        }

    def explain_from_parquet(
        self,
        game_id: int,
        parquet_path: Union[str, Path],
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load game from predictions parquet and explain.

        Args:
            game_id: Game ID to explain.
            parquet_path: Path to predictions parquet file.
            top_n: If provided, return only top N contributors.

        Returns:
            DataFrame of feature contributions.

        Raises:
            ValueError: If game_id not found in parquet.
        """
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        df = pl.read_parquet(parquet_path)
        game_df = df.filter(pl.col("game_id") == game_id)

        if len(game_df) == 0:
            raise ValueError(f"Game ID {game_id} not found in {parquet_path}")

        return self.explain(game_df.to_pandas(), top_n=top_n)

    def explain_from_experiment_data(
        self,
        game_id: int,
        split: str = "test",
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load game from experiment's data folder and explain.

        This uses the exact data that was used during training/evaluation,
        ensuring feature consistency.

        Args:
            game_id: Game ID to explain.
            split: Which data split to search ('train', 'tune', or 'test').
            top_n: If provided, return only top N contributors.

        Returns:
            DataFrame of feature contributions.

        Raises:
            ValueError: If experiment_dir not set or game_id not found.
        """
        if self.experiment_dir is None:
            raise ValueError(
                "experiment_dir must be set to use explain_from_experiment_data"
            )

        data_dir = self.experiment_dir / "data"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        parquet_path = data_dir / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Data file not found: {parquet_path}")

        df = pl.read_parquet(parquet_path)
        game_df = df.filter(pl.col("game_id") == game_id)

        if len(game_df) == 0:
            # Try other splits if not found
            for other_split in ["train", "tune", "test"]:
                if other_split == split:
                    continue
                other_path = data_dir / f"{other_split}.parquet"
                if other_path.exists():
                    other_df = pl.read_parquet(other_path)
                    game_df = other_df.filter(pl.col("game_id") == game_id)
                    if len(game_df) > 0:
                        logger.info(f"Found game {game_id} in {other_split} split")
                        break

        if len(game_df) == 0:
            raise ValueError(
                f"Game ID {game_id} not found in experiment data "
                f"(searched train, tune, test)"
            )

        return self.explain(game_df.to_pandas(), top_n=top_n)

    def explain_game(
        self,
        game_id: int,
        top_n: Optional[int] = 15,
    ) -> dict:
        """Convenience method to explain a game with full context.

        Searches experiment data for the game and returns a complete explanation
        including game name, actual value, prediction, and contributions.

        Args:
            game_id: Game ID to explain.
            top_n: Number of top contributors to include (default 15).

        Returns:
            Dictionary with game_id, game_name, actual, prediction,
            intercept, and contributions DataFrame.
        """
        if self.experiment_dir is None:
            raise ValueError("experiment_dir must be set to use explain_game")

        data_dir = self.experiment_dir / "data"

        # Search all splits for the game
        game_df = None
        for split in ["test", "tune", "train"]:
            parquet_path = data_dir / f"{split}.parquet"
            if parquet_path.exists():
                df = pl.read_parquet(parquet_path)
                found = df.filter(pl.col("game_id") == game_id)
                if len(found) > 0:
                    game_df = found.to_pandas()
                    break

        if game_df is None:
            raise ValueError(f"Game ID {game_id} not found in experiment data")

        # Get game info
        game_name = game_df["name"].iloc[0] if "name" in game_df.columns else None

        # Determine target column from metadata if available
        actual = None
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            target_col = metadata.get("metadata", {}).get("target_column")
            if target_col and target_col in game_df.columns:
                actual = game_df[target_col].iloc[0]

        # Get prediction and explanation
        prediction = self.pipeline.predict(game_df)[0]
        contributions = self.explain(game_df, top_n=top_n)

        return {
            "game_id": game_id,
            "game_name": game_name,
            "actual": actual,
            "prediction": prediction,
            "intercept": self.intercept_,
            "contributions": contributions,
        }

    def plot_explanation(
        self,
        game_id: int,
        top_n: int = 15,
        save_path: Optional[Union[str, Path]] = None,
        figsize: tuple = (10, 8),
    ):
        """Plot a SHAP-style force plot showing feature contributions.

        Bars extend from the intercept (base value), with positive contributions
        pushing right and negative pushing left. Features sorted by absolute
        contribution with largest at top.

        Args:
            game_id: Game ID to explain and plot.
            top_n: Number of top features to show individually (default 15).
            save_path: Optional path to save the figure.
            figsize: Figure size tuple (width, height).

        Returns:
            matplotlib Figure object.
        """
        import matplotlib.pyplot as plt

        if self.experiment_dir is None:
            raise ValueError("experiment_dir must be set to use plot_explanation")

        # Get full explanation (all features) to calculate "other"
        data_dir = self.experiment_dir / "data"
        game_df = None
        for split in ["test", "tune", "train"]:
            parquet_path = data_dir / f"{split}.parquet"
            if parquet_path.exists():
                df = pl.read_parquet(parquet_path)
                found = df.filter(pl.col("game_id") == game_id)
                if len(found) > 0:
                    game_df = found.to_pandas()
                    break

        if game_df is None:
            raise ValueError(f"Game ID {game_id} not found in experiment data")

        game_name = game_df["name"].iloc[0] if "name" in game_df.columns else f"Game {game_id}"

        # Get actual from metadata
        actual = None
        metadata_path = self.experiment_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            target_col = metadata.get("metadata", {}).get("target_column")
            if target_col and target_col in game_df.columns:
                actual = game_df[target_col].iloc[0]

        # Get ALL contributions (includes value column)
        all_contributions = self.explain(game_df, top_n=None)
        prediction = self.pipeline.predict(game_df)[0]
        intercept = self.intercept_

        # Sort by absolute contribution to get top N
        all_contributions["abs_contrib"] = np.abs(all_contributions["contribution"])
        all_contributions = all_contributions.sort_values("abs_contrib", ascending=False)

        # Split into top N and others
        top_contributions = all_contributions.head(top_n).copy()
        other_contributions = all_contributions.iloc[top_n:]
        other_sum = other_contributions["contribution"].sum()

        # Add "other" row if there are remaining contributions
        if len(other_contributions) > 0:
            other_row = pd.DataFrame([{
                "feature": f"{len(other_contributions)} other features",
                "raw_value": np.nan,
                "value": np.nan,
                "coefficient": np.nan,
                "contribution": other_sum,
                "abs_contrib": np.abs(other_sum),
            }])
            top_contributions = pd.concat([top_contributions, other_row], ignore_index=True)

        # Sort by absolute contribution descending (largest at top)
        top_contributions = top_contributions.sort_values("abs_contrib", ascending=False)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        y_pos = np.arange(len(top_contributions))
        bar_height = 0.7

        # Colors
        color_positive = "#2196F3"  # Blue for positive
        color_negative = "#E91E63"  # Pink for negative

        # Draw bars extending from intercept
        for i, (_, row) in enumerate(top_contributions.iterrows()):
            contrib = row["contribution"]
            color = color_positive if contrib >= 0 else color_negative

            if contrib >= 0:
                ax.barh(i, contrib, left=intercept, color=color, height=bar_height, edgecolor="white")
            else:
                ax.barh(i, contrib, left=intercept, color=color, height=bar_height, edgecolor="white")

            # Add contribution value inside bar
            bar_center = intercept + contrib / 2
            text_color = "white"
            ax.text(bar_center, i, f"{contrib:+.2f}", va="center", ha="center",
                    fontsize=9, fontweight="bold", color=text_color)

        # Y-axis labels: "raw_value = feature_name" format (use raw values when available)
        y_labels = []
        for _, row in top_contributions.iterrows():
            raw_val = row.get("raw_value")
            if pd.isna(raw_val):
                y_labels.append(row["feature"])
            else:
                # Format value nicely
                if isinstance(raw_val, (int, np.integer)) or (isinstance(raw_val, float) and raw_val == int(raw_val)):
                    val_str = f"{int(raw_val)}"
                elif abs(raw_val) >= 1000:
                    val_str = f"{raw_val:.0f}"
                elif abs(raw_val) >= 1:
                    val_str = f"{raw_val:.1f}"
                else:
                    val_str = f"{raw_val:.2f}"
                y_labels.append(f"{val_str} = {row['feature']}")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.invert_yaxis()  # Put largest at top

        # Reference line for intercept (base value)
        ax.axvline(x=intercept, color="gray", linewidth=1, linestyle="-", alpha=0.7)

        # Set x-axis limits centered around contributions
        max_contrib = top_contributions["contribution"].abs().max()
        padding = max_contrib * 0.3
        ax.set_xlim(intercept - max_contrib - padding, intercept + max_contrib + padding)

        # Labels
        ax.set_xlabel(f"Average = {intercept:.2f}", fontsize=10)
        title = f"{game_name}\nPredicted: {prediction:.2f}"
        if actual is not None:
            title += f"  |  Actual: {actual:.2f}"
        ax.set_title(title, fontsize=12, fontweight="bold")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")

        return fig

    def explain_from_bigquery(
        self,
        game_id: int,
        use_embeddings: bool = True,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load game from BigQuery and explain.

        Args:
            game_id: Game ID to explain.
            use_embeddings: Whether to load embeddings.
            top_n: If provided, return only top N contributors.

        Returns:
            DataFrame of feature contributions.

        Raises:
            ValueError: If game_id not found.
        """
        from src.utils.config import load_config

        config = load_config()
        bq_config = config.get_bigquery_config()
        client = bq_config.get_client()

        if use_embeddings:
            query = f"""
            SELECT
                f.*,
                e.embedding
            FROM `{bq_config.project_id}.{bq_config.dataset}.{bq_config.table}` f
            LEFT JOIN `bgg-data-warehouse.predictions.bgg_description_embeddings` e
                ON f.game_id = e.game_id
            WHERE f.game_id = {game_id}
            """
        else:
            query = f"""
            SELECT *
            FROM `{bq_config.project_id}.{bq_config.dataset}.{bq_config.table}`
            WHERE game_id = {game_id}
            """

        logger.info(f"Loading game {game_id} from BigQuery")
        result = client.query(query).to_dataframe()

        if len(result) == 0:
            raise ValueError(f"Game ID {game_id} not found in BigQuery")

        # Expand embeddings if present
        if "embedding" in result.columns and result["embedding"].iloc[0] is not None:
            embedding = result["embedding"].iloc[0]
            emb_cols = {f"emb_{i}": [v] for i, v in enumerate(embedding)}
            emb_df = pd.DataFrame(emb_cols)
            result = pd.concat([result.drop(columns=["embedding"]), emb_df], axis=1)

        return self.explain(result, top_n=top_n)

    def summary(self) -> dict:
        """Get summary statistics about the model.

        Returns:
            Dictionary with model summary info.
        """
        return {
            "model_type": type(self.model).__name__,
            "n_features": len(self.coef_),
            "intercept": self.intercept_,
            "top_positive_features": self._top_features(positive=True, n=10),
            "top_negative_features": self._top_features(positive=False, n=10),
        }

    def _top_features(self, positive: bool = True, n: int = 10) -> list:
        """Get top features by coefficient magnitude."""
        df = pd.DataFrame({
            "feature": self.feature_names_,
            "coefficient": self.coef_,
        })

        if positive:
            df = df[df["coefficient"] > 0].nlargest(n, "coefficient")
        else:
            df = df[df["coefficient"] < 0].nsmallest(n, "coefficient")

        return df.to_dict("records")


def load_explainer(
    experiment_name: str,
    model_type: str,
    base_dir: Optional[str] = None,
) -> LinearExplainer:
    """Load an explainer for a trained experiment.

    Args:
        experiment_name: Name of the experiment.
        model_type: Type of model (complexity, rating, etc.)
        base_dir: Optional base directory for experiments.

    Returns:
        LinearExplainer instance with experiment_dir set for feature name accuracy.
    """
    from src.models.score import load_model
    from src.models.experiments import ExperimentTracker

    pipeline = load_model(experiment_name, model_type, base_dir=base_dir)

    # Get experiment directory for coefficients.csv access
    if base_dir is None:
        base_dir = "models/experiments"

    tracker = ExperimentTracker(model_type, base_dir=base_dir)
    experiments = tracker.list_experiments()

    # Find matching experiment
    matching = [exp for exp in experiments if exp["name"] == experiment_name]
    if matching:
        latest = max(matching, key=lambda x: x["version"])
        experiment = tracker.load_experiment(latest["name"], latest["version"])

        # Check for version subdirectories
        version_dirs = [
            d for d in experiment.exp_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        ]
        if version_dirs:
            experiment_dir = max(version_dirs, key=lambda x: int(x.name[1:]))
        else:
            experiment_dir = experiment.exp_dir
    else:
        experiment_dir = None

    return LinearExplainer(pipeline, experiment_dir=experiment_dir)
