"""Data loading and preprocessing for BGG predictive models."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from google.cloud import bigquery

from .config import BigQueryConfig


class BGGDataLoader:
    """Loader for BGG data from BigQuery warehouse."""

    def __init__(self, config: BigQueryConfig):
        """Initialize loader.
        
        Args:
            config: BigQuery configuration
        """
        self.config = config
        self.client = config.get_client()

    def load_training_data(
        self,
        end_train_year: int = 2021,
        min_ratings: int = 25,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Load training data from warehouse.
        
        Args:
            end_train_year: Last year to include in training
            min_ratings: Minimum number of ratings threshold
            
        Returns:
            Tuple of (features_df, target_dict) where target_dict contains:
            - hurdle: Binary indicating if game reached min_ratings
            - complexity: Game weight/complexity scores
            - rating: Average ratings
            - users_rated: Number of ratings
        """
        # Load data directly from BigQuery
        query = f"""
        SELECT *
        FROM `{self.config.project_id}.{self.config.dataset}.games_features`
        WHERE year_published <= {end_train_year}
        """
        
        # Load data
        df = self.client.query(query).to_dataframe()
        
        # Process features
        features = self._process_features(df)
        
        # Create target variables
        targets = {
            "hurdle": (df["users_rated"] >= min_ratings).astype(int),
            "complexity": df["average_weight"],
            "rating": df["average_rating"],
            "users_rated": df["users_rated"],
        }
        
        return features, targets

    def _process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw features into model-ready format.
        
        Args:
            df: Raw DataFrame from BigQuery
            
        Returns:
            Processed feature matrix
        """
        # Start with numeric features
        features = pd.DataFrame({
            "year_published": df["year_published"],
            "min_players": df["min_players"],
            "max_players": df["max_players"],
            "min_playtime": df["min_playtime"],
            "max_playtime": df["max_playtime"],
            "min_age": df["min_age"],
            "publisher_count": df["publisher_count"],
            "designer_count": df["designer_count"],
        })
        
        # Add derived numeric features
        features["player_range"] = features["max_players"] - features["min_players"]
        features["playtime_range"] = features["max_playtime"] - features["min_playtime"]
        
        # One-hot encode categories and mechanics
        category_dummies = self._get_array_dummies(
            df["categories"],
            prefix="category",
            min_freq=100,  # Only keep common categories
        )
        mechanic_dummies = self._get_array_dummies(
            df["mechanics"],
            prefix="mechanic",
            min_freq=100,  # Only keep common mechanics
        )
        
        # Combine all features
        features = pd.concat(
            [features, category_dummies, mechanic_dummies],
            axis=1,
        )
        
        return features

    def _get_array_dummies(
        self,
        array_series: pd.Series,
        prefix: str,
        min_freq: int = 100,
    ) -> pd.DataFrame:
        """Convert array column to dummy variables.
        
        Args:
            array_series: Series containing arrays of strings
            prefix: Prefix for dummy column names
            min_freq: Minimum frequency to keep a value
            
        Returns:
            DataFrame of dummy variables
        """
        # Explode arrays into long format
        values = array_series.explode()
        
        # Get value counts and filter by frequency
        value_counts = values.value_counts()
        keep_values = value_counts[value_counts >= min_freq].index
        
        # Create dummy variables only for frequent values
        dummies = pd.get_dummies(
            values[values.isin(keep_values)],
            prefix=prefix,
        )
        
        # Aggregate back to original shape
        return dummies.groupby(level=0).max()

    def load_prediction_data(
        self,
        game_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Load feature data for making predictions.
        
        Args:
            game_ids: Optional list of specific game IDs to load
            
        Returns:
            Feature matrix ready for prediction
        """
        # Modify query based on game_ids
        if game_ids:
            game_id_filter = f"WHERE game_id IN ({','.join(map(str, game_ids))})"
        else:
            game_id_filter = ""
            
        query = f"""
        SELECT *
        FROM `{self.config.project_id}.{self.config.dataset}.games_features`
        {game_id_filter}
        """
        
        # Load and process data
        df = self.client.query(query).to_dataframe()
        features = self._process_features(df)
        
        return features
