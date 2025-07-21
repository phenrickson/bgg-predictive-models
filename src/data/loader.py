"""Data loading for BGG predictive models."""

from typing import Dict, List, Optional, Tuple, Union

import polars as pl
from google.cloud import bigquery

from .config import BigQueryConfig


class BGGDataLoader:
    """Simple loader for BGG data from BigQuery warehouse."""

    def __init__(self, config: BigQueryConfig):
        """Initialize loader.

        Args:
            config: BigQuery configuration
        """
        self.config = config
        self.client = config.get_client()

    def load_data(
        self,
        where_clause: str = "",
        preprocessor=None,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, pl.Series]]]:
        """Load data from BigQuery warehouse.

        Args:
            where_clause: Optional WHERE clause for the SQL query
            preprocessor: Optional preprocessor to transform data

        Returns:
            If preprocessor is None:
                Raw DataFrame from BigQuery
            If preprocessor is provided:
                Tuple of (features_df, target_dict)
        """
        # Build query with optional WHERE clause
        query = f"""
        SELECT 
            *,
            CASE WHEN users_rated >= 25 THEN 1 ELSE 0 END as hurdle,
            average_weight as complexity,
            average_rating as rating,
            LN(users_rated + 1) as log_users_rated
        FROM `{self.config.project_id}.{self.config.dataset}.games_features_materialized`
        """

        # Add year_published IS NOT NULL filter
        if not where_clause:
            query += " WHERE year_published IS NOT NULL"
        else:
            query += f" WHERE ({where_clause}) AND year_published IS NOT NULL"

        # Execute query and convert to polars DataFrame
        print(f"Executing query: {query}")
        try:
            query_job = self.client.query(query)
            print("Query job created successfully")

            # Check job status and get results
            query_job.result(timeout=30)  # Wait for the query to complete
            print("Query job completed successfully")

            pandas_df = query_job.to_dataframe()
            df = pl.from_pandas(pandas_df)

            print(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
            return df

        except Exception as e:
            print(f"Error executing query: {str(e)}")
            # If possible, print additional job error details
            if hasattr(e, "errors"):
                print("Query Errors:")
                for error in e.errors:
                    print(error)
            raise

        # Apply preprocessor if provided
        if preprocessor is not None:
            return preprocessor.fit_transform(df)

        # Otherwise return raw data
        return df

    def load_training_data(
        self,
        end_train_year: int = 2021,
        min_ratings: int = 25,
        min_weights: Optional[float] = None,
        preprocessor=None,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, pl.Series]]]:
        """Load training data from warehouse.

        Args:
            end_train_year: Last year to include in training
            min_ratings: Minimum number of ratings threshold
            min_weights: Optional minimum game complexity weight threshold
            preprocessor: Optional preprocessor to transform data

        Returns:
            If preprocessor is None:
                Raw DataFrame from BigQuery
            If preprocessor is provided:
                Tuple of (features_df, target_dict)
        """
        where_clauses = [f"year_published <= {end_train_year}"]

        # Add min_ratings filter
        where_clauses.append(f"users_rated >= {min_ratings}")

        if min_weights is not None:
            where_clauses.append(f"num_weights >= {min_weights}")

        where_clause = " AND ".join(where_clauses)
        return self.load_data(where_clause, preprocessor)

    def load_prediction_data(
        self,
        game_ids: Optional[List[int]] = None,
        preprocessor=None,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, Dict[str, pl.Series]]]:
        """Load data for making predictions.

        Args:
            game_ids: Optional list of specific game IDs to load
            preprocessor: Optional preprocessor to transform data

        Returns:
            If preprocessor is None:
                Raw DataFrame from BigQuery
            If preprocessor is provided:
                Processed feature matrix
        """
        where_clause = ""
        if game_ids:
            where_clause = f"game_id IN ({','.join(map(str, game_ids))})"

        return self.load_data(where_clause, preprocessor)
