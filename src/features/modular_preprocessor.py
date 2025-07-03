"""Modular scikit-learn compatible BGG preprocessors."""
from typing import List, Optional, Callable
import pandas as pd
import numpy as np
import re
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values with configurable imputation and transformation strategies.
    
    Parameters
    ----------
    columns : List[str], optional
        Specific columns to process. If None, processes all numeric columns.
    
    create_indicators : bool, default=False
        Whether to create missingness indicator columns.
    
    zero_to_nan : bool, default=True
        Whether to replace zeros with NaN.
    
    log_transform_columns : List[str], optional
        Columns to apply log transformation.
    
    imputation_strategy : str, default='median'
        Strategy for imputing missing values. Options: 'median', 'mean', 'zero'.
    """
    
    def __init__(
        self, 
        columns: Optional[List[str]] = None,
        create_indicators: bool = False,
        zero_to_nan: bool = True,
        log_transform_columns: Optional[List[str]] = None,
        imputation_strategy: str = 'median'
    ):
        self.columns = columns
        self.create_indicators = create_indicators
        self.zero_to_nan = zero_to_nan
        self.log_transform_columns = log_transform_columns or []
        self.imputation_strategy = imputation_strategy
        
        self.imputation_values_ = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Learn imputation values from training data."""
        X_copy = X.copy()
        
        # Determine columns to process
        if self.columns is None:
            self.columns = X_copy.select_dtypes(include=['int64', 'float64']).columns
        
        for col in self.columns:
            # Replace zeros with NaN if specified
            if self.zero_to_nan:
                X_copy[col] = X_copy[col].replace(0, np.nan)
            
            # Compute imputation value based on strategy
            if self.imputation_strategy == 'median':
                non_zero_vals = X_copy[X_copy[col] > 0][col]
                self.imputation_values_[col] = non_zero_vals.median() if len(non_zero_vals) > 0 else 1.0
            elif self.imputation_strategy == 'mean':
                non_zero_vals = X_copy[X_copy[col] > 0][col]
                self.imputation_values_[col] = non_zero_vals.mean() if len(non_zero_vals) > 0 else 1.0
            else:  # 'zero'
                self.imputation_values_[col] = 0.0
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value handling and transformations."""
        X_copy = X.copy()
        result_dfs = []
        
        for col in self.columns:
            # Create missingness indicators if requested
            if self.create_indicators:
                missingness_df = pd.DataFrame({f"{col}_missing": X_copy[col].isna().astype(int)})
                result_dfs.append(missingness_df)
            
            # Replace zeros with NaN if specified
            if self.zero_to_nan:
                X_copy[col] = X_copy[col].replace(0, np.nan)
            
            # Impute missing values
            X_copy[col] = X_copy[col].fillna(self.imputation_values_[col])
            
            # Log transform specific columns
            if col in self.log_transform_columns:
                X_copy[col] = np.log1p(X_copy[col])
            
            result_dfs.append(X_copy[[col]])
        
        return pd.concat(result_dfs, axis=1)


class YearTransformer(BaseEstimator, TransformerMixin):
    """
    Transform year features with multiple strategies.
    
    Parameters
    ----------
    reference_year : int, default=2000
        Year to center transformations around.
    
    normalization_factor : int, default=25
        Factor for normalizing year differences.
    
    transform_type : str, default='normalized'
        Type of year transformation. Options: 'centered', 'normalized', 'transformed'.
    """
    
    def __init__(
        self, 
        reference_year: int = 2000, 
        normalization_factor: int = 25,
        transform_type: str = 'normalized'
    ):
        self.reference_year = reference_year
        self.normalization_factor = normalization_factor
        self.transform_type = transform_type
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply year transformation."""
        X_copy = X.copy()
        
        # Compute year transformations
        X_copy["year_published_centered"] = X_copy["year_published"] - self.reference_year
        X_copy["year_published_normalized"] = (X_copy["year_published"] - self.reference_year) / self.normalization_factor
        X_copy["year_published_transformed"] = np.where(
            X_copy["year_published"] <= self.reference_year,
            np.log(np.maximum(self.reference_year - X_copy["year_published"] + 1, 1e-8)),
            np.log(np.maximum(X_copy["year_published"] - self.reference_year + 1, 1e-8))
        )
        
        # Select transformation type
        transform_map = {
            'centered': 'year_published_centered',
            'normalized': 'year_published_normalized',
            'transformed': 'year_published_transformed'
        }
        selected_column = transform_map.get(self.transform_type, 'year_published_normalized')
        
        return X_copy[[selected_column]]


class PlayerCountTransformer(BaseEstimator, TransformerMixin):
    """
    Create player count dummy variables.
    
    Parameters
    ----------
    max_player_count : int, default=10
        Maximum player count to generate dummy variables for.
    """
    
    def __init__(self, max_player_count: int = 10):
        self.max_player_count = max_player_count
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate player count dummy variables."""
        result = pd.DataFrame(index=X.index)
        
        for count in range(1, self.max_player_count + 1):
            result[f"player_count_{count}"] = (
                (X["min_players"] <= count) & 
                (X["max_players"] >= count)
            ).astype(int)
        
        return result


class ArrayFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Create one-hot encoded features for array/list columns.
    
    Parameters
    ----------
    column : str
        Name of the column containing array/list features.
    
    prefix : str
        Prefix for generated feature names.
    
    min_freq : int, default=10
        Minimum frequency for a value to be included as a feature.
    
    max_features : int, default=100
        Maximum number of features to generate.
    
    filter_func : Optional[Callable]
        Optional function to filter values before processing.
    """
    
    def __init__(
        self, 
        column: str, 
        prefix: str, 
        min_freq: int = 10, 
        max_features: int = 100,
        filter_func: Optional[Callable] = None
    ):
        self.column = column
        self.prefix = prefix
        self.min_freq = min_freq
        self.max_features = max_features
        self.filter_func = filter_func
        
        self.frequent_values_ = None
        self.feature_names_ = None
    
    def _safe_column_name(self, name: str) -> str:
        """Create a safe column name from a string."""
        return str(name).lower().replace(' ', '_').replace('-', '_').replace(':', '').replace('/', '_')
    
    def fit(self, X: pd.DataFrame, y=None):
        """Learn frequent values in the array column."""
        # Apply filter if provided
        X_filtered = X.copy()
        if self.filter_func:
            X_filtered[self.column] = X_filtered[self.column].apply(self.filter_func)
        
        # Collect all values
        all_values = []
        for values in X_filtered[self.column]:
            if isinstance(values, list):
                all_values.extend([v for v in values if v and v != ''])
            elif hasattr(values, '__array__'):
                values_list = values.tolist()
                all_values.extend([v for v in values_list if v and v != ''])
        
        # Count and filter values
        value_counts = pd.Series(all_values).value_counts()
        self.frequent_values_ = value_counts[value_counts >= self.min_freq].head(self.max_features).index.tolist()
        
        # Generate feature names
        self.feature_names_ = [f"{self.prefix}_{self._safe_column_name(val)}" for val in self.frequent_values_]
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create one-hot encoded features."""
        result = pd.DataFrame(index=X.index, columns=self.feature_names_)
        result.fillna(0, inplace=True)
        
        for idx, values in X[self.column].items():
            if isinstance(values, list):
                row_values = values
            elif hasattr(values, '__array__'):
                row_values = values.tolist()
            else:
                continue
            
            for value in row_values:
                if value in self.frequent_values_:
                    col_name = f"{self.prefix}_{self._safe_column_name(value)}"
                    result.loc[idx, col_name] = 1
        
        return result


def create_bgg_preprocessor(
    max_player_count: int = 10,
    reference_year: int = 2000,
    normalization_factor: int = 25,
    year_transform_type: str = 'normalized',
    create_missingness_features: bool = False,
    log_transform_columns: Optional[List[str]] = None,
    array_feature_configs: Optional[List[dict]] = None
) -> ColumnTransformer:
    """
    Create a comprehensive preprocessor for BGG data.
    
    Parameters
    ----------
    max_player_count : int, default=10
        Maximum player count for dummy variables.
    
    reference_year : int, default=2000
        Reference year for year transformations.
    
    normalization_factor : int, default=25
        Normalization factor for year transformations.
    
    year_transform_type : str, default='normalized'
        Type of year transformation.
    
    create_missingness_features : bool, default=False
        Whether to create missingness indicator features.
    
    log_transform_columns : Optional[List[str]], optional
        Columns to apply log transformation.
    
    array_feature_configs : Optional[List[dict]], optional
        Configurations for array feature transformers.
    
    Returns
    -------
    ColumnTransformer
        A scikit-learn compatible preprocessor for BGG data.
    """
    # Default log transform columns
    log_transform_columns = log_transform_columns or ['min_age', 'min_playtime', 'max_playtime', 'users_rated']
    
    # Default array feature configurations
    default_array_configs = [
        {
            'column': 'categories',
            'prefix': 'category',
            'min_freq': 10,
            'max_features': 100
        },
        {
            'column': 'mechanics',
            'prefix': 'mechanic',
            'min_freq': 10,
            'max_features': 100
        },
        {
            'column': 'designers',
            'prefix': 'designer',
            'min_freq': 10,
            'max_features': 50
        },
        {
            'column': 'artists',
            'prefix': 'artist',
            'min_freq': 10,
            'max_features': 50
        },
        {
            'column': 'publishers',
            'prefix': 'publisher',
            'min_freq': 5,
            'max_features': 25,
            'filter_func': lambda x: [p for p in x if p in {
                "Hasbro", "Mayfair Games", "Days of Wonder", "Asmodee", 
                "Fantasy Flight Games", "Rio Grande Games", "Z-Man Games"
            }]
        }
    ]
    
    array_feature_configs = array_feature_configs or default_array_configs
    
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ('missing_handler', MissingValueHandler(
            create_indicators=create_missingness_features,
            log_transform_columns=log_transform_columns
        ))
    ])
    
    # Year pipeline
    year_pipeline = Pipeline([
        ('year_transformer', YearTransformer(
            reference_year=reference_year,
            normalization_factor=normalization_factor,
            transform_type=year_transform_type
        ))
    ])
    
    # Player count pipeline
    player_pipeline = Pipeline([
        ('player_transformer', PlayerCountTransformer(max_player_count=max_player_count))
    ])
    
    # Array feature pipelines
    array_transformers = [
        (
            f"{config['column']}_transformer", 
            ArrayFeatureTransformer(
                column=config['column'],
                prefix=config.get('prefix', config['column']),
                min_freq=config.get('min_freq', 10),
                max_features=config.get('max_features', 100),
                filter_func=config.get('filter_func')
            ),
            [config['column']]  # Add the column name as the third element
        ) for config in array_feature_configs
    ]
    
    # Combine all transformers
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, ['min_age', 'min_playtime', 'max_playtime', 'users_rated']),
        ('year', year_pipeline, ['year_published']),
        ('player', player_pipeline, ['min_players', 'max_players']),
        *array_transformers
    ])
    
    return preprocessor


# Example usage
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config
    from src.models.splitting import time_based_split
    
    # Load data
    config = load_config()
    loader = BGGDataLoader(config)
    full_data = loader.load_training_data(end_train_year=2022, min_ratings=0)
    
    # split data
    # Create train-validation-test split using time_split
    train_data, validation_data = time_based_split(full_data, 
                                                   train_end_year = 2022, 
                                                   prediction_window = 2)
    
    # convert to pandas
    train_pandas = train_data.to_pandas()
    train_y = train_pandas['hurdle']
    train_X = train_pandas.drop(columns=['hurdle'])
    
    # Create preprocessor
    preprocessor = create_bgg_preprocessor()
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
        #('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    pipeline.fit(train_X, train_y)
    
    # For demonstration purposes only - in a real scenario you would use validation data
    # Convert validation data to pandas
    validation_pandas = validation_data.to_pandas()
    validation_y = validation_pandas['hurdle']
    validation_X = validation_pandas.drop(columns=['hurdle'])
    
    # Make predictions
    y_pred = pipeline.predict(validation_X)
    
    # Print classification report
    print(classification_report(validation_y, y_pred))
