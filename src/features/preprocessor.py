import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from .transformers import (
    LogTransformer, 
    YearTransformer, 
    BaseBGGTransformer
)

def create_bgg_preprocessor(
    bgg_preprocessor = BaseBGGTransformer(),
    reference_year: int = 2000, 
    normalization_factor: int = 25,
    log_columns: list = ['min_age', 'min_playtime', 'max_playtime']
) -> Pipeline:
    """
    Create a standard preprocessing pipeline for board game data.
    
    Parameters
    ----------
    reference_year : int, optional (default=2000)
        Reference year for year transformations.
    
    normalization_factor : int, optional (default=25)
        Normalization factor for year transformations.
    
    log_columns : list, optional
        Columns to apply log transformation to. 
        Defaults to ['min_age', 'min_playtime', 'max_playtime'].
    
    numeric_columns : list, optional
        Numeric columns to scale. 
        Defaults to ['min_age', 'min_playtime', 'max_playtime'].
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        A preprocessing pipeline for board game data.
    """
    pipeline = Pipeline([
        ('bgg_preprocessor', bgg_preprocessor),
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=False)),
        ('log', LogTransformer(columns=log_columns)),
        ('year', YearTransformer(
            reference_year=reference_year, 
            normalization_factor=normalization_factor
        )),
        ('variance_selector', VarianceThreshold(threshold=0)),
        ('scaler', StandardScaler())
    ])
    pipeline.set_output(transform="pandas")

    return pipeline