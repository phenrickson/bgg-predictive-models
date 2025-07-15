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
    model_type: str = 'linear', 
    reference_year: int = 2000, 
    normalization_factor: int = 25,
    log_columns: list = ['min_age', 'min_playtime', 'max_playtime', 'time_per_player', 'description_word_count'],
    **kwargs
) -> Pipeline:
    """
    Create a preprocessing pipeline for board game data.
    
    Parameters
    ----------
    model_type : str, optional (default='linear')
        Type of model to preprocess for. 
        Options:
        - 'linear': Full preprocessing with scaling and transformations
        - 'tree': Minimal preprocessing suitable for tree-based models
    
    reference_year : int, optional (default=2000)
        Reference year for year transformations.
    
    normalization_factor : int, optional (default=25)
        Normalization factor for year transformations.
    
    log_columns : list, optional
        Columns to apply log transformation to. 
        Defaults to ['min_age', 'min_playtime', 'max_playtime'].
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        A preprocessing pipeline for board game data.
    
    Raises
    ------
    ValueError
        If an unsupported model_type is provided.
    """
    # Validate model_type
    if model_type not in ['linear', 'tree']:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'linear' or 'tree'.")
    
    # Create BGG preprocessor with kwargs
    bgg_preprocessor = BaseBGGTransformer(**kwargs)
    
    # Define pipeline steps based on model type
    pipeline_steps = [
        ('bgg_preprocessor', bgg_preprocessor),
        ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=False))
    ]
    
    # Add additional steps for linear models
    if model_type == 'linear':
        pipeline_steps.extend([
            ('log', LogTransformer(columns=log_columns)),
            ('year', YearTransformer(
                reference_year=reference_year, 
                normalization_factor=normalization_factor
            )),
            ('variance_selector', VarianceThreshold(threshold=0)),
            ('scaler', StandardScaler())
        ])
    elif model_type == 'tree':
        # Minimal preprocessing for tree-based models
        pipeline_steps.extend([
            ('variance_selector', VarianceThreshold(threshold=0))
        ])
    
    pipeline = Pipeline(pipeline_steps)
    pipeline.set_output(transform="pandas")

    return pipeline
