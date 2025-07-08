"""Train/Tune/Test Complexity Regression Model for Board Game Complexity Prediction"""
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Project imports
from src.models.experiments import ExperimentTracker

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, QuantileRegressor
from sklearn.base import clone
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
import lightgbm as lgb

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

# Project imports
from src.data.config import load_config
from src.data.loader import BGGDataLoader
from src.features.preprocessor import create_bgg_preprocessor
from src.models.splitting import time_based_split

def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the training process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logger.addHandler(file_handler)
    
    return logger

# Sample weights calculation removed
def constrain_predictions(predictions: np.ndarray) -> np.ndarray:
    """Constrain predictions to be between 1 and 5."""
    return np.clip(predictions, 1, 5)

def calculate_complexity_weights(
    complexities: np.ndarray, 
    base: float = 10.0,  # Increased base for more dramatic weighting
    min_weight: float = 1.0,
    max_weight: float = 100.0  # Added max weight to prevent extreme values
) -> np.ndarray:
    """
    Calculate exponential weights for complexity values.
    
    Args:
        complexities (array-like): Complexity values
        base (float): Base for exponential weighting. Higher values 
                      create more dramatic weight increases
        min_weight (float): Minimum weight to apply
        max_weight (float): Maximum weight to prevent extreme values
    
    Returns:
        numpy array of weights normalized to have mean 1.0
    """
    logger = logging.getLogger(__name__)
    
    # Log input complexity distribution
    logger.info("Complexity Weight Calculation Diagnostic:")
    logger.info(f"  Complexity Range: min={complexities.min():.2f}, max={complexities.max():.2f}")
    logger.info(f"  Complexity Mean: {complexities.mean():.2f}")
    logger.info(f"  Complexity Std Dev: {complexities.std():.2f}")
    
    # Normalize complexities to start from 0
    normalized_complexities = complexities - complexities.min()
    
    # Calculate exponential weights with more dramatic scaling
    weights = np.power(base, normalized_complexities)
    
    # Clip weights to prevent extreme values
    weights = np.clip(weights, min_weight, max_weight)
    
    # Normalize weights to have mean 1.0
    weights = weights / np.mean(weights)
    
    # Log weight distribution
    logger.info("  Weight Distribution:")
    logger.info(f"    Weight Range: min={weights.min():.2f}, max={weights.max():.2f}")
    logger.info(f"    Weight Mean: {weights.mean():.2f}")
    logger.info(f"    Weight Std Dev: {weights.std():.2f}")
    
    # Log a few example mappings
    logger.info("  Sample Complexity to Weight Mapping:")
    for complexity, weight in zip(complexities[:10], weights[:10]):
        logger.info(f"    Complexity {complexity:.2f} -> Weight {weight:.2f}")
    
    return weights

def select_X_y(df, y_column, to_pandas = True):
    """
    Extract features (X) and target (y) from a dataframe.
    
    Args:
        df: Polars DataFrame
        y_column: Name of the target column
        to_pandas: Whether to convert to pandas DataFrame/Series
        
    Returns:
        X, y as either Polars or Pandas objects
        
    Raises:
        ValueError: If y_column is not in the dataframe
    """
    if y_column not in df.columns:
        raise ValueError(f"Target column '{y_column}' not found in dataframe. Available columns: {df.columns}")
    
    X = df.drop(y_column)
    y = df.select(y_column)
    
    if to_pandas:
        X = X.to_pandas()
        y = y.to_pandas().squeeze()
        return X, y
    else:
        return X, y

def create_preprocessing_pipeline() -> Pipeline:
    """Create preprocessing pipeline with feature engineering, imputation, and scaling."""
    # Create preprocessing pipeline using the standard BGG preprocessor
    pipeline = create_bgg_preprocessor(
        reference_year=2000,
        normalization_factor=25,
        log_columns=['min_age', 'min_playtime', 'max_playtime']
    )
    
    # Update the BGG preprocessor with custom parameters
    bgg_preprocessor = pipeline.named_steps['bgg_preprocessor']
    bgg_preprocessor.verbose = False
    bgg_preprocessor.category_min_freq = 0
    bgg_preprocessor.mechanic_min_freq = 0
    bgg_preprocessor.designer_min_freq = 10
    bgg_preprocessor.artist_min_freq = 10
    bgg_preprocessor.publisher_min_freq = 5
    bgg_preprocessor.family_min_freq = 10
    bgg_preprocessor.max_artist_features = 500
    bgg_preprocessor.max_publisher_features = 250
    bgg_preprocessor.max_designer_features = 500
    bgg_preprocessor.max_family_features = 500
    bgg_preprocessor.max_mechanic_features = 500
    bgg_preprocessor.max_category_features = 500
    bgg_preprocessor.create_category_features = True
    bgg_preprocessor.create_mechanic_features = True
    bgg_preprocessor.create_designer_features = True,
    bgg_preprocessor.create_artist_features = True,
    bgg_preprocessor.create_publisher_features = True,
    bgg_preprocessor.create_family_features = True,
    bgg_preprocessor.create_player_dummies = True
    bgg_preprocessor.include_base_numeric = True
    
    return pipeline

def preprocess_data(
    df: pl.DataFrame,
    preprocessing_pipeline: Pipeline,
    fit: bool = False,
    dataset_name: str = "data"
) -> pd.DataFrame:
    """Preprocess data for model training."""
    # Convert array columns to lists before converting to pandas
    array_columns = ['categories', 'mechanics', 'designers', 'artists', 'publishers', 'families']
    df_converted = df.clone()
    
    for col in array_columns:
        if col in df.columns:
            # Convert polars array to python list with explicit return type
            df_converted = df_converted.with_columns(
                pl.col(col).map_elements(lambda x: x.to_list() if x is not None else [], return_dtype=pl.List(pl.Utf8))
            )
    
    # Convert to pandas
    df_pandas = df_converted.to_pandas()
    
    # Setup logging
    logger = logging.getLogger(__name__)
    logger.info(f"Preprocessing {dataset_name} set: {len(df_pandas)} rows")
    
    # Process with pipeline
    if fit:
        features = preprocessing_pipeline.fit_transform(df_pandas)
        logger.info(f"  Fitted preprocessing pipeline on {dataset_name} data")
    else:
        features = preprocessing_pipeline.transform(df_pandas)
        logger.info(f"  Applied preprocessing pipeline to {dataset_name} data")
    
    logger.info(f"  {dataset_name.title()} features shape: {features.shape}")
    
    # Get feature names from BGG preprocessor
    try:
        feature_names = preprocessing_pipeline.named_steps['bgg_preprocessor'].get_feature_names_out()
    except Exception as e:
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
    
    # Convert to DataFrame
    features = pd.DataFrame(features, columns=feature_names)
    return features

def configure_model(model_name: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Set up regression model and parameter grid."""
    model_MAPPING = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'lightgbm': lgb.LGBMRegressor,
        'quantile': QuantileRegressor
    }
    
    PARAM_GRIDS = {
        'linear': {},  # Linear Regression has no hyperparameters to tune
        'ridge': {
            'model__alpha': [0.0001, 0.0005, 0.01, 0.1, 1.0, 5],  # Expanded alpha range
            'model__solver': ['auto'],
            'model__fit_intercept': [True]
        },
        'lasso': {
            'model__alpha': [0.1, 1.0, 10.0],
            'model__selection': ['cyclic', 'random']
        },
        'lightgbm': {
            'model__n_estimators': [500],
            'model__learning_rate': [0.01],
            'model__max_depth': [-1],  # -1 means no limit
            'model__num_leaves': [50, 100],
            'model__min_child_samples': [10],
            'model__reg_alpha': [0.1],
        },
        'quantile': {
            'model__quantile': [0.4, 0.5, 0.6],  # Different quantiles to explore
            'model__solver': ['highs-ds'],
            'model__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0],  # Regularization strength
            'model__fit_intercept': [True]
        }
    }
    
    model = model_MAPPING[model_name]()
    param_grid = PARAM_GRIDS[model_name]
    
    return model, param_grid

def plot_learning_curve(
    pipeline: Pipeline,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    tune_X: pd.DataFrame,
    tune_y: pd.Series,
    metric: str = 'rmse',
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10)
) -> Tuple[plt.Figure, Dict[str, np.ndarray]]:
    """
    Generate learning curve plot showing model performance vs training set size.
    
    Args:
        pipeline: The pipeline to evaluate
        train_X: Training features
        train_y: Training target
        tune_X: Tuning features
        tune_y: Tuning target
        metric: Metric to evaluate ('rmse', 'mae', 'r2', 'mape')
        train_sizes: Array of training set size proportions
        
    Returns:
        Tuple of (figure, learning curve data)
    """
    from sklearn.base import clone
    import matplotlib.pyplot as plt
    
    # Setup scoring function
    scoring_functions = {
        'rmse': lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error,
        'r2': r2_score,
        'mape': mean_absolute_percentage_error
    }
    
    if metric not in scoring_functions:
        raise ValueError(f"Unsupported metric: {metric}")
    
    score_func = scoring_functions[metric]
    
    # Calculate absolute sizes
    n_samples = len(train_X)
    train_sizes_abs = np.round(train_sizes * n_samples).astype(int)
    
    # Initialize arrays to store scores
    train_scores = np.zeros(len(train_sizes))
    val_scores = np.zeros(len(train_sizes))
    
    # For each training size
    for idx, n_train in enumerate(train_sizes_abs):
        # Get subset of training data
        train_subset_X = train_X.iloc[:n_train]
        train_subset_y = train_y.iloc[:n_train]
        
        # Fit model
        current_pipeline = clone(pipeline)
        current_pipeline.fit(train_subset_X, train_subset_y)
        
        # Get predictions
        train_pred = constrain_predictions(current_pipeline.predict(train_subset_X))
        val_pred = constrain_predictions(current_pipeline.predict(tune_X))
        
        # Calculate scores
        train_scores[idx] = score_func(train_subset_y, train_pred)
        val_scores[idx] = score_func(tune_y, val_pred)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', color='blue')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', color='red')
    
    plt.xlabel('Training Set Size (proportion)', fontsize=12)
    plt.ylabel(f'{metric.upper()} Score', fontsize=12)
    plt.title('Learning Curve', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Return figure and data
    curve_data = {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'val_scores': val_scores
    }
    
    return plt.gcf(), curve_data

def tune_model(
    pipeline: Pipeline,
    train_X: pd.DataFrame, 
    train_y: pd.Series,
    tune_X: pd.DataFrame,
    tune_y: pd.Series,
    param_grid: Dict[str, Any],
    metric: str = 'rmse',
    patience: int = 15,
    min_delta: float = 1e-4,
    use_sample_weights: bool = False
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Tune hyperparameters using separate tuning set.
    
    Args:
        pipeline: The pipeline to tune
        train_X: Training features
        train_y: Training target
        tune_X: Tuning features
        tune_y: Tuning target
        param_grid: Grid of parameters to search
        metric: Metric to optimize ('rmse', 'mae', 'r2')
        patience: Number of iterations without improvement before early stopping
        min_delta: Minimum change in score to be considered an improvement
        
    Returns:
        Tuple of tuned pipeline and best parameters
    """
    from sklearn.base import clone
    import gc
    
    logger = logging.getLogger(__name__)
    
    # Get current model instance from pipeline
    current_model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Fit and transform the data once with the preprocessor
    logger.info("Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(train_X)
    X_tune_transformed = preprocessor.transform(tune_X)
    logger.info(f"Transformed features shape: Train {X_train_transformed.shape}, Tune {X_tune_transformed.shape}")
    
    # Validate param_grid
    if param_grid is None:
        param_grid = {}
    
    # Calculate sample weights for training data
    if use_sample_weights:
        sample_weights = calculate_complexity_weights(train_y.values)
    
    # Scoring functions for regression
    scoring_functions = {
        'rmse': lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error,
        'r2': r2_score,
        'mape': mean_absolute_percentage_error
    }
    
    if metric not in scoring_functions:
        raise ValueError(f"Unsupported metric: {metric}. Choose from {list(scoring_functions.keys())}")
    
    # For minimization-based metrics, negate the score
    if metric in ['rmse', 'mae', 'mape']:
        score_func = lambda y, y_pred: -scoring_functions[metric](y, y_pred)
    else:
        score_func = scoring_functions[metric]
    
    best_score = np.inf  # Initialize to infinity since we want to minimize
    best_params = None
    best_model = None
    tuning_results = []
    patience_counter = 0
    
    try:
        # Get total number of parameter combinations for logging
        param_combinations = list(ParameterGrid(param_grid)) if param_grid else [{}]
        n_combinations = len(param_combinations)
        logger.info(f"Testing {n_combinations} parameter combinations")
        
        # Use tqdm for progress tracking
        for i, params in enumerate(tqdm(param_combinations, desc="Tuning hyperparameters")):
            logger.info(f"Evaluating combination {i+1}/{n_combinations}: {params}")
            
            try:
                # Create a fresh copy of the model
                current_model = clone(pipeline.named_steps['model'])
                
                # Set parameters if any
                if params:
                    current_model.set_params(**{k.replace('model__', ''): v for k, v in params.items()})
                
                # Fit with sample weights
                if use_sample_weights:
                    current_model.fit(X_train_transformed, train_y, sample_weight=sample_weights)
                else:
                    current_model.fit(X_train_transformed, train_y)

                # Predict on tuning set
                y_tune_pred = constrain_predictions(current_model.predict(X_tune_transformed))
                
                # Calculate score
                score = score_func(tune_y, y_tune_pred)
                
                # Store detailed results
                result = {
                    'params': params,
                    'score': score,
                    'metric': metric
                }
                tuning_results.append(result)
                
                logger.info(f"Params {params}: {metric} = {score:.4f}")
                
                # Check if this is the best score
                if score < best_score - min_delta:  # Improvement beyond threshold
                    best_score = score
                    best_params = params.copy()  # Make a copy to be safe
                    best_model = clone(current_model)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} iterations without improvement")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to train with params {params}: {str(e)}")
                continue
            finally:
                # Clean up to prevent memory leaks
                if 'current_model' in locals():
                    del current_model
                    gc.collect()
    
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        if best_params is None:
            raise RuntimeError("Hyperparameter tuning failed completely") from e
    
    # Log full tuning results
    logger.info("Hyperparameter Tuning Results:")
    for result in sorted(tuning_results, key=lambda x: x['score']):
        logger.info(f"  Params: {result['params']}, {metric}: {result['score']:.4f}")
    
    logger.info(f"Best params: {best_params} ({metric} = {best_score:.4f})")
    
    # Ensure we have valid parameters
    if best_params is None:
        best_params = {}
    
    # Create new pipeline with fitted preprocessor and best model
    tuned_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', best_model or current_model)
    ])
    
    return tuned_pipeline, best_params

def extract_model_coefficients(fitted_pipeline) -> pd.DataFrame:
    """
    Extract coefficients and feature names from a fitted pipeline.
    
    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted scikit-learn pipeline containing a preprocessor and model
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names, coefficients, and absolute coefficients,
        sorted by absolute coefficient value in descending order.
    """
    # Get the preprocessor and model from pipeline
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    model = fitted_pipeline.named_steps['model']
    
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
        except:
            # Last resort: check for feature_names_ attribute
            if hasattr(preprocessor, 'feature_names_'):
                feature_names = preprocessor.feature_names_
            else:
                raise ValueError("Could not get feature names from any preprocessing step")
    
    # Get coefficients from model
    if not hasattr(model, 'coef_'):
        raise ValueError("model does not have coefficients")
        
    coefficients = model.coef_
    
    # Validate lengths match
    if len(feature_names) != len(coefficients):
        raise ValueError(
            f"Mismatch between number of features ({len(feature_names)}) "
            f"and coefficients ({len(coefficients)})"
        )
    
    # Create DataFrame with coefficients
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
    
    # Add rank
    coef_df['rank'] = range(1, len(coef_df) + 1)
    
    return coef_df

def evaluate_model(
    model, 
    X: pd.DataFrame, 
    y: pd.Series,
    dataset_name: str = "test"
) -> Dict[str, float]:
    """Evaluate regression model performance."""
    # Diagnostic logging for raw predictions
    logger = logging.getLogger(__name__)
    
    # Get raw predictions before constraining
    raw_predictions = model.predict(X)
    logger.info(f"Raw Predictions Diagnostic - {dataset_name}:")
    logger.info(f"  Raw Prediction Range: min={raw_predictions.min():.4f}, max={raw_predictions.max():.4f}")
    logger.info(f"  Raw Prediction Mean: {raw_predictions.mean():.4f}")
    logger.info(f"  Raw Prediction Std Dev: {raw_predictions.std():.4f}")
    
    # Print first 10 raw predictions
    logger.info("  First 10 Raw Predictions:")
    for i, pred in enumerate(raw_predictions[:10], 1):
        logger.info(f"    Prediction {i}: {pred:.4f}")
    
    # Constrain predictions
    predictions = constrain_predictions(raw_predictions)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'mae': mean_absolute_error(y, predictions),
        'r2': r2_score(y, predictions),
        'mape': mean_absolute_percentage_error(y, predictions)
    }
    
    logger.info(f"{dataset_name.title()} Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def load_data(local_data_path: Optional[str] = None, min_weights = 0, end_train_year: Optional[int] = None) -> pl.DataFrame:
    """Load data from either local parquet or BigQuery."""
    logger = logging.getLogger(__name__)
    
    if local_data_path:
        try:
            df = pl.read_parquet(local_data_path)
            logger.info(f"Loaded local data from {local_data_path}: {len(df)} rows")
            
            required_columns = ['year_published', 'complexity', 'num_weights']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in data: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading local data from {local_data_path}: {e}")
            raise
    else:
        try:
            config = load_config()
            loader = BGGDataLoader(config)
            df = loader.load_training_data(
                end_train_year=end_train_year + 1, 
                min_weights=min_weights,
                min_ratings=0  # Filter for games with at least 10 weights
            )
            logger.info(f"Loaded data from BigQuery: {len(df)} total rows")
        except Exception as e:
            logger.error(f"Error loading data from BigQuery: {e}")
            logger.error("If BigQuery access is not configured, use --local-data to specify a local file")
            raise
    
    logger.info(f"Year range: {df['year_published'].min()} - {df['year_published'].max()}")
    return df

def create_data_splits(
    df: pl.DataFrame,
    args: argparse.Namespace
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create train/tune/test splits based on provided arguments."""
    logger = logging.getLogger(__name__)
    logger.info("Creating data splits...")
    
    validation_window = args.tune_end_year - args.tune_start_year + 1
    test_window = args.test_end_year - args.test_start_year + 1
    
    try:
        train_df, tune_df, test_df = time_based_split(
            df=df,
            train_end_year=args.train_end_year,
            prediction_window=validation_window,
            test_window=test_window,
            time_col="year_published",
            return_dict=False
        )
        
        if len(train_df) == 0:
            raise ValueError(f"Training set is empty. Check train_end_year={args.train_end_year}")
        if len(tune_df) == 0:
            raise ValueError(f"Tuning set is empty. Check tune years: {args.tune_start_year}-{args.tune_end_year}")
        if len(test_df) == 0:
            raise ValueError(f"Test set is empty. Check test years: {args.test_start_year}-{args.test_end_year}")
            
    except Exception as e:
        logger.error(f"Error creating data splits: {e}")
        raise
    
    logger.info(f"Training data: {len(train_df)} rows (years < {args.train_end_year})")
    logger.info(f"Tuning data: {len(tune_df)} rows (years {args.tune_start_year}-{args.tune_end_year})")
    logger.info(f"Test data: {len(test_df)} rows (years {args.test_start_year}-{args.test_end_year})")
    
    return train_df, tune_df, test_df

def log_experiment(
    experiment: ExperimentTracker,
    pipeline: Pipeline,
    train_metrics: Dict[str, float],
    tune_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_params: Dict[str, Any],
    args: argparse.Namespace,
    train_df: pl.DataFrame,
    tune_df: pl.DataFrame,
    test_df: pl.DataFrame,
    train_X: pd.DataFrame,
    tune_X: pd.DataFrame,
    test_X: pd.DataFrame,
    train_y: pd.Series,
    tune_y: pd.Series,
    test_y: pd.Series
) -> None:
    """Log all experiment results and artifacts."""
    # Extract preprocessing settings from pipeline
    bgg_preprocessor = pipeline.named_steps['preprocessor'].named_steps['bgg_preprocessor']
    
    # Create configuration dictionary
    config = {
        'model_params': best_params,
        'preprocessing': {
            'category_min_freq': bgg_preprocessor.category_min_freq,
            'mechanic_min_freq': bgg_preprocessor.mechanic_min_freq,
            'designer_min_freq': bgg_preprocessor.designer_min_freq,
            'artist_min_freq': bgg_preprocessor.artist_min_freq,
            'publisher_min_freq': bgg_preprocessor.publisher_min_freq,
            'family_min_freq': bgg_preprocessor.family_min_freq,
            'max_artist_features': bgg_preprocessor.max_artist_features,
            'max_publisher_features': bgg_preprocessor.max_publisher_features,
            'max_designer_features': bgg_preprocessor.max_designer_features,
            'max_family_features': bgg_preprocessor.max_family_features,
            'max_mechanic_features': bgg_preprocessor.max_mechanic_features,
            'max_category_features': bgg_preprocessor.max_category_features,
            'create_category_features': bgg_preprocessor.create_category_features,
            'create_mechanic_features': bgg_preprocessor.create_mechanic_features,
            'create_designer_features': bgg_preprocessor.create_designer_features,
            'create_artist_features': bgg_preprocessor.create_artist_features,
            'create_publisher_features': bgg_preprocessor.create_publisher_features,
            'create_family_features': bgg_preprocessor.create_family_features,
            'create_player_dummies': bgg_preprocessor.create_player_dummies,
            'include_base_numeric': bgg_preprocessor.include_base_numeric
        },
        'data_splits': {
            'train_end_year': args.train_end_year,
            'tune_start_year': args.tune_start_year,
            'tune_end_year': args.tune_end_year,
            'test_start_year': args.test_start_year,
            'test_end_year': args.test_end_year
        },
        'target': 'complexity',
        'target_type': 'regression'
    }
    
    # No need to recreate the experiment, use the passed experiment directly
    
    # Log metrics
    experiment.log_metrics(train_metrics, "train")
    experiment.log_metrics(tune_metrics, "tune")
    experiment.log_metrics(test_metrics, "test")
    experiment.log_parameters(best_params)
    
    # Calculate and log sample weights for training and combined datasets
    logger = logging.getLogger(__name__)
    try:
        # Calculate sample weights for training data
        train_sample_weights = calculate_complexity_weights(train_y.values)
        
        # Calculate sample weights for combined data
        y_combined = pd.concat([train_y, tune_y])
        combined_sample_weights = calculate_complexity_weights(y_combined.values)
        
        # Log sample weights to experiment metadata
        experiment.log_metadata({
            'sample_weights': {
                'train': train_sample_weights.tolist(),
                'combined': combined_sample_weights.tolist()
            }
        })
        
        logger.info("Sample Weights Logged to Experiment Metadata:")
        logger.info(f"  Train Weights - Range: {train_sample_weights.min():.2f} to {train_sample_weights.max():.2f}")
        logger.info(f"  Combined Weights - Range: {combined_sample_weights.min():.2f} to {combined_sample_weights.max():.2f}")
        
    except Exception as e:
        logger.error(f"Error logging sample weights: {e}")
    
    # Extract and save feature importance
    logger = logging.getLogger(__name__)
    try:
        # Extract feature importance
        importance_df = extract_model_coefficients(pipeline)
        importance_pl = pl.from_pandas(importance_df)
        experiment.log_coefficients(importance_pl)
        
        # Log top features
        logger.info("Top 10 most important features (by absolute coefficient):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['rank']:2d}. {row['feature']:30s} = {row['coefficient']:8.4f}")
        
        # Save model info
        model_info = {
            'n_features': len(importance_df),
            'best_params': best_params,
            'intercept': float(pipeline.named_steps['model'].intercept_)
        }
        
        experiment.log_model_info(model_info)
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        logger.error("Continuing without saving feature importance")
    
    # Save pipeline
    experiment.save_pipeline(pipeline)
    
    # Create artifacts directory
    artifacts_dir = Path(experiment.exp_dir) / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment model separately for easier loading
    import joblib
    experiment_model_path = Path(experiment.exp_dir) / 'experiment_model.pkl'
    try:
        joblib.dump(pipeline, experiment_model_path)
        logger.info(f"Experiment model saved to {experiment_model_path}")
    except Exception as e:
        logger.error(f"Error saving experiment model: {e}")
    
    # Save predictions and scatter plots for both validation and test sets
    datasets = [
        ('validation', tune_df, tune_X, tune_y),
        ('test', test_df, test_X, test_y)
    ]
    
    for dataset_name, df, X, y in datasets:
        try:
            # Predict on dataset
            predictions = constrain_predictions(pipeline.predict(X))
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'game_id': df.select('game_id').to_pandas().squeeze(),
                'name': df.select('name').to_pandas().squeeze(),
                'year_published': df.select('year_published').to_pandas().squeeze(),
                'true_complexity': y,
                'predicted_complexity': predictions
            })
            
            # Convert to Polars and save
            predictions_pl = pl.from_pandas(predictions_df)
            predictions_path = artifacts_dir / f"{dataset_name}_predictions.parquet"
            predictions_pl.write_parquet(predictions_path)
            
            logger.info(f"{dataset_name.title()} set predictions saved to {predictions_path}")
            
            # Create scatter plot
            import matplotlib.pyplot as plt
            import numpy as np
            from sklearn.metrics import r2_score, mean_squared_error
            
            plt.figure(figsize=(12, 8))
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Scatter plot with color gradient based on year
            scatter = plt.scatter(
                predictions_df['true_complexity'], 
                predictions_df['predicted_complexity'], 
                c=predictions_df['year_published'], 
                cmap='viridis', 
                alpha=0.7,
                edgecolors='black', 
                linewidth=0.5
            )
            plt.colorbar(scatter, label='Year Published')
            
            # Perfect prediction line
            min_val = min(predictions_df['true_complexity'].min(), predictions_df['predicted_complexity'].min())
            max_val = max(predictions_df['true_complexity'].max(), predictions_df['predicted_complexity'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            # Calculate R² and RMSE
            r2 = r2_score(predictions_df['true_complexity'], predictions_df['predicted_complexity'])
            rmse = np.sqrt(mean_squared_error(predictions_df['true_complexity'], predictions_df['predicted_complexity']))
            
            # Annotations
            plt.title(f'{dataset_name.title()} Complexity Predictions\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=14)
            plt.xlabel('Actual Complexity', fontsize=12)
            plt.ylabel('Predicted Complexity', fontsize=12)
            plt.legend()
            
            # Tight layout and save
            plt.tight_layout()
            scatter_plot_path = artifacts_dir / f"{dataset_name}_predictions_scatter.png"
            plt.savefig(scatter_plot_path, dpi=300)
            plt.close()
            
            logger.info(f"{dataset_name.title()} scatter plot saved to {scatter_plot_path}")
            
        except Exception as e:
            logger.error(f"Error saving {dataset_name} set predictions and scatter plot: {e}")

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Train/Tune/Test Complexity Regression Model")
    parser.add_argument("--train-end-year", type=int, default=2022, 
                       help="End year for training (exclusive)")
    parser.add_argument("--tune-start-year", type=int, default=2022,
                       help="Start year for tuning (inclusive)")
    parser.add_argument("--tune-end-year", type=int, default=2023,
                       help="End year for tuning (inclusive)")
    parser.add_argument("--test-start-year", type=int, default=2024,
                       help="Start year for testing (inclusive)")
    parser.add_argument("--test-end-year", type=int, default=2025,
                       help="End year for testing (inclusive)")
    parser.add_argument("--min-weights", type=int, default = 10)
    parser.add_argument("--output-dir", type=str, default="./models/experiments")
    parser.add_argument("--experiment", type=str, 
                       default="complexity_regression")
    parser.add_argument("--description", type=str,
                       help="Description of the experiment")
    parser.add_argument("--local-data", type=str,
                       help="Path to local parquet file for training data")
    parser.add_argument("--model", type=str, default="ridge",
                       choices=['linear', 'ridge', 'lasso', 'lightgbm', 'quantile'],
                       help="Regression model type to use")
    parser.add_argument("--quantile", type=float, default=0.5,
                       help="Quantile to use when model is 'quantile' (default: 0.5, median)")
    parser.add_argument("--metric", type=str, default="rmse",
                       choices=["rmse", "mae", "r2", "mape"],
                       help="Metric to optimize during hyperparameter tuning")
    parser.add_argument("--patience", type=int, default=5,
                       help="Number of iterations without improvement before early stopping")
    parser.add_argument("--use-sample-weights", action="store_true",
                       help="Enable sample weights based on complexity values")
    
    args = parser.parse_args()
    
    # Validate year ranges
    if args.tune_start_year != args.train_end_year:
        raise ValueError(f"tune_start_year ({args.tune_start_year}) must equal train_end_year ({args.train_end_year})")
    
    if not (args.tune_start_year <= args.tune_end_year < args.test_start_year <= args.test_end_year):
        raise ValueError("Invalid year ranges. Must satisfy: tune_start <= tune_end < test_start <= test_end")
    
    # Validate quantile argument
    if args.model == 'quantile' and (args.quantile < 0 or args.quantile > 1):
        raise ValueError("Quantile must be between 0 and 1")
    
    return args

def main():
    """Main script for training, tuning, and testing a complexity regression model."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging()
    
    # Load and split data
    # load full data
    df = load_data(local_data_path=args.local_data, min_weights = args.min_weights, end_train_year = args.test_end_year)
    
    # filtered for training/evaluation
    logger.info(f"Training on games with at least {args.min_weights} weights")    
    train_df, tune_df, test_df = create_data_splits(df, args)
    
    # Get X, y splits
    train_X, train_y = select_X_y(train_df, y_column='complexity')
    tune_X, tune_y = select_X_y(tune_df, y_column='complexity')
    test_X, test_y = select_X_y(test_df, y_column='complexity')
    
    # Setup model and pipeline
    model, param_grid = configure_model(args.model)
    preprocessor = create_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"model: {model.__class__.__name__}")
    logger.info(f"Parameter Grid: {param_grid}")
    logger.info(f"Optimization metric: {args.metric}")
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")
    
    # Calculate sample weights for training data
    train_sample_weights = None
    if args.use_sample_weights:
        train_sample_weights = calculate_complexity_weights(train_y.values)
        logger.info("Training Sample Weights Diagnostic:")
        logger.info(f"  Weight Range: min={train_sample_weights.min():.2f}, max={train_sample_weights.max():.2f}")
        logger.info(f"  Weight Mean: {train_sample_weights.mean():.2f}")
    
    # Tune model
    tuned_pipeline, best_params = tune_model(
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        tune_X=tune_X,
        tune_y=tune_y,
        param_grid=param_grid,
        metric=args.metric,
        patience=args.patience,
        use_sample_weights=args.use_sample_weights
    )
    
    # Fit on train data with optional sample weights
    if args.use_sample_weights:
        train_pipeline = clone(tuned_pipeline).fit(train_X, train_y, model__sample_weight=train_sample_weights)
    else:
        train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")
    
    # Evaluate tuning set
    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")
    
    # Fit final model on combined train+tune data
    logger.info("Fitting final model on combined training + validation data...")
    X_combined = pd.concat([train_X, tune_X])
    y_combined = pd.concat([train_y, tune_y])
    
    # Calculate sample weights for combined data
    combined_sample_weights = None
    if args.use_sample_weights:
        combined_sample_weights = calculate_complexity_weights(y_combined.values)
        logger.info("Combined Sample Weights Diagnostic:")
        logger.info(f"  Weight Range: min={combined_sample_weights.min():.2f}, max={combined_sample_weights.max():.2f}")
        logger.info(f"  Weight Mean: {combined_sample_weights.mean():.2f}")
    
    # Fit final model with optional sample weights
    if args.use_sample_weights:
        final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined, model__sample_weight=combined_sample_weights)
    else:
        final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)
    
    # Evaluate on test set (filtered)
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")

    # Log experiment results
    tracker = ExperimentTracker("complexity", args.output_dir)
    
    # Create experiment
    experiment = tracker.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata={
            'train_end_year_exclusive': args.train_end_year,
            'tune_start_year': args.tune_start_year,
            'tune_end_year': args.tune_end_year,
            'test_start_year': args.test_start_year,
            'test_end_year': args.test_end_year,
            'model_type': 'complexity_regression',
            'target': 'complexity',
            'min_weights': args.min_weights,
        },
        config={
            'model_params': best_params,
            'preprocessing': {
                'category_min_freq': 0,
                'mechanic_min_freq': 0,
                'designer_min_freq': 10,
                'artist_min_freq': 10,
                'publisher_min_freq': 5,
                'family_min_freq': 10,
                'max_artist_features': 500,
                'max_publisher_features': 250,
                'max_designer_features': 500,
                'max_family_features': 500,
                'max_mechanic_features': 500,
                'max_category_features': 500,
                'create_category_features': True,
                'create_mechanic_features': True,
                'create_designer_features': True,
                'create_artist_features': True,
                'create_publisher_features': True,
                'create_family_features': True,
                'create_player_dummies': True,
                'include_base_numeric': True
            },
            'data_splits': {
                'train_end_year': args.train_end_year,
                'tune_start_year': args.tune_start_year,
                'tune_end_year': args.tune_end_year,
                'test_start_year': args.test_start_year,
                'test_end_year': args.test_end_year
            },
            'target': 'complexity',
            'target_type': 'regression'
        }
    )
    
    # Generate and save learning curve
    logger.info("Generating learning curve...")
    try:
        fig, curve_data = plot_learning_curve(
            pipeline=final_pipeline,
            train_X=train_X,
            train_y=train_y,
            tune_X=tune_X,
            tune_y=tune_y,
            metric=args.metric
        )
        learning_curve_path = Path(experiment.exp_dir) / "learning_curve.png"
        fig.savefig(learning_curve_path, dpi=300)
        plt.close()
        logger.info(f"Learning curve saved to {learning_curve_path}")
    except Exception as e:
        logger.error(f"Error generating learning curve: {e}")
    
    log_experiment(
        experiment=experiment,
        pipeline=final_pipeline,
        train_metrics=train_metrics,
        tune_metrics=tune_metrics,
        test_metrics=test_metrics,
        train_df=train_df,
        tune_df=tune_df,
        test_df=test_df,
        train_X=train_X,
        tune_X=tune_X,
        test_X=test_X,
        train_y=train_y,
        tune_y=tune_y,
        test_y=test_y,
        best_params=best_params,
        args=args
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
