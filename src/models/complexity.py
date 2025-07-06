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
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.base import BaseEstimator, clone
from tqdm import tqdm
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

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
    REGRESSOR_MAPPING = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso
    }
    
    PARAM_GRIDS = {
        'linear': {},  # Linear Regression has no hyperparameters to tune
        'ridge': {
            'regressor__alpha': [0.0001, 0.0005, 0.01, 0.1, 1.0, 5],  # Expanded alpha range
            'regressor__solver': ['auto'],
            'regressor__fit_intercept': [True]
        },
        'lasso': {
            'regressor__alpha': [0.1, 1.0, 10.0],
            'regressor__selection': ['cyclic', 'random']
        }
    }
    
    regressor = REGRESSOR_MAPPING[model_name]()
    param_grid = PARAM_GRIDS[model_name]
    
    return regressor, param_grid

def tune_model(
    pipeline: Pipeline,
    train_X: pd.DataFrame, 
    train_y: pd.Series,
    tune_X: pd.DataFrame,
    tune_y: pd.Series,
    param_grid: Dict[str, Any],
    metric: str = 'rmse',
    patience: int = 5,
    min_delta: float = 1e-4
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
    
    # Get current regressor instance from pipeline
    current_regressor = pipeline.named_steps['regressor']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Fit and transform the data once with the preprocessor
    logger.info("Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(train_X)
    X_tune_transformed = preprocessor.transform(tune_X)
    logger.info(f"Transformed features shape: Train {X_train_transformed.shape}, Tune {X_tune_transformed.shape}")
    
    # Validate param_grid
    if param_grid is None:
        param_grid = {}
    
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
                # Create a fresh copy of the regressor
                current_regressor = clone(pipeline.named_steps['regressor'])
                
                # Set parameters if any
                if params:
                    current_regressor.set_params(**{k.replace('regressor__', ''): v for k, v in params.items()})
                
                # Train regressor with these parameters on transformed data
                current_regressor.fit(X_train_transformed, train_y)
                
                # Predict on tuning set
                y_tune_pred = constrain_predictions(current_regressor.predict(X_tune_transformed))
                
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
                    best_model = clone(current_regressor)
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
                if 'current_regressor' in locals():
                    del current_regressor
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
        ('regressor', best_model or current_regressor)
    ])
    
    return tuned_pipeline, best_params

def extract_model_coefficients(fitted_pipeline) -> pd.DataFrame:
    """
    Extract coefficients and feature names from a fitted pipeline.
    
    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted scikit-learn pipeline containing a preprocessor and regressor
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names, coefficients, and absolute coefficients,
        sorted by absolute coefficient value in descending order.
    """
    # Get the preprocessor and regressor from pipeline
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    regressor = fitted_pipeline.named_steps['regressor']
    
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
    
    # Get coefficients from regressor
    if not hasattr(regressor, 'coef_'):
        raise ValueError("Regressor does not have coefficients")
        
    coefficients = regressor.coef_
    
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
    predictions = constrain_predictions(model.predict(X))
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'mae': mean_absolute_error(y, predictions),
        'r2': r2_score(y, predictions),
        'mape': mean_absolute_percentage_error(y, predictions)
    }
    
    logger = logging.getLogger(__name__)
    logger.info(f"{dataset_name.title()} Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def load_data(local_data_path: Optional[str] = None, end_train_year: Optional[int] = None) -> pl.DataFrame:
    """Load data from either local parquet or BigQuery."""
    logger = logging.getLogger(__name__)
    
    if local_data_path:
        try:
            df = pl.read_parquet(local_data_path)
            df = df.filter(pl.col('users_rated') >=25)
            df = df.filter(pl.col('num_weights') >=10)
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
                min_ratings=0  # Filter for games with at least 10 weights
            ).filter(pl.col('num_weights')>=10)
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
    test_df: pl.DataFrame,
    test_X: pd.DataFrame,
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
    
    # Create experiment
    experiment = experiment.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata={
            'train_end_year_exclusive': args.train_end_year,
            'tune_start_year': args.tune_start_year,
            'tune_end_year': args.tune_end_year,
            'test_start_year': args.test_start_year,
            'test_end_year': args.test_end_year,
            'model_type': 'complexity_regression',
            'target': 'complexity'
        },
        config=config
    )
    
    # Log metrics
    experiment.log_metrics(train_metrics, "train")
    experiment.log_metrics(tune_metrics, "tune")
    experiment.log_metrics(test_metrics, "test")
    experiment.log_parameters(best_params)
    
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
            'intercept': float(pipeline.named_steps['regressor'].intercept_)
        }
        
        experiment.log_model_info(model_info)
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        logger.error("Continuing without saving feature importance")
    
    # Save pipeline
    experiment.save_pipeline(pipeline)
    
    # Save test set predictions
    try:
        # Predict on test set
        test_predictions = constrain_predictions(pipeline.predict(test_X))
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'game_id': test_df.select('game_id').to_pandas().squeeze(),
            'name': test_df.select('name').to_pandas().squeeze(),
            'year_published': test_df.select('year_published').to_pandas().squeeze(),
            'true_complexity': test_y,
            'predicted_complexity': test_predictions
        })
        
        # Convert to Polars and save
        predictions_pl = pl.from_pandas(predictions_df)
        predictions_path = f"data/predictions/{args.experiment}_complexity_predictions.parquet"
        predictions_pl.write_parquet(predictions_path)
        
        logger.info(f"Test set predictions saved to {predictions_path}")
        
    except Exception as e:
        logger.error(f"Error saving test set predictions: {e}")

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
    parser.add_argument("--output-dir", type=str, default="./models/experiments")
    parser.add_argument("--experiment", type=str, 
                       default="complexity_regression")
    parser.add_argument("--description", type=str,
                       help="Description of the experiment")
    parser.add_argument("--local-data", type=str,
                       help="Path to local parquet file for training data")
    parser.add_argument("--regressor", type=str, default="ridge",
                       choices=['linear', 'ridge', 'lasso'],
                       help="Regressor type to use")
    parser.add_argument("--metric", type=str, default="rmse",
                       choices=["rmse", "mae", "r2", "mape"],
                       help="Metric to optimize during hyperparameter tuning")
    parser.add_argument("--patience", type=int, default=5,
                       help="Number of iterations without improvement before early stopping")
    
    args = parser.parse_args()
    
    # Validate year ranges
    if args.tune_start_year != args.train_end_year:
        raise ValueError(f"tune_start_year ({args.tune_start_year}) must equal train_end_year ({args.train_end_year})")
    
    if not (args.tune_start_year <= args.tune_end_year < args.test_start_year <= args.test_end_year):
        raise ValueError("Invalid year ranges. Must satisfy: tune_start <= tune_end < test_start <= test_end")
    
    return args

def main():
    """Main script for training, tuning, and testing a complexity regression model."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging()
    
    # Load and split data
    df = load_data(args.local_data, args.test_end_year)
    train_df, tune_df, test_df = create_data_splits(df, args)
    
    # Get X, y splits
    train_X, train_y = select_X_y(train_df, y_column='complexity')
    tune_X, tune_y = select_X_y(tune_df, y_column='complexity')
    test_X, test_y = select_X_y(test_df, y_column='complexity')
    
    # Setup model and pipeline
    regressor, param_grid = configure_model(args.regressor)
    preprocessor = create_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"Regressor: {regressor.__class__.__name__}")
    logger.info(f"Parameter Grid: {param_grid}")
    logger.info(f"Optimization metric: {args.metric}")
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")
    
    # Tune model
    tuned_pipeline, best_params = tune_model(
        pipeline=pipeline,
        train_X=train_X,
        train_y=train_y,
        tune_X=tune_X,
        tune_y=tune_y,
        param_grid=param_grid,
        metric=args.metric,
        patience=args.patience
    )
    
    # Fit on train data and evaluate
    train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")
    
    # Evaluate tuning set
    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning")
    
    # Fit final model on combined train+tune data
    logger.info("Fitting final model on combined training + validation data...")
    X_combined = pd.concat([train_X, tune_X])
    y_combined = pd.concat([train_y, tune_y])
    
    final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)
    
    # Evaluate on test set
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test")

    # Log experiment results
    tracker = ExperimentTracker("complexity", args.output_dir)
    log_experiment(
        experiment=tracker,
        pipeline=final_pipeline,
        train_metrics=train_metrics,
        tune_metrics=tune_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        args=args
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
