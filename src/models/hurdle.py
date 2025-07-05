"""Train/Tune/Test Hurdle Model for Board Game Ratings Prediction"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Project imports
from src.models.experiments import ExperimentTracker

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    log_loss
)
from typing import Type, Union

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

def extract_model_coefficients(fitted_pipeline) -> pd.DataFrame:
    """
    Extract coefficients and feature names from a fitted pipeline.
    
    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted scikit-learn pipeline containing a preprocessor and classifier
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names, coefficients, and absolute coefficients,
        sorted by absolute coefficient value in descending order.
        
    Raises
    ------
    ValueError
        If pipeline is not fitted or doesn't contain required components
    """
    # Get the preprocessor and classifier from pipeline
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    classifier = fitted_pipeline.named_steps['classifier']
    
    # Find the last preprocessing step that has get_feature_names_out method
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
    
    # Get coefficients from classifier
    if not hasattr(classifier, 'coef_'):
        raise ValueError("Classifier does not have coefficients")
        
    coefficients = classifier.coef_[0]  # For binary classification
    
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

# function to select column and convert to pandas
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
    bgg_preprocessor.max_artist_features = 250
    bgg_preprocessor.max_publisher_features = 250
    bgg_preprocessor.max_designer_features = 250
    bgg_preprocessor.max_family_features = 250
    bgg_preprocessor.max_mechanic_features = 500
    bgg_preprocessor.max_category_features = 500
    bgg_preprocessor.create_category_features = True
    bgg_preprocessor.create_mechanic_features = True
    bgg_preprocessor.create_designer_features = True
    bgg_preprocessor.create_artist_features = True
    bgg_preprocessor.create_publisher_features = True
    bgg_preprocessor.create_family_features = True
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

def tune_hyperparameters(
    pipeline,
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    classifier_type: Type[BaseEstimator],
    param_grid: Optional[Dict[str, Any]] = None,
    metric: str = 'log_loss',
    patience: int = 5,
    min_delta: float = 1e-4
) -> Dict[str, Any]:
    """
    Tune hyperparameters using separate tuning set.
    
    Args:
        pipeline: The pipeline to tune
        X_train: Training features
        y_train: Training target
        X_tune: Tuning features
        y_tune: Tuning target
        classifier_type: Type of classifier being used
        param_grid: Grid of parameters to search
        metric: Metric to optimize ('log_loss', 'f1', 'auc')
        patience: Number of iterations without improvement before early stopping
        min_delta: Minimum change in score to be considered an improvement
        
    Returns:
        Dictionary with best parameters and score
    """
    from sklearn.base import clone
    import gc
    
    logger = logging.getLogger(__name__)
    
    # Get current classifier instance from pipeline
    current_classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Fit and transform the data once with the preprocessor
    logger.info("Fitting preprocessor and transforming data...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_tune_transformed = preprocessor.transform(X_tune)
    logger.info(f"Transformed features shape: Train {X_train_transformed.shape}, Tune {X_tune_transformed.shape}")
    
    # Default parameter grid if not provided
    if param_grid is None:
        if isinstance(current_classifier, LogisticRegression):
            param_grid = {
                'classifier__C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
                'classifier__penalty': ['l2'],  # Using only L2 with default solver
                'classifier__max_iter': [4000]
            }
        elif isinstance(current_classifier, RandomForestClassifier):
            param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]
            }
        else:
            # Generic grid for other classifiers
            param_grid = {}
            if hasattr(current_classifier, 'C'):
                param_grid['classifier__C'] = [0.001, 0.01, 0.1, 1.0, 10.0]
            if not param_grid:
                raise ValueError(f"No default parameter grid available for classifier type: {current_classifier.__class__.__name__}")
    
    # Validate param_grid
    if not param_grid:
        raise ValueError("Parameter grid cannot be empty")
    
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter tuning...")
    logger.info(f"Parameter grid: {param_grid}")
    
    # Define scoring function based on metric
    scoring_functions = {
        'log_loss': lambda y, y_pred_proba: log_loss(y, y_pred_proba),
        'f1': lambda y, y_pred: -f1_score(y, y_pred),  # Negative for minimization
        'auc': lambda y, y_pred_proba: -roc_auc_score(y, y_pred_proba[:, 1])
    }
    
    if metric not in scoring_functions:
        raise ValueError(f"Unsupported metric: {metric}. Choose from {list(scoring_functions.keys())}")
    
    score_func = scoring_functions[metric]
    
    best_score = np.inf  # Initialize to infinity since we want to minimize
    best_params = None
    best_model = None
    tuning_results = []
    patience_counter = 0
    
    try:
        # Get total number of parameter combinations for logging
        param_combinations = list(ParameterGrid(param_grid))
        n_combinations = len(param_combinations)
        logger.info(f"Testing {n_combinations} parameter combinations")
        
        # Use tqdm for progress tracking
        for i, params in enumerate(tqdm(param_combinations, desc="Tuning hyperparameters")):
            logger.info(f"Evaluating combination {i+1}/{n_combinations}: {params}")
            
            try:
                # Create a fresh copy of the classifier
                current_classifier = clone(pipeline.named_steps['classifier'])
                current_classifier.set_params(**{k.replace('classifier__', ''): v for k, v in params.items()})
                
                # Train classifier with these parameters on transformed data
                current_classifier.fit(X_train_transformed, y_train)
                
                # Evaluate on tuning set
                if metric == 'f1':
                    y_tune_pred = current_classifier.predict(X_tune_transformed)
                    score = score_func(y_tune, y_tune_pred)
                else:
                    y_tune_pred_proba = current_classifier.predict_proba(X_tune_transformed)
                    score = score_func(y_tune, y_tune_pred_proba)
                
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
                    best_model = clone(current_classifier)
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
                if 'current_classifier' in locals():
                    del current_classifier
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
        raise ValueError("Hyperparameter tuning failed to find any valid parameters. Check your data and parameter grid.")
    
    # Return best parameters, fitted preprocessor, best model and additional info
    return {
        'best_params': best_params,
        'best_score': best_score,
        'metric': metric,
        'n_combinations_tested': len(tuning_results),
        'fitted_preprocessor': preprocessor,
        'best_model': best_model
    }

def evaluate_model(
    model: LogisticRegression, 
    X: pd.DataFrame, 
    y: pd.Series,
    dataset_name: str = "test"
) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'log_loss': log_loss(y, model.predict_proba(X))
    }
    
    logger = logging.getLogger(__name__)
    logger.info(f"{dataset_name.title()} Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return metrics

def main():
    """
    Main script for training, tuning, and testing a hurdle model for board game ratings prediction.
    
    This script implements a complete machine learning pipeline for predicting whether board games
    will reach a certain threshold of user ratings (the "hurdle"). It includes:
    
    1. Data loading from BigQuery or local parquet file
    2. Time-based data splitting into train/tune/test sets
    3. Feature preprocessing with the BGG preprocessor
    4. Model training with hyperparameter tuning
    5. Model evaluation on test data
    6. Experiment tracking and result logging
    
    The script supports multiple classifier types and optimization metrics.
    
    Example usage:
        # Basic usage with default parameters
        python src/models/hurdle.py
        
        # Custom configuration
        python src/models/hurdle.py --classifier rf --metric f1 --train-end-year 2021 --tune-start-year 2021 --tune-end-year 2022 --test-start-year 2023 --test-end-year 2024
        
        # Use local data file
        python src/models/hurdle.py --local-data ./data/processed/games.parquet
    """
    import argparse
    
    # Define available classifiers
    CLASSIFIER_MAPPING = {
        'logistic': LogisticRegression,
        'rf': RandomForestClassifier,
        'svc': SVC
    }
    
    # Default parameter grids for different classifiers
    PARAM_GRIDS = {
        'logistic': {
            'classifier__C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
            'classifier__penalty': ['l2'],
            'classifier__max_iter': [4000]
        },
        'rf': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'svc': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01]
        }
    }
    
    parser = argparse.ArgumentParser(description="Train/Tune/Test Hurdle Model")
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
                       default="hurdle_model")
    parser.add_argument("--description", type=str,
                       help="Description of the experiment")
    parser.add_argument("--local-data", type=str,
                       help="Path to local parquet file for training data")
    parser.add_argument("--classifier", type=str, default="logistic",
                       choices=list(CLASSIFIER_MAPPING.keys()),
                       help="Classifier type to use")
    parser.add_argument("--metric", type=str, default="log_loss",
                       choices=["log_loss", "f1", "auc"],
                       help="Metric to optimize during hyperparameter tuning")
    parser.add_argument("--patience", type=int, default=5,
                       help="Number of iterations without improvement before early stopping")
    
    args = parser.parse_args()
    
    # Validate year ranges
    # Note: In time_based_split, tune_start_year should equal train_end_year
    if args.tune_start_year != args.train_end_year:
        raise ValueError(f"tune_start_year ({args.tune_start_year}) must equal train_end_year ({args.train_end_year})")
    
    # Validate the rest of the year ranges
    if not (args.tune_start_year <= args.tune_end_year < args.test_start_year <= args.test_end_year):
        raise ValueError("Invalid year ranges. Must satisfy: tune_start <= tune_end < test_start <= test_end")
    
    # Setup logging
    logger = setup_logging()
    
    # Initialize classifier and get parameter grid
    classifier = CLASSIFIER_MAPPING[args.classifier]()
    PARAM_GRID = PARAM_GRIDS[args.classifier]
    
    # Log experiment name
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"Classifier: {classifier.__class__.__name__}")
    logger.info(f"Parameter Grid: {PARAM_GRID}")
    logger.info(f"Optimization metric: {args.metric}")
    
    # Load data (load through the latest test year)
    if args.local_data:
        # Load from local parquet file
        try:
            df = pl.read_parquet(args.local_data)
            logger.info(f"Loaded local data from {args.local_data}: {len(df)} rows")
            
            # Verify required columns exist
            required_columns = ['year_published', 'hurdle']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in data: {missing_columns}")
                
        except Exception as e:
            logger.error(f"Error loading local data from {args.local_data}: {e}")
            raise
    else:
        # Load from BigQuery using existing loader
        try:
            config = load_config()
            loader = BGGDataLoader(config)
            df = loader.load_training_data(end_train_year=args.test_end_year + 1, min_ratings=0)
            logger.info(f"Loaded data from BigQuery: {len(df)} total rows")
        except Exception as e:
            logger.error(f"Error loading data from BigQuery: {e}")
            logger.error("If BigQuery access is not configured, use --local-data to specify a local file")
            raise
    
    logger.info(f"Year range: {df['year_published'].min()} - {df['year_published'].max()}")
    
    # Create data splits using time_based_split function
    logger.info("Creating data splits...")
    
    # Calculate windows for validation and test sets
    validation_window = args.tune_end_year - args.tune_start_year + 1
    test_window = args.test_end_year - args.test_start_year + 1
    
    try:
        # Use time_based_split to get all three splits
        train_df, tune_df, test_df = time_based_split(
            df=df,
            train_end_year=args.train_end_year,
            prediction_window=validation_window,
            test_window=test_window,
            time_col="year_published",
            return_dict=False
        )
        
        # Check if any split is empty
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
    
    # Preprocessing pipeline to be used with model
    preprocessor = create_preprocessing_pipeline()

    # Combine preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
        
    # Get X, y for train and validation set
    train_X, train_y = select_X_y(train_df, y_column = 'hurdle')
    tune_X, tune_y = select_X_y(tune_df, y_column = 'hurdle')
    
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")
    
    # Tune hyperparameters
    tuning_result = tune_hyperparameters(
        pipeline, 
        train_X, 
        train_y, 
        tune_X, 
        tune_y, 
        classifier.__class__, 
        PARAM_GRID,
        metric=args.metric,
        patience=args.patience
    )
    best_params = tuning_result['best_params']
    
    # Create new pipeline with fitted preprocessor and best model
    pipeline = Pipeline([
        ('preprocessor', tuning_result['fitted_preprocessor']),
        ('classifier', tuning_result['best_model'])
    ])
    
    # fit to training set only
    train_only_model = pipeline.fit(train_X, train_y)
    
    # Evaluate on training and tuning sets
    train_metrics = evaluate_model(train_only_model, train_X, train_y, "training")
    tune_metrics = evaluate_model(train_only_model, tune_X, tune_y, "tuning")
    
    # Get test data
    test_X, test_y = select_X_y(test_df, y_column='hurdle')

    # Refit on combined training + validation data for final evaluation
    logger.info("Refitting final model on training + validation data...")
    X_train_val = pd.concat([train_X, tune_X])
    y_train_val = pd.concat([train_y, tune_y])
    
    logger.info(f"Feature dimensions: Train + Tune {X_train_val.shape}")
    
    # Create final pipeline with preprocessor and final classifier
    final_classifier = clone(tuning_result['best_model'])
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', final_classifier)
    ])
    # Refit model on combined training + validation data for final evaluation
    final_pipeline.fit(X_train_val, y_train_val)
    
    # Evaluate final model on test set
    test_metrics = evaluate_model(final_pipeline, test_X, test_y, "test (using model fit on train+validation)")
    
    # Initialize experiment tracker and create new experiment
    tracker = ExperimentTracker("hurdle", args.output_dir)
    
    # Create configuration dictionary for hashing
    config = {
        'model_params': best_params,
        'preprocessing': {
            'reference_year': 2000,
            'normalization_factor': 25,
            'log_columns': ['min_age', 'min_playtime', 'max_playtime'],
            'category_min_freq': 0,
            'mechanic_min_freq': 0,
            'designer_min_freq': 10,
            'artist_min_freq': 10,
            'publisher_min_freq': 5,
            'family_min_freq': 5,
            'max_artist_features': 250,
            'max_publisher_features': 250,
            'max_designer_features': 250,
            'max_family_features': 250,
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
        'target': 'users_rated >= 25',
        'target_type': 'binary_classification'
    }
    
    experiment = tracker.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata={
            'train_end_year_exclusive': args.train_end_year,
            'tune_start_year': args.tune_start_year,
            'tune_end_year': args.tune_end_year,
            'test_start_year': args.test_start_year,
            'test_end_year': args.test_end_year,
            'model_type': 'hurdle',
            'target': 'users_rated >= 25'
        },
        config=config
    )
    
    # Log metrics for each dataset
    experiment.log_metrics(train_metrics, "train")
    experiment.log_metrics(tune_metrics, "tune")
    experiment.log_metrics(test_metrics, "test")
    
    # Log best parameters and tuning results
    experiment.log_parameters(best_params)
    experiment.log_metrics({'tuning_score': tuning_result['best_score']}, "tuning_result")
    
    # Extract and save model coefficients
    try:
        coefficients_df = extract_model_coefficients(final_pipeline)
        
        # Convert to polars for saving
        coefficients_pl = pl.from_pandas(coefficients_df)
        experiment.log_coefficients(coefficients_pl)
        
        # Log top 10 most important features
        logger.info("Top 10 most important features (by absolute coefficient):")
        for _, row in coefficients_df.head(10).iterrows():
            logger.info(f"  {row['rank']:2d}. {row['feature']:30s} = {row['coefficient']:8.4f}")
        
        # Save model info
        model_info = {
            'intercept': float(final_pipeline.named_steps['classifier'].intercept_[0]),
            'n_features': len(coefficients_df),
            'best_params': best_params
        }
        experiment.log_model_info(model_info)
        
    except Exception as e:
        logger.error(f"Error extracting model coefficients: {e}")
        logger.error("Continuing without saving coefficients")
    
    # Save complete pipeline
    experiment.save_pipeline(final_pipeline)
    
    logger.info(f"Results saved to {experiment.exp_dir}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
