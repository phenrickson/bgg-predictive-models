"""Train/Tune/Test Hurdle Model for Board Game Ratings Prediction"""
import logging
import argparse
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
    log_loss,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix
)

# CatBoost imports
from catboost import CatBoostClassifier
from typing import Type, Union, Tuple

def extract_feature_importance(
    fitted_pipeline: Pipeline, 
    classifier_type: Type[BaseEstimator]
) -> pd.DataFrame:
    """
    Extract feature importance for different classifier types.
    
    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted scikit-learn pipeline containing a preprocessor and classifier
    classifier_type : Type[BaseEstimator]
        The type of classifier to extract feature importance for
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names, importance values, 
        and absolute importance, sorted in descending order.
    """
    # Get the preprocessor and classifier from pipeline
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    classifier = fitted_pipeline.named_steps['classifier']
    
    # Find feature names
    steps = list(preprocessor.named_steps.items())
    feature_names = None
    
    for name, step in reversed(steps):
        try:
            feature_names = step.get_feature_names_out()
            break
        except (AttributeError, TypeError):
            continue
    
    if feature_names is None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            if hasattr(preprocessor, 'feature_names_'):
                feature_names = preprocessor.feature_names_
            else:
                raise ValueError("Could not get feature names from any preprocessing step")
    
    # Extract feature importance based on classifier type
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    
    if classifier_type == LogisticRegression:
        # Use existing extract_model_coefficients for LogisticRegression
        return extract_model_coefficients(fitted_pipeline)
    
    elif classifier_type == CatBoostClassifier:
        # For CatBoost, use feature importance from the model
        importance = classifier.get_feature_importance()
        
        # Validate lengths match
        if len(feature_names) != len(importance):
            raise ValueError(
                f"Mismatch between number of features ({len(feature_names)}) "
                f"and feature importance values ({len(importance)})"
            )
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'abs_importance': np.abs(importance)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('abs_importance', ascending=False)
        
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    else:
        # For other classifiers, raise an error or handle differently
        raise ValueError(f"Feature importance extraction not implemented for {classifier_type.__name__}")

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

def extract_feature_importance(
    fitted_pipeline: Pipeline, 
    classifier_type: Type[BaseEstimator]
) -> pd.DataFrame:
    """
    Extract feature importance for different classifier types.
    
    Parameters
    ----------
    fitted_pipeline : Pipeline
        A fitted scikit-learn pipeline containing a preprocessor and classifier
    classifier_type : Type[BaseEstimator]
        The type of classifier to extract feature importance for
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing feature names, importance values, 
        and absolute importance, sorted in descending order.
    """
    # Get the preprocessor and classifier from pipeline
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    classifier = fitted_pipeline.named_steps['classifier']
    
    # Find feature names
    steps = list(preprocessor.named_steps.items())
    feature_names = None
    
    for name, step in reversed(steps):
        try:
            feature_names = step.get_feature_names_out()
            break
        except (AttributeError, TypeError):
            continue
    
    if feature_names is None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            if hasattr(preprocessor, 'feature_names_'):
                feature_names = preprocessor.feature_names_
            else:
                raise ValueError("Could not get feature names from any preprocessing step")
    
    # Extract feature importance based on classifier type
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    
    if classifier_type == LogisticRegression:
        # Use existing extract_model_coefficients for LogisticRegression
        return extract_model_coefficients(fitted_pipeline)
    
    elif classifier_type == CatBoostClassifier:
        # For CatBoost, use feature importance from the model
        importance = classifier.get_feature_importance()
        
        # Validate lengths match
        if len(feature_names) != len(importance):
            raise ValueError(
                f"Mismatch between number of features ({len(feature_names)}) "
                f"and feature importance values ({len(importance)})"
            )
        
        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'abs_importance': np.abs(importance)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('abs_importance', ascending=False)
        
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    else:
        # For other classifiers, raise an error or handle differently
        raise ValueError(f"Feature importance extraction not implemented for {classifier_type.__name__}")

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

def tune_model(
    pipeline: Pipeline,
    train_X: pd.DataFrame, 
    train_y: pd.Series,
    tune_X: pd.DataFrame,
    tune_y: pd.Series,
    param_grid: Dict[str, Any],
    metric: str = 'log_loss',
    patience: int = 5,
    min_delta: float = 1e-4
) -> Tuple[Pipeline, Dict[str, Any]]:
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
    X_train_transformed = preprocessor.fit_transform(train_X)
    X_tune_transformed = preprocessor.transform(tune_X)
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
                current_classifier.fit(X_train_transformed, train_y)
                
                # Evaluate on tuning set
                if metric == 'f1':
                    y_tune_pred = current_classifier.predict(X_tune_transformed)
                    score = score_func(tune_y, y_tune_pred)
                else:
                    y_tune_pred_proba = current_classifier.predict_proba(X_tune_transformed)
                    score = score_func(tune_y, y_tune_pred_proba)
                
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
    
    # Create new pipeline with fitted preprocessor and best model
    tuned_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_model)
    ])
    
    return tuned_pipeline, best_params

def find_optimal_threshold(
    y_true: pd.Series, 
    y_pred_proba: np.ndarray, 
    metric: str = 'f1'
) -> Dict[str, float]:
    """
    Find the optimal probability threshold for classification.
    
    Parameters
    ----------
    y_true : pd.Series
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class
    metric : str, optional (default='f1')
        Metric to optimize. Options: 'f1', 'precision', 'recall', 'accuracy'
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing the optimal threshold and corresponding metric value
    """
    # Validate metric
    valid_metrics = ['f1', 'precision', 'recall', 'accuracy', 'f2']
    if metric not in valid_metrics:
        raise ValueError(f"Metric must be one of {valid_metrics}")
    
    # Scoring functions
    scoring_functions = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'accuracy': accuracy_score,
        'f2': lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2.0)  # F2 score weights recall higher than precision
    }
    
    # Thresholds to test
    thresholds = np.linspace(0, 1, 101)  # 101 points from 0 to 1
    
    # Track best threshold
    best_threshold = 0.5
    best_score = 0
    
    # Track best scores
    best_f1 = 0
    
    # Evaluate each threshold
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate scores
        score = scoring_functions[metric](y_true, y_pred)
        f1 = f1_score(y_true, y_pred)  # Always calculate F1 for reference
        
        # Update best threshold if needed
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_f1 = f1
    
    logger = logging.getLogger(__name__)
    logger.info(f"Optimal Threshold Analysis:")
    logger.info(f"  Best {metric} threshold: {best_threshold:.4f}")
    logger.info(f"  Best {metric} score: {best_score:.4f}")
    logger.info(f"  F1 score at optimal threshold: {best_f1:.4f}")
    
    return {
        'threshold': best_threshold,
        f'{metric}_score': best_score,
        'f1_score': best_f1
    }

def format_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Format confusion matrix results into a dictionary."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

def evaluate_model(
    model: LogisticRegression, 
    X: pd.DataFrame, 
    y: pd.Series,
    dataset_name: str = "test",
    find_threshold: bool = False,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Evaluate model performance."""
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Use provided threshold or default to 0.5
    if threshold is not None:
        y_pred = (y_pred_proba >= threshold).astype(int)
        logger = logging.getLogger(__name__)
        logger.info(f"Using threshold {threshold:.4f} for {dataset_name} predictions")
    else:
        y_pred = model.predict(X)
    
    # Calculate standard metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'mcc': matthews_corrcoef(y, y_pred),  # Matthews Correlation Coefficient
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'log_loss': log_loss(y, model.predict_proba(X))
    }
    
    # Add confusion matrix results
    confusion_results = format_confusion_matrix(y, y_pred)
    metrics.update(confusion_results)
    
    logger = logging.getLogger(__name__)
    logger.info(f"{dataset_name.title()} Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Optional threshold optimization
    if find_threshold:
        threshold_results = find_optimal_threshold(y, y_pred_proba)
        metrics.update(threshold_results)
    
    return metrics

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments."""
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
                       choices=['logistic', 'rf', 'svc', 'catboost'],
                       help="Classifier type to use")
    parser.add_argument("--metric", type=str, default="log_loss",
                       choices=["log_loss", "f1", "auc"],
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

def load_data(local_data_path: Optional[str] = None, end_train_year: Optional[int] = None) -> pl.DataFrame:
    """Load data from either local parquet or BigQuery."""
    logger = logging.getLogger(__name__)
    
    if local_data_path:
        try:
            df = pl.read_parquet(local_data_path)
            logger.info(f"Loaded local data from {local_data_path}: {len(df)} rows")
            
            required_columns = ['year_published', 'hurdle']
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
            df = loader.load_training_data(end_train_year=end_train_year + 1, min_ratings=0)
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

def configure_model(classifier_name: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Set up classifier and parameter grid based on classifier type."""
    CLASSIFIER_MAPPING = {
        'logistic': LogisticRegression,
        'rf': RandomForestClassifier,
        'svc': SVC,
        'catboost': CatBoostClassifier
    }
    
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
        },
        'catboost': {
            'classifier__iterations': [100, 300, 500],
            'classifier__learning_rate': [0.01, 0.1, 0.3],
            'classifier__depth': [4, 6, 8],
            'classifier__l2_leaf_reg': [1, 3, 5],
            'classifier__random_strength': [0.5, 1.0, 1.5]
        }
    }
    
    classifier = CLASSIFIER_MAPPING[classifier_name]()
    param_grid = PARAM_GRIDS[classifier_name]
    
    return classifier, param_grid

def fit_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series
) -> Pipeline:
    """
    Fit model on provided data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The sklearn pipeline to fit
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
        
    Returns
    -------
    Pipeline
        Fitted pipeline
    """
    return clone(pipeline).fit(X, y)

def log_experiment(
    experiment: ExperimentTracker,
    pipeline: Pipeline,
    train_metrics: Dict[str, float],
    tune_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_params: Dict[str, Any],
    args: argparse.Namespace
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
        'target': 'users_rated >= 25',
        'target_type': 'binary_classification'
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
            'model_type': 'hurdle',
            'target': 'users_rated >= 25'
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
        from sklearn.linear_model import LogisticRegression
        from catboost import CatBoostClassifier
        
        # Determine the classifier type
        classifier_type = type(pipeline.named_steps['classifier'])
        
        # Extract feature importance
        importance_df = extract_feature_importance(pipeline, classifier_type)
        importance_pl = pl.from_pandas(importance_df)
        experiment.log_coefficients(importance_pl)
        
        # Log top features
        if classifier_type == LogisticRegression:
            logger.info("Top 10 most important features (by absolute coefficient):")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['rank']:2d}. {row['feature']:30s} = {row['coefficient']:8.4f}")
        else:
            logger.info("Top 10 most important features (by importance):")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['rank']:2d}. {row['feature']:30s} = {row['importance']:8.4f}")
        
        # Save model info
        model_info = {
            'n_features': len(importance_df),
            'best_params': best_params,
            'threshold': test_metrics.get('threshold', 0.5),  # Get optimal threshold from test metrics
            'threshold_f1_score': test_metrics.get('f1_score', None),  # Score at optimal threshold
            'confusion_matrix': {
                'validation': {k: tune_metrics[k] for k in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']},
                'test': {k: test_metrics[k] for k in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']}
            }
        }
        
        # Add intercept only for Logistic Regression
        if classifier_type == LogisticRegression:
            model_info['intercept'] = float(pipeline.named_steps['classifier'].intercept_[0])
        
        experiment.log_model_info(model_info)
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {e}")
        logger.error("Continuing without saving feature importance")
    
    # Save pipeline
    experiment.save_pipeline(pipeline)

def main():
    """Main script for training, tuning, and testing a hurdle model."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging()
    
    # Load and split data
    df = load_data(args.local_data, args.test_end_year)
    train_df, tune_df, test_df = create_data_splits(df, args)
    
    # Get X, y splits
    train_X, train_y = select_X_y(train_df, y_column='hurdle')
    tune_X, tune_y = select_X_y(tune_df, y_column='hurdle')
    test_X, test_y = select_X_y(test_df, y_column='hurdle')
    
    # Setup model and pipeline
    classifier, param_grid = configure_model(args.classifier)
    preprocessor = create_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"Classifier: {classifier.__class__.__name__}")
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
    
    # Fit on train data only and evaluate
    train_pipeline = fit_model(tuned_pipeline, train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training")
    
    # Find optimal threshold using tuning data with F2 score
    tune_pred_proba = train_pipeline.predict_proba(tune_X)[:, 1]
    threshold_results = find_optimal_threshold(tune_y, tune_pred_proba, metric='f2')
    optimal_threshold = threshold_results['threshold']
    logger.info(f"Found optimal threshold {optimal_threshold:.4f} on tuning data")
    
    # Evaluate tuning set with optimal threshold
    tune_metrics = evaluate_model(train_pipeline, tune_X, tune_y, "tuning", threshold=optimal_threshold)
    tune_metrics.update(threshold_results)  # Add threshold info to tuning metrics
    
    # Fit final model on combined train+tune data
    logger.info("Fitting final model on combined training + validation data...")
    X_combined = pd.concat([train_X, tune_X])
    y_combined = pd.concat([train_y, tune_y])
    final_pipeline = fit_model(tuned_pipeline, X_combined, y_combined)
    
    # Evaluate on test set with optimal threshold
    test_metrics = evaluate_model(
        final_pipeline, 
        test_X, 
        test_y, 
        "test", 
        threshold=optimal_threshold
    )
    
    # Add threshold to test metrics
    test_metrics.update(threshold_results)
    
    # Log experiment results
    tracker = ExperimentTracker("hurdle", args.output_dir)
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
