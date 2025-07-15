"""Train/Tune/Test Hurdle Model for Board Game Ratings Prediction"""
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
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

# Project imports
from src.models.experiments import (
    ExperimentTracker, 
    log_experiment
)

from src.features.transformers import BaseBGGTransformer
from src.data.config import load_config
from src.data.loader import BGGDataLoader
from src.features.preprocessor import create_bgg_preprocessor
from src.models.splitting import time_based_split
from src.models.training import (
    load_data, 
    create_data_splits, 
    select_X_y, 
    create_preprocessing_pipeline, 
    preprocess_data, 
    tune_model, 
    evaluate_model
)

# CatBoost imports
from catboost import CatBoostClassifier
# LightGBM imports
import lightgbm as lgb
from typing import Type, Union, Tuple

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

def configure_model(classifier_name: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Set up classifier and parameter grid based on classifier type."""
    CLASSIFIER_MAPPING = {
        'logistic': LogisticRegression,
        'rf': RandomForestClassifier,
        'oblique_rf': ObliqueRandomForestClassifier,
        'svc': SVC,
        'catboost': CatBoostClassifier,
        'lightgbm': lambda: lgb.LGBMClassifier(
            objective='binary'
        )
    }
    
    PARAM_GRIDS = {
        'logistic': {
            'model__C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
            'model__penalty': ['l2'],
            'model__max_iter': [4000]
        },
        'rf': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        },
        'svc': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['rbf', 'linear'],
            'model__gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'catboost': {
            'model__iterations': [100, 300, 500],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__depth': [4, 6, 8],
            'model__l2_leaf_reg': [1, 3, 5],
            'model__random_strength': [0.5, 1.0, 1.5]
        },
        'lightgbm': {
            'model__n_estimators': [500],
            'model__learning_rate': [0.01, 0.05],
            'model__max_depth': [3, 7],
            'model__num_leaves': [10, 20, 50],
            'model__min_child_samples': [20],
            'model__scale_pos_weight': [1, 2, 5, 7]
        }
    }
    
    classifier = CLASSIFIER_MAPPING[classifier_name]()
    param_grid = PARAM_GRIDS[classifier_name]
    
    return classifier, param_grid

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
    parser.add_argument("--model", type=str, default="logistic",
                       choices=['logistic', 'rf', 'oblique_rf', 'svc', 'catboost', 'lightgbm'],
                       help="Classifier type to use")
    parser.add_argument("--metric", type=str, default="log_loss",
                       choices=["log_loss", "f1", "auc"],
                       help="Metric to optimize during hyperparameter tuning")
    parser.add_argument("--patience", type=int, default=15,
                       help="Number of iterations without improvement before early stopping")
    parser.add_argument("--preprocessor-type", type=str, default="linear",
                       choices=['linear', 'tree'],
                       help="Type of preprocessor to use")
    
    args = parser.parse_args()
    
    # Validate year ranges
    if args.tune_start_year != args.train_end_year:
        raise ValueError(f"tune_start_year ({args.tune_start_year}) must equal train_end_year ({args.train_end_year})")
    
    if not (args.tune_start_year <= args.tune_end_year < args.test_start_year <= args.test_end_year):
        raise ValueError("Invalid year ranges. Must satisfy: tune_start <= tune_end < test_start <= test_end")
    
    return args

def main():
    """Main script for training, tuning, and testing a hurdle model."""
    # Parse arguments and setup logging
    args = parse_arguments()
    logger = setup_logging()
    
    # Load and split data
    df = load_data(args.local_data, args.test_end_year)
    train_df, tune_df, test_df = create_data_splits(
        df,
        train_end_year=args.train_end_year,
        tune_start_year=args.tune_start_year,
        tune_end_year=args.tune_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year
    )
    
    # Get X, y splits
    train_X, train_y = select_X_y(train_df, y_column='hurdle')
    tune_X, tune_y = select_X_y(tune_df, y_column='hurdle')
    test_X, test_y = select_X_y(test_df, y_column='hurdle')
    
    # Setup model and pipeline
    model, param_grid = configure_model(args.model)
    preprocessor = create_preprocessing_pipeline()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Log experiment details
    logger.info(f"Training experiment: {args.experiment}")
    logger.info(f"Classifier: {model.__class__.__name__}")
    logger.info(f"Preprocessor Type: {args.preprocessor_type}")
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
    train_pipeline = clone(tuned_pipeline).fit(train_X, train_y)
    train_metrics = evaluate_model(train_pipeline, train_X, train_y, "training", threshold=0.5)  # Use default threshold for initial evaluation
    
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
    final_pipeline = clone(tuned_pipeline).fit(X_combined, y_combined)
    
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
    experiment = tracker.create_experiment(
        name=args.experiment,
        description=args.description,
        metadata={
            'model_type': 'classification',
            'classifier': args.model,
            'train_end_year': args.train_end_year,
            'tune_end_year': args.tune_end_year,
            'test_end_year': args.test_end_year,
            'optimal_threshold': optimal_threshold
        }
    )
    log_experiment(
        experiment=experiment,
        pipeline=final_pipeline,
        train_metrics=train_metrics,
        tune_metrics=tune_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        args=args,
        train_df=train_df,
        tune_df=tune_df,
        test_df=test_df,
        train_X=train_X,
        tune_X=tune_X,
        test_X=test_X,
        train_y=train_y,
        tune_y=tune_y,
        test_y=test_y,
        model_type='classification'
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
