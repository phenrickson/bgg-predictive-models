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
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    log_loss
)

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

# function to select column and conver tto pandas
def select_X_y(df, y_column, to_pandas = True):
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
    bgg_preprocessor.verbose = True
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
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    X_tune: pd.DataFrame,
    y_tune: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Tune hyperparameters using separate tuning set."""
    
    if param_grid is None:
        param_grid = {
            'C': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
            'penalty': ['l2'],  # Using only L2 with default solver
            'max_iter': [4000]
        }
    
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter tuning...")
    
    best_score = np.inf  # Initialize to infinity since we want to minimize log loss
    best_params = None
    
    for params in ParameterGrid(param_grid):
        # Train model with these parameters
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Evaluate on tuning set
        y_tune_pred_proba = model.predict_proba(X_tune)
        score = log_loss(y_tune, y_tune_pred_proba)
        
        logger.info(f"Params {params}: Log Loss = {score:.4f}")
        
        if score < best_score:  # Lower log loss is better
            best_score = score
            best_params = params
    
    logger.info(f"Best params: {best_params} (Log Loss = {best_score:.4f})")
    return best_params

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
    """Main script for train/tune/test hurdle model."""
    import argparse
    
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
    parser.add_argument("--c", type=float,
                       help="Override C parameter for LogisticRegression")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Log experiment name
    logger.info(f"Training experiment: {args.experiment}")
    
    # Load data (load through the latest test year)
    config = load_config()
    loader = BGGDataLoader(config)
    df = loader.load_training_data(end_train_year=args.test_end_year + 1, min_ratings=0)
    
    logger.info(f"Loaded {len(df)} total rows")
    logger.info(f"Year range: {df['year_published'].min()} - {df['year_published'].max()}")
    
    # Create data splits using time_based_split function
    logger.info("Creating data splits...")
    
    # Calculate windows for validation and test sets
    validation_window = args.tune_end_year - args.tune_start_year + 1
    test_window = args.test_end_year - args.test_start_year + 1
    
    # Use time_based_split to get all three splits
    train_df, tune_df, test_df = time_based_split(
        df=df,
        train_end_year=args.train_end_year,
        prediction_window=validation_window,
        test_window=test_window,
        time_col="year_published",
        return_dict=False,
        to_pandas = True
    )
    
    logger.info(f"Training data: {len(train_df)} rows (years < {args.train_end_year})")
    logger.info(f"Tuning data: {len(tune_df)} rows (years {args.tune_start_year}-{args.tune_end_year})")
    logger.info(f"Test data: {len(test_df)} rows (years {args.test_start_year}-{args.test_end_year})")
    
    # Preprocessing pipeline to be used with model
    preprocessor = create_preprocessing_pipeline()
    
    # Create pipeine to be used with model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])
    
    # Get X, y for train and validation set
    train_X, train_y = select_X_y(train_df, y_column = 'hurdle')
    tune_X, tune_y = select_X_y(tune_df, y_column = 'hurdle')
    
    logger.info(f"Feature dimensions: Train {train_X.shape}, Tune {tune_X.shape}")
    
    # fit to training set
    pipeline.fit(train_X, train_y)
    train_only_model = pipeline
    
    # evaluate
    train_metrics = evaluate_model(train_only_model, train_X, train_y, "training")
    tune_metrics = evaluate_model(train_only_model, tune_X, tune_y, "tuning")
    
    # Now refit on combined training + validation data for final evaluation
    logger.info("Refitting final model on training + validation data...")
    X_train_val = pd.concat([train_X, tune_X])
    y_train_val = pd.concat([train_y, tune_y])
    
    logger.info(f"Feature dimensions: Train + Tune {train_X_val.shape}")

    # fit
    final_model = LogisticRegression(**best_params)
    final_model.fit(train_X train_y_val)

    # Use provided C value or tune hyperparameters
    if args.c is not None:
        best_params = {'C': args.c, 'max_iter': 4000, 'penalty': 'l2'}
        logger.info(f"Using provided C value: {args.c}")
    else:
        best_params = tune_hyperparameters(train_X, train_y, tune_X, tune_y)
    
    # First evaluate model trained only on training data
    logger.info("Evaluating model trained on training data only...")
    train_only_model = LogisticRegression(**best_params)
    train_only_model.fit(X_train, y_train)
    
    train_metrics = evaluate_model(train_only_model, X_train, y_train, "training")
    tune_metrics = evaluate_model(train_only_model, X_tune, y_tune, "tuning")
    

    final_model = LogisticRegression(**best_params)
    
    # Create complete pipeline with final model
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', final_model)
    ])
    
    # Evaluate final model on test set
    test_metrics = evaluate_model(final_model, X_test, y_test, "test (using model fit on train+validation)")
    
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
    
    # Log best parameters
    experiment.log_parameters(best_params)
    
    # Save model coefficients
    feature_names = X_train.columns.tolist()
    coefficients_data = []
    
    for i, (feature, coef) in enumerate(zip(feature_names, final_model.coef_[0])):
        coefficients_data.append({
            'feature': feature,
            'coefficient': coef,
            'abs_coefficient': abs(coef),
            'rank': i + 1  # Will be re-ranked after sorting
        })
    
    # Sort by absolute coefficient value (most important features first)
    coefficients_data.sort(key=lambda x: x['abs_coefficient'], reverse=True)
    
    # Update ranks after sorting
    for i, coef_data in enumerate(coefficients_data):
        coef_data['rank'] = i + 1
    
    # Save coefficients
    coefficients_df = pl.DataFrame(coefficients_data)
    experiment.log_coefficients(coefficients_df)
    
    # Save model info
    model_info = {
        'intercept': float(final_model.intercept_[0]),
        'n_features': len(feature_names),
        'best_params': best_params
    }
    experiment.log_model_info(model_info)
    
    # Save complete pipeline
    experiment.save_pipeline(full_pipeline)
    
    # Log top 10 most important features
    logger.info("Top 10 most important features (by absolute coefficient):")
    for i, coef_data in enumerate(coefficients_data[:10]):
        logger.info(f"  {i+1:2d}. {coef_data['feature']:30s} = {coef_data['coefficient']:8.4f}")
    
    logger.info(f"Model intercept: {final_model.intercept_[0]:.4f}")
    logger.info(f"Results saved to {experiment.exp_dir}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
