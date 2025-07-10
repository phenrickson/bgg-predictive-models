"""Tests for experiment tracking and management functions."""
import numpy as np
import pandas as pd
import pytest
import logging
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.models.experiments import log_experiment, Experiment

def create_dummy_pipeline():
    """Create a dummy sklearn pipeline for testing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

def test_log_experiment_confusion_matrix_logging():
    """
    Test that log_experiment correctly logs confusion matrix details 
    for classification models.
    """
    # Create dummy data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        n_classes=2, 
        random_state=42
    )
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Create and fit pipeline
    pipeline = create_dummy_pipeline()
    pipeline.fit(X_df, y_series)
    
    # Predict to generate confusion matrix
    y_pred = pipeline.predict(X_df)
    
    # Prepare mock metrics with confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_series, y_pred)
    
    train_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'confusion_matrix': cm.tolist()
    }
    
    # Create mock experiment and other required arguments
    mock_experiment = MagicMock(spec=Experiment)
    # Add metadata attribute to the mock
    mock_experiment.metadata = {}
    mock_experiment._save_metadata = MagicMock()
    
    # Prepare best params
    best_params = {'C': 1.0, 'penalty': 'l2'}
    
    # Mock logger to capture log messages
    with patch('logging.getLogger') as mock_get_logger:
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call log_experiment
        log_experiment(
            experiment=mock_experiment,
            pipeline=pipeline,
            train_metrics=train_metrics,
            tune_metrics={},
            test_metrics={},
            best_params=best_params,
            args=None,
            train_X=X_df,
            train_y=y_series,
            model_type='classification'
        )
        
        # Verify logging calls
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Check for specific log messages
        assert any("Confusion Matrix" in call for call in log_calls), "Should log confusion matrix header"
        assert any("True Negatives" in call for call in log_calls), "Should log true negatives"
        assert any("False Positives" in call for call in log_calls), "Should log false positives"
        assert any("False Negatives" in call for call in log_calls), "Should log false negatives"
        assert any("True Positives" in call for call in log_calls), "Should log true positives"
        assert any("Total Predictions" in call for call in log_calls), "Should log total predictions"
        assert any("Prediction Breakdown" in call for call in log_calls), "Should log prediction breakdown"
        assert any("Accuracy" in call for call in log_calls), "Should log accuracy"

def test_log_experiment_confusion_matrix_no_metrics():
    """
    Test log_experiment behavior when no confusion matrix is present.
    """
    # Create dummy data
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=5, 
        n_classes=2, 
        random_state=42
    )
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Create and fit pipeline
    pipeline = create_dummy_pipeline()
    pipeline.fit(X_df, y_series)
    
    # Create mock experiment and other required arguments
    mock_experiment = MagicMock(spec=Experiment)
    # Add metadata attribute to the mock
    mock_experiment.metadata = {}
    mock_experiment._save_metadata = MagicMock()
    
    # Prepare best params
    best_params = {'C': 1.0, 'penalty': 'l2'}
    
    # Prepare metrics without confusion matrix
    train_metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88
    }
    
    # Mock logger to capture log messages
    with patch('logging.getLogger') as mock_get_logger:
        # Create a mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Call log_experiment
        log_experiment(
            experiment=mock_experiment,
            pipeline=pipeline,
            train_metrics=train_metrics,
            tune_metrics={},
            test_metrics={},
            best_params=best_params,
            args=None,
            train_X=X_df,
            train_y=y_series,
            model_type='classification'
        )
        
        # Verify no confusion matrix logging occurs
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        
        # Ensure no confusion matrix related logs are present
        assert not any("Confusion Matrix" in call for call in log_calls), "Should not log confusion matrix"
        assert not any("True Negatives" in call for call in log_calls), "Should not log true negatives"
