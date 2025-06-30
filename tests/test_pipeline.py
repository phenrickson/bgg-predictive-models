"""Tests for the BGG prediction pipeline."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from bgg_predictive_models.models.pipeline import BGGPipeline


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n_samples = 1000
    n_features = 20
    
    # Generate base features
    X, _ = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42,
    )
    
    # Create DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    X = pd.DataFrame(X, columns=feature_names)
    
    # Add year column
    X["year_published"] = np.random.randint(2000, 2022, size=n_samples)
    
    # Generate target variables
    y_hurdle = np.random.binomial(1, 0.7, size=n_samples)
    y_complexity = np.random.normal(2.5, 0.5, size=n_samples).clip(1, 5)
    y_rating = np.random.normal(7, 1, size=n_samples).clip(1, 10)
    y_users_rated = np.exp(np.random.normal(5, 1, size=n_samples))
    
    return X, y_hurdle, y_complexity, y_rating, y_users_rated


def test_pipeline_initialization():
    """Test pipeline initialization with default parameters."""
    pipeline = BGGPipeline()
    assert pipeline.valid_years == 2
    assert pipeline.min_ratings == 25
    assert pipeline.random_state is None


def test_pipeline_initialization_with_params():
    """Test pipeline initialization with custom parameters."""
    model_params = {
        "hurdle": {"num_leaves": 15},
        "complexity": {"num_leaves": 20},
        "rating": {"alpha": 0.1},
        "users_rated": {"alpha": 0.2},
    }
    
    pipeline = BGGPipeline(
        valid_years=3,
        min_ratings=50,
        random_state=42,
        model_params=model_params,
    )
    
    assert pipeline.valid_years == 3
    assert pipeline.min_ratings == 50
    assert pipeline.random_state == 42
    assert pipeline.model_params == model_params


def test_pipeline_fit_predict(sample_data):
    """Test full pipeline fit and predict workflow."""
    X, y_hurdle, y_complexity, y_rating, y_users_rated = sample_data
    
    # Initialize pipeline
    pipeline = BGGPipeline(random_state=42)
    
    # Fit pipeline
    pipeline.fit(
        X=X,
        y_hurdle=y_hurdle,
        y_complexity=y_complexity,
        y_rating=y_rating,
        y_users_rated=y_users_rated,
    )
    
    # Generate predictions
    predictions = pipeline.predict(X)
    
    # Check prediction shapes and ranges
    assert predictions["hurdle"].shape == (len(X),)
    assert predictions["complexity"].shape == (len(X),)
    assert predictions["rating"].shape == (len(X),)
    assert predictions["users_rated"].shape == (len(X),)
    
    assert np.all((predictions["hurdle"] >= 0) & (predictions["hurdle"] <= 1))
    assert np.all((predictions["complexity"] >= 1) & (predictions["complexity"] <= 5))
    assert np.all((predictions["rating"] >= 1) & (predictions["rating"] <= 10))
    assert np.all(predictions["users_rated"] >= 0)


def test_pipeline_bayesaverage(sample_data):
    """Test bayesaverage calculation."""
    X, y_hurdle, y_complexity, y_rating, y_users_rated = sample_data
    
    # Initialize and fit pipeline
    pipeline = BGGPipeline(random_state=42)
    pipeline.fit(
        X=X,
        y_hurdle=y_hurdle,
        y_complexity=y_complexity,
        y_rating=y_rating,
        y_users_rated=y_users_rated,
    )
    
    # Calculate bayesaverage
    bayesavg = pipeline.predict_bayesaverage(
        X,
        prior_rating=5.5,
        prior_weight=100,
    )
    
    # Check shape and range
    assert bayesavg.shape == (len(X),)
    assert np.all((bayesavg >= 1) & (bayesavg <= 10))
    
    # Test with custom prior
    bayesavg_custom = pipeline.predict_bayesaverage(
        X,
        prior_rating=7.0,
        prior_weight=50,
    )
    
    # Custom prior should give different results
    assert not np.allclose(bayesavg, bayesavg_custom)


def test_pipeline_evaluate(sample_data):
    """Test pipeline evaluation metrics."""
    X, y_hurdle, y_complexity, y_rating, y_users_rated = sample_data
    
    # Initialize and fit pipeline
    pipeline = BGGPipeline(random_state=42)
    pipeline.fit(
        X=X,
        y_hurdle=y_hurdle,
        y_complexity=y_complexity,
        y_rating=y_rating,
        y_users_rated=y_users_rated,
    )
    
    # Get evaluation metrics
    metrics = pipeline.evaluate(
        X=X,
        y_hurdle=y_hurdle,
        y_complexity=y_complexity,
        y_rating=y_rating,
        y_users_rated=y_users_rated,
    )
    
    # Check metric structure
    assert "hurdle" in metrics
    assert "complexity" in metrics
    assert "rating" in metrics
    assert "users_rated" in metrics
    
    # Check specific metrics
    assert "auc" in metrics["hurdle"]
    assert "rmse" in metrics["complexity"]
    assert "r2" in metrics["rating"]
    assert "log_rmse" in metrics["users_rated"]


def test_pipeline_feature_validation():
    """Test pipeline validates feature names during prediction."""
    # Train on one set of features
    X_train = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "year_published": np.random.randint(2000, 2022, size=100),
    })
    y = pd.Series(np.random.randint(0, 2, size=100))
    
    pipeline = BGGPipeline(random_state=42)
    pipeline.fit(
        X=X_train,
        y_hurdle=y,
        y_complexity=y,
        y_rating=y,
        y_users_rated=y,
    )
    
    # Try to predict with missing features
    X_test_invalid = pd.DataFrame({
        "feature_1": np.random.randn(10),
        "different_feature": np.random.randn(10),
    })
    
    with pytest.raises(KeyError):
        pipeline.predict(X_test_invalid)
