"""Tests for geek rating prediction module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from tests.conftest import check_experiments_exist
from src.models.geek_rating import (
    predict_game,
    calculate_geek_rating,
    predict_geek_rating,
)

# Skip entire module if experiments are not available
pytestmark = pytest.mark.skipif(
    not check_experiments_exist(),
    reason="Required experiments not available (need experiments in hurdle, complexity, rating, and users_rated folders)",
)


@pytest.fixture
def sample_features():
    """Create a sample DataFrame with game features."""
    return pd.DataFrame(
        {
            "game_id": [12345],
            "name": ["Test Game"],
            "year_published": [2020],
            "designer_count": [2],
            "category_count": [3],
            "mechanism_count": [4],
        }
    )


@pytest.fixture
def mock_models():
    """Create mock models for testing."""
    models = {
        "hurdle": MagicMock(),
        "complexity": MagicMock(),
        "rating": MagicMock(),
        "users_rated": MagicMock(),
    }

    # Setup mock predictions
    models["hurdle"].predict_proba.return_value = np.array([[0.1, 0.9]])
    models["complexity"].predict.return_value = np.array([3.5])
    models["rating"].predict.return_value = np.array([7.2])
    models["users_rated"].predict.return_value = np.array([np.log1p(500)])

    return models


def test_predict_game(sample_features, mock_models):
    """Test game prediction function."""
    predictions = predict_game(sample_features, mock_models)

    assert "predicted_hurdle_prob" in predictions
    assert "predicted_complexity" in predictions
    assert "predicted_rating" in predictions
    assert "predicted_users_rated" in predictions

    assert predictions["predicted_hurdle_prob"].iloc[0] > 0.5
    assert predictions["predicted_complexity"].iloc[0] > 0
    assert predictions["predicted_rating"].iloc[0] > 0
    assert predictions["predicted_users_rated"].iloc[0] >= 25


def test_calculate_geek_rating(sample_features, mock_models):
    """Test geek rating calculation."""
    predictions = predict_game(sample_features, mock_models)
    geek_rating = calculate_geek_rating(predictions)

    assert 1 <= geek_rating.iloc[0] <= 10


def test_predict_geek_rating(sample_features, mock_models):
    """Test full geek rating prediction."""
    result = predict_geek_rating(sample_features, models=mock_models)

    assert "prediction" in result
    assert 1 <= result["prediction"].iloc[0] <= 10


def test_predict_geek_rating_low_probability(sample_features, mock_models):
    """Test prediction for game with low rating probability."""
    # Mock low probability of getting ratings
    mock_models["hurdle"].predict_proba.return_value = np.array([[0.9, 0.1]])

    result = predict_geek_rating(sample_features, models=mock_models)

    # Check default values for low probability game
    assert result["predicted_hurdle_prob"].iloc[0] < 0.5
    assert result["predicted_complexity"].iloc[0] == 1.0
    assert result["predicted_rating"].iloc[0] == 5.5
    assert result["predicted_users_rated"].iloc[0] == 25
    assert result["prediction"].iloc[0] == 5.5


def test_predict_geek_rating_experiments(sample_features, mock_models, monkeypatch):
    """Test loading models from experiments."""

    # Mock load_all_models to return mock_models
    def mock_load_models(*args, **kwargs):
        return mock_models

    monkeypatch.setattr("src.models.geek_rating.load_all_models", mock_load_models)

    result = predict_geek_rating(
        sample_features,
        experiments={
            "hurdle_experiment": "hurdle/test",
            "complexity_experiment": "complexity/test",
            "rating_experiment": "rating/test",
            "users_rated_experiment": "users_rated/test",
        },
    )

    assert "prediction" in result
    assert 1 <= result["prediction"].iloc[0] <= 10


def test_predict_geek_rating_no_models_or_experiments():
    """Test error handling when no models or experiments provided."""
    with pytest.raises(ValueError, match="No valid method provided to load models"):
        predict_geek_rating(pd.DataFrame())
