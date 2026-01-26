"""Outcome prediction models for board game metrics."""

from src.models.outcomes.base import (
    DataConfig,
    TrainingConfig,
    Predictor,
    TrainableModel,
    CompositeModel,
)
from src.models.outcomes.hurdle import HurdleModel
from src.models.outcomes.complexity import ComplexityModel
from src.models.outcomes.rating import RatingModel
from src.models.outcomes.users_rated import UsersRatedModel
from src.models.outcomes.geek_rating import GeekRatingModel
from src.models.outcomes.data import (
    load_training_data,
    create_data_splits,
    select_X_y,
)
from src.models.outcomes.train import train_model, get_model_class

__all__ = [
    # Base classes
    "DataConfig",
    "TrainingConfig",
    "Predictor",
    "TrainableModel",
    "CompositeModel",
    # Model classes
    "HurdleModel",
    "ComplexityModel",
    "RatingModel",
    "UsersRatedModel",
    "GeekRatingModel",
    # Data utilities
    "load_training_data",
    "create_data_splits",
    "select_X_y",
    # Training
    "train_model",
    "get_model_class",
]
