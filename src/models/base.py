"""Base classes for BGG predictive models."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


class BGGModel(ABC, BaseEstimator):
    """Base class for all BGG predictive models."""

    def __init__(
        self,
        name: str,
        valid_years: int = 2,
        min_ratings: int = 25,
        random_state: Optional[int] = None,
    ):
        """Initialize base model.
        
        Args:
            name: Model identifier
            valid_years: Number of years to use for validation
            min_ratings: Minimum number of ratings threshold
            random_state: Random seed for reproducibility
        """
        self.name = name
        self.valid_years = valid_years
        self.min_ratings = min_ratings
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.feature_names_: Optional[list[str]] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BGGModel":
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data."""
        pass

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "name": self.name,
            "valid_years": self.valid_years,
            "min_ratings": self.min_ratings,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "BGGModel":
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class TimeBasedSplit(BaseCrossValidator):
    """Custom time-based cross-validation splitter.
    
    Implements the time-based validation strategy from the R codebase where:
    - Training data includes games up to end_train_year
    - Validation data includes games from the following valid_years
    """

    def __init__(
        self, 
        year_column: str = "year_published",
        end_train_year: int = 2021,
        valid_years: int = 2,
    ):
        """Initialize splitter.
        
        Args:
            year_column: Column containing publication years
            end_train_year: Last year to include in training
            valid_years: Number of years to use for validation
        """
        self.year_column = year_column
        self.end_train_year = end_train_year
        self.valid_years = valid_years

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[Any] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate train/validation indices.
        
        Args:
            X: Feature matrix containing year_column
            y: Target variable (not used)
            groups: Group labels (not used)
            
        Returns:
            Tuple of (train_idx, val_idx)
        """
        years = X[self.year_column]
        train_mask = years <= self.end_train_year
        val_mask = (years > self.end_train_year) & (
            years <= self.end_train_year + self.valid_years
        )
        
        yield (
            np.where(train_mask)[0],
            np.where(val_mask)[0]
        )

    def get_n_splits(
        self, X: Optional[Any] = None, y: Optional[Any] = None, groups: Optional[Any] = None
    ) -> int:
        """Return number of splits (always 1)."""
        return 1
