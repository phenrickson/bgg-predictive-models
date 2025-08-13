"""Pipeline for coordinating multi-stage model predictions."""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import BGGModel
from .complexity import ComplexityModel
from .hurdle import HurdleModel
from .rating import RatingModel
from .users_rated import UsersRatedModel


class BGGPipeline:
    """Pipeline for coordinating multi-stage model predictions.
    
    This implements the multi-stage modeling approach from the R codebase:
    1. Hurdle model predicts if game will reach rating threshold
    2. Complexity model predicts game weight and imputes missing values
    3. Rating model predicts average rating
    4. Users rated model predicts number of ratings
    5. Combines predictions to calculate bayesaverage
    """

    def __init__(
        self,
        valid_years: int = 2,
        min_ratings: int = 25,
        random_state: Optional[int] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize pipeline.
        
        Args:
            valid_years: Number of years to use for validation
            min_ratings: Minimum number of ratings threshold
            random_state: Random seed for reproducibility
            model_params: Optional dict of model-specific parameters
                Format: {"model_name": {param_dict}}
        """
        self.valid_years = valid_years
        self.min_ratings = min_ratings
        self.random_state = random_state
        
        # Default model parameters
        self.model_params = {
            "hurdle": {},
            "complexity": {},
            "rating": {},
            "users_rated": {},
        }
        # Update with any user-provided parameters
        if model_params:
            for model_name, params in model_params.items():
                if model_name in self.model_params:
                    self.model_params[model_name].update(params)
        
        # Initialize models
        self.hurdle_model = HurdleModel(
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
            **self.model_params["hurdle"],
        )
        
        self.complexity_model = ComplexityModel(
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
            **self.model_params["complexity"],
        )
        
        self.rating_model = RatingModel(
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
            **self.model_params["rating"],
        )
        
        self.users_rated_model = UsersRatedModel(
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
            **self.model_params["users_rated"],
        )

    def fit(
        self,
        X: pd.DataFrame,
        y_hurdle: pd.Series,
        y_complexity: pd.Series,
        y_rating: pd.Series,
        y_users_rated: pd.Series,
        complexity_sample_weights: pd.Series = None,
    ) -> "BGGPipeline":
        """Fit all models in pipeline.
        
        Args:
            X: Feature matrix
            y_hurdle: Binary target for hurdle model
            y_complexity: Complexity scores
            y_rating: Average ratings
            y_users_rated: Number of ratings
            complexity_sample_weights: Optional sample weights for complexity model
            
        Returns:
            Self for chaining
        """
        # Fit hurdle model first
        self.hurdle_model.fit(X, y_hurdle)
        
        # Fit complexity model with optional sample weights
        self.complexity_model.fit(X, y_complexity, sample_weights=complexity_sample_weights)
        
        # Get complexity predictions for imputation
        complexity_preds = self.complexity_model.predict(X)
        
        # Add predicted complexity as feature for rating/users_rated models
        X_with_complexity = X.copy()
        X_with_complexity["predicted_complexity"] = complexity_preds
        
        # Fit rating and users_rated models with complexity feature
        self.rating_model.fit(X_with_complexity, y_rating)
        self.users_rated_model.fit(X_with_complexity, y_users_rated)
        
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions from all models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict with predictions from each model
        """
        # Get hurdle predictions
        hurdle_preds = self.hurdle_model.predict(X)
        
        # Get complexity predictions
        complexity_preds = self.complexity_model.predict(X)
        
        # Add predicted complexity for rating/users_rated models
        X_with_complexity = X.copy()
        X_with_complexity["predicted_complexity"] = complexity_preds
        
        # Get rating and users_rated predictions
        rating_preds = self.rating_model.predict(X_with_complexity)
        users_rated_preds = self.users_rated_model.predict(X_with_complexity)
        
        return {
            "hurdle": hurdle_preds,
            "complexity": complexity_preds,
            "rating": rating_preds,
            "users_rated": users_rated_preds,
        }

    def calculate_bayesaverage(
        self,
        rating_pred: float,
        users_rated_pred: float,
        min_ratings: Optional[int] = None,
        prior_rating: float = 5.5,
        prior_weight: float = 100,
    ) -> float:
        """Calculate bayesaverage using predicted rating and users_rated.
        
        Args:
            rating_pred: Predicted average rating
            users_rated_pred: Predicted number of ratings
            min_ratings: Optional override for minimum ratings threshold
            prior_rating: Prior mean rating (default 5.5)
            prior_weight: Weight given to prior (default 100)
            
        Returns:
            Calculated bayesaverage
        """
        if min_ratings is None:
            min_ratings = self.min_ratings
            
        # Calculate bayesaverage
        n_ratings = max(users_rated_pred, min_ratings)
        bayesavg = (
            (rating_pred * n_ratings + prior_rating * prior_weight) /
            (n_ratings + prior_weight)
        )
        return bayesavg

    def predict_bayesaverage(
        self,
        X: pd.DataFrame,
        prior_rating: float = 5.5,
        prior_weight: float = 100,
    ) -> np.ndarray:
        """Generate bayesaverage predictions.
        
        Args:
            X: Feature matrix
            prior_rating: Prior mean rating
            prior_weight: Weight given to prior
            
        Returns:
            Array of predicted bayesaverages
        """
        # Get component predictions
        preds = self.predict(X)
        
        # Calculate bayesaverage for each game
        bayesavg = np.array([
            self.calculate_bayesaverage(
                rating_pred=r,
                users_rated_pred=n,
                prior_rating=prior_rating,
                prior_weight=prior_weight,
            )
            for r, n in zip(preds["rating"], preds["users_rated"])
        ])
        
        return bayesavg

    def evaluate(
        self,
        X: pd.DataFrame,
        y_hurdle: pd.Series,
        y_complexity: pd.Series,
        y_rating: pd.Series,
        y_users_rated: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all models in pipeline.
        
        Args:
            X: Feature matrix
            y_hurdle: Binary target for hurdle model
            y_complexity: Complexity scores
            y_rating: Average ratings
            y_users_rated: Number of ratings
            
        Returns:
            Dict with metrics for each model
        """
        # Add predicted complexity for rating/users_rated evaluation
        X_with_complexity = X.copy()
        X_with_complexity["predicted_complexity"] = self.complexity_model.predict(X)
        
        return {
            "hurdle": self.hurdle_model.evaluate(X, y_hurdle),
            "complexity": self.complexity_model.evaluate(X, y_complexity),
            "rating": self.rating_model.evaluate(X_with_complexity, y_rating),
            "users_rated": self.users_rated_model.evaluate(
                X_with_complexity, y_users_rated
            ),
        }
