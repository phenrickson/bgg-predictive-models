"""Model for predicting game average ratings."""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from .base import BGGModel


class RatingModel(BGGModel):
    """Elastic net regressor to predict game average ratings."""

    def __init__(
        self,
        valid_years: int = 2,
        min_ratings: int = 25,
        random_state: Optional[int] = None,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        **elastic_params: Any,
    ):
        """Initialize rating model.
        
        Args:
            valid_years: Number of years to use for validation
            min_ratings: Minimum number of ratings threshold
            random_state: Random seed for reproducibility
            alpha: Elastic net regularization strength
            l1_ratio: Elastic net mixing parameter (0=ridge, 1=lasso)
            **elastic_params: Additional ElasticNet parameters
        """
        super().__init__(
            name="rating",
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
        )
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.elastic_params = elastic_params
        
        # Initialize but don't fit
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[ElasticNet] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RatingModel":
        """Fit rating model.
        
        Args:
            X: Feature matrix
            y: Average rating values
            
        Returns:
            Self for chaining
        """
        self.feature_names_ = list(X.columns)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit elastic net
        self.model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            random_state=self.random_state,
            **self.elastic_params,
        )
        self.model.fit(X_scaled, y)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate rating predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted average ratings
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fit before predicting")
            
        X_scaled = self.scaler.transform(X[self.feature_names_])
        return self.model.predict(X_scaled)

    def feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients.
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if self.model is None:
            raise ValueError("Model must be fit before getting coefficients")
            
        return pd.DataFrame({
            "feature": self.feature_names_,
            "coefficient": self.model.coef_,
        }).sort_values("coefficient", key=abs, ascending=False)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True average ratings
            
        Returns:
            Dict with performance metrics
        """
        preds = self.predict(X)
        return {
            "rmse": np.sqrt(mean_squared_error(y, preds)),
            "r2": r2_score(y, preds),
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get all model parameters."""
        params = super().get_params(deep)
        params.update({
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            **self.elastic_params,
        })
        return params

    def set_params(self, **params: Any) -> "RatingModel":
        """Set model parameters."""
        elastic_params = {}
        base_params = {}
        
        # Handle alpha and l1_ratio specially
        if "alpha" in params:
            self.alpha = params.pop("alpha")
        if "l1_ratio" in params:
            self.l1_ratio = params.pop("l1_ratio")
            
        # Separate elastic net params from base params
        for key, value in params.items():
            if key in ElasticNet().get_params():
                elastic_params[key] = value
            else:
                base_params[key] = value
                
        super().set_params(**base_params)
        self.elastic_params.update(elastic_params)
        return self
