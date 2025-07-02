"""Model for predicting game complexity/weight."""
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .base import BGGModel


class ComplexityModel(BGGModel):
    """LightGBM regressor to predict game complexity (weight)."""

    def __init__(
        self,
        valid_years: int = 2,
        min_ratings: int = 25,
        random_state: Optional[int] = None,
        **lgb_params: Any,
    ):
        """Initialize complexity model.
        
        Args:
            valid_years: Number of years to use for validation
            min_ratings: Minimum number of ratings threshold
            random_state: Random seed for reproducibility
            **lgb_params: Parameters to pass to LightGBM
        """
        super().__init__(
            name="complexity",
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
        )
        
        # Default LightGBM parameters optimized for complexity prediction
        self.lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": random_state,
        }
        # Update with any user-provided parameters
        self.lgb_params.update(lgb_params)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weights: pd.Series = None) -> "ComplexityModel":
        """Fit complexity model.
        
        Args:
            X: Feature matrix
            y: Complexity scores (1-5 scale)
            sample_weights: Optional weights for each sample
            
        Returns:
            Self for chaining
        """
        self.feature_names_ = list(X.columns)
        
        # Create LightGBM dataset with optional weights
        train_data = lgb.Dataset(
            X,
            label=y,
            feature_name=self.feature_names_,
            weight=sample_weights,  # Add optional sample weights
            free_raw_data=False,
        )
        
        # Train model
        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_data,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate complexity predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted complexity scores
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting")
            
        return self.model.predict(X[self.feature_names_])

    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fit before getting feature importance")
            
        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": self.feature_names_,
            "importance": importance,
        }).sort_values("importance", ascending=False)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True complexity scores
            
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
        params.update(self.lgb_params)
        return params

    def set_params(self, **params: Any) -> "ComplexityModel":
        """Set model parameters."""
        lgb_params = {}
        base_params = {}
        
        # Separate LightGBM params from base params
        for key, value in params.items():
            if key in self.lgb_params:
                lgb_params[key] = value
            else:
                base_params[key] = value
                
        super().set_params(**base_params)
        self.lgb_params.update(lgb_params)
        return self
