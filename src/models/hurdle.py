"""Hurdle model for predicting if games will reach minimum rating threshold."""
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .base import BGGModel


class HurdleModel(BGGModel):
    """LightGBM classifier to predict if games reach rating threshold."""

    def __init__(
        self,
        valid_years: int = 2,
        min_ratings: int = 25,
        random_state: Optional[int] = None,
        **lgb_params: Any,
    ):
        """Initialize hurdle model.
        
        Args:
            valid_years: Number of years to use for validation
            min_ratings: Minimum number of ratings threshold
            random_state: Random seed for reproducibility
            **lgb_params: Parameters to pass to LightGBM
        """
        super().__init__(
            name="hurdle",
            valid_years=valid_years,
            min_ratings=min_ratings,
            random_state=random_state,
        )
        
        # Default LightGBM parameters optimized for hurdle prediction
        self.lgb_params = {
            "objective": "binary",
            "metric": "auc",
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HurdleModel":
        """Fit hurdle model.
        
        Args:
            X: Feature matrix
            y: Binary target indicating if game reached min_ratings
            
        Returns:
            Self for chaining
        """
        self.feature_names_ = list(X.columns)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X,
            label=y,
            feature_name=self.feature_names_,
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
        """Generate probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities that each game reaches min_ratings
        """
        if self.model is None:
            raise ValueError("Model must be fit before predicting")
            
        return self.model.predict(X[self.feature_names_])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Alias for predict to match sklearn interface."""
        return self.predict(X).reshape(-1, 1)

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
            y: True binary labels
            
        Returns:
            Dict with performance metrics
        """
        preds = self.predict(X)
        return {
            "auc": roc_auc_score(y, preds),
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get all model parameters."""
        params = super().get_params(deep)
        params.update(self.lgb_params)
        return params

    def set_params(self, **params: Any) -> "HurdleModel":
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
