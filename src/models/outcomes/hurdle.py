"""Hurdle model for predicting if games receive enough ratings."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

from src.models.outcomes.base import DataConfig, TrainableModel, TrainingConfig


def find_optimal_threshold(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
) -> Dict[str, float]:
    """Find optimal probability threshold for classification.

    This is a standalone function for use outside of HurdleModel.

    Args:
        y_true: True binary labels.
        y_pred_proba: Predicted probabilities for positive class.
        metric: Metric to optimize ('f1', 'f2', 'precision', 'recall', 'accuracy').

    Returns:
        Dictionary with optimal threshold and scores.
    """
    scoring_functions = {
        "f1": f1_score,
        "f2": lambda y_true, y_pred: fbeta_score(y_true, y_pred, beta=2.0),
        "precision": precision_score,
        "recall": recall_score,
        "accuracy": accuracy_score,
    }

    if metric not in scoring_functions:
        raise ValueError(f"Metric must be one of {list(scoring_functions.keys())}")

    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = 0
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = scoring_functions[metric](y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_f1 = f1

    return {
        "threshold": best_threshold,
        f"{metric}_score": best_score,
        "f1_score": best_f1,
    }


class HurdleModel(TrainableModel):
    """Binary classification model predicting if games receive minimum ratings.

    The hurdle model is the first stage in the prediction pipeline,
    determining which games are likely to receive enough user ratings
    to have meaningful rating/complexity predictions.
    """

    model_type = "hurdle"
    target_column = "hurdle"
    model_task = "classification"

    data_config = DataConfig(
        min_ratings=0,
        requires_complexity_predictions=False,
        supports_embeddings=True,
    )

    def __init__(self, training_config: TrainingConfig = None, **kwargs):
        """Initialize HurdleModel.

        Args:
            training_config: Training configuration.
            **kwargs: Additional arguments passed to TrainableModel.
        """
        super().__init__(training_config=training_config, **kwargs)
        self.optimal_threshold: float = 0.5

    def configure_model(
        self, algorithm: str, algorithm_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Configure classifier and parameter grid.

        Args:
            algorithm: Algorithm name.
            algorithm_params: Optional algorithm-specific parameters from config.

        Returns:
            Tuple of (classifier_instance, param_grid).
        """
        CLASSIFIER_MAPPING = {
            "logistic": LogisticRegression,
            "rf": RandomForestClassifier,
            "svc": SVC,
            "catboost": CatBoostClassifier,
            "lightgbm": lambda: lgb.LGBMClassifier(objective="binary"),
        }

        PARAM_GRIDS = {
            "logistic": {
                "model__C": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1],
                "model__penalty": ["l2"],
                "model__max_iter": [4000],
            },
            "rf": {
                "model__n_estimators": [300],
                "model__max_depth": [10, 20],
                "model__min_samples_split": [10],
                "model__max_features": ["sqrt"],
                "model__bootstrap": [True],
            },
            "svc": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto", 0.1, 0.01],
            },
            "catboost": {
                "model__iterations": [500],
                "model__learning_rate": [0.01, 0.5],
                "model__depth": [4, 6, 8],
                "model__l2_leaf_reg": [1, 3, 5],
            },
            "lightgbm": {
                "model__n_estimators": [500],
                "model__learning_rate": [0.01, 0.05],
                "model__max_depth": [3, 7],
                "model__num_leaves": [10, 20, 50],
                "model__min_child_samples": [20],
                "model__scale_pos_weight": [1, 2, 5, 7],
            },
        }

        if algorithm not in CLASSIFIER_MAPPING:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {list(CLASSIFIER_MAPPING.keys())}"
            )

        # Create model instance with optional config params
        model_class = CLASSIFIER_MAPPING[algorithm]
        if algorithm_params:
            classifier = model_class(**algorithm_params)
        else:
            classifier = model_class()
        param_grid = PARAM_GRIDS[algorithm]

        return classifier, param_grid

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Get probability predictions.

        Args:
            features: Input features.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        if self.pipeline is None:
            raise ValueError("Model has not been trained or loaded")
        return self.pipeline.predict_proba(features)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Generate binary predictions using optimal threshold.

        Args:
            features: Input features.

        Returns:
            Binary predictions.
        """
        proba = self.predict_proba(features)[:, 1]
        return (proba >= self.optimal_threshold).astype(int)

    def find_optimal_threshold(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray,
        metric: str = "f2",
    ) -> Dict[str, float]:
        """Find optimal probability threshold for classification.

        Delegates to the module-level find_optimal_threshold function
        and stores the result in self.optimal_threshold.

        Args:
            y_true: True binary labels.
            y_pred_proba: Predicted probabilities for positive class.
            metric: Metric to optimize ('f1', 'f2', 'precision', 'recall', 'accuracy').

        Returns:
            Dictionary with optimal threshold and scores.
        """
        result = find_optimal_threshold(y_true, y_pred_proba, metric)
        self.optimal_threshold = result["threshold"]
        return result

    def compute_additional_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Compute classification-specific metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            dataset_name: Name of dataset.

        Returns:
            Dictionary with confusion matrix and threshold info.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "optimal_threshold": self.optimal_threshold,
        }


if __name__ == "__main__":
    from src.models.outcomes.train import train_model

    train_model(HurdleModel)
