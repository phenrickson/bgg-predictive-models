"""Geek rating model - composite model combining predictions from sub-models."""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import logging
import joblib

from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, BayesianRidge, ARDRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from catboost import CatBoostRegressor
import lightgbm as lgb

from src.models.outcomes.base import (
    CompositeModel,
    DataConfig,
    Predictor,
    TrainableModel,
    TrainingConfig,
)


logger = logging.getLogger(__name__)


class GeekRatingModel(TrainableModel):
    """Model that predicts geek_rating from sub-model predictions.

    Supports two modes:
    - bayesian: Uses hardcoded Bayesian average formula (no training)
    - stacking: Trains a model on sub-model predictions

    For stacking mode, trains on tune set predictions from sub-model experiments.
    """

    model_type = "geek_rating"
    target_column = "geek_rating"
    model_task = "regression"

    # Default data_config - will be overridden per-instance in __init__ for direct mode
    data_config = DataConfig(
        min_ratings=0,
        requires_complexity_predictions=False,
        supports_embeddings=True,  # Sub-models need embeddings
    )

    # Feature columns from sub-model predictions (interactions added via PolynomialFeatures)
    PREDICTION_FEATURES = [
        "predicted_complexity",
        "predicted_rating",
        "predicted_users_rated_log",  # Log-transformed for comparable scale
    ]

    # Optional prediction features (controlled by include_predictions flag)
    # predicted_complexity is always included as it's part of the model chain
    OPTIONAL_PREDICTION_FEATURES = [
        "predicted_rating",
        "predicted_users_rated_log",
    ]

    def __init__(
        self,
        training_config: TrainingConfig = None,
        mode: str = "stacking",
        sub_models: Optional[Dict[str, Predictor]] = None,
        prior_rating: float = 5.5,
        prior_weight: float = 2000,
        hurdle_threshold: Optional[float] = None,
        min_ratings: int = 25,
        include_predictions: bool = True,
        **kwargs,
    ):
        """Initialize GeekRatingModel.

        Args:
            training_config: Training configuration.
            mode: Either "bayesian", "stacking", or "direct".
                - bayesian: Uses hardcoded Bayesian average formula (no training)
                - stacking: Trains on sub-model predictions only (tune set)
                - direct: Trains on sub-model predictions + all game features (all games with min_ratings)
            sub_models: Dictionary of sub-models (hurdle, complexity, rating, users_rated).
            prior_rating: Prior mean rating for Bayesian average.
            prior_weight: Weight given to prior rating.
            hurdle_threshold: Threshold for hurdle model classification.
            min_ratings: Minimum ratings for training data filtering (default 25).
                NOTE: This is NOT passed to data_config. Data is loaded with min_ratings=0
                so tune/test include all games. Filtering by min_ratings happens in
                prepare_features for train split only.
            include_predictions: Whether to include sub-model predictions as features (direct mode only).
            **kwargs: Additional arguments passed to TrainableModel.
        """
        # Don't pass min_ratings to super - keep class-level data_config with min_ratings=0
        # so ALL data is loaded. Filtering happens in prepare_features.
        super().__init__(training_config=training_config, **kwargs)
        self.mode = mode
        self.sub_models = sub_models or {}
        self.prior_rating = prior_rating
        self.prior_weight = prior_weight
        self.hurdle_threshold = hurdle_threshold
        self.min_ratings = min_ratings  # Store for use in prepare_features
        self.include_predictions = include_predictions
        self._sub_model_experiments: Dict[str, str] = {}
        self._direct_feature_columns: List[str] = []  # Store feature columns for direct mode

    def configure_model(
        self, algorithm: str, algorithm_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """Configure regressor and parameter grid for stacking model.

        Args:
            algorithm: Algorithm name.
            algorithm_params: Optional algorithm-specific parameters.

        Returns:
            Tuple of (regressor_instance, param_grid).
        """
        MODEL_MAPPING = {
            "ridge": Ridge,
            "lasso": Lasso,
            "bayesian_ridge": BayesianRidge,
            "ard": ARDRegression,
            "catboost": CatBoostRegressor,
            "lightgbm": lgb.LGBMRegressor,
        }

        PARAM_GRIDS = {
            "ridge": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            },
            "lasso": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
            },
            "bayesian_ridge": {},
            "ard": {
                "model__threshold_lambda": [10000, 100000],
            },
            "catboost": {
                "model__iterations": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__depth": [3, 4, 5],
            },
            "lightgbm": {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.01, 0.1],
                "model__max_depth": [3, 5],
                "model__num_leaves": [15, 31],
            },
        }

        if algorithm not in MODEL_MAPPING:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Supported: {list(MODEL_MAPPING.keys())}"
            )

        model_class = MODEL_MAPPING[algorithm]
        if algorithm_params:
            model = model_class(**algorithm_params)
        else:
            model = model_class()

        param_grid = PARAM_GRIDS[algorithm]

        return model, param_grid

    def post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Clip predictions to valid geek rating range.

        Args:
            predictions: Raw model predictions.

        Returns:
            Predictions clipped to reasonable range.
        """
        # Geek ratings are typically between 1 and 10, but with Bayesian
        # shrinkage they cluster more narrowly
        return np.clip(predictions, 1.0, 10.0)

    def prepare_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split_name: str,
        args: Any = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features by running sub-models to generate predictions.

        Args:
            X: Input features DataFrame.
            y: Target values.
            split_name: Name of the split ('train', 'tune', 'test').
            args: Arguments namespace with sub-model experiment names.

        Returns:
            Tuple of (prepared_X with prediction columns, y).
        """
        from src.utils.config import load_config

        # Get sub-model experiments from args or config
        if args is not None and hasattr(args, "sub_model_experiments"):
            sub_model_experiments = args.sub_model_experiments
        else:
            config = load_config()
            sub_model_experiments = {
                "hurdle": getattr(config.models.get("hurdle"), "experiment_name", None),
                "complexity": getattr(config.models.get("complexity"), "experiment_name", None),
                "rating": getattr(config.models.get("rating"), "experiment_name", None),
                "users_rated": getattr(config.models.get("users_rated"), "experiment_name", None),
            }

        # Store for later use
        self._sub_model_experiments = sub_model_experiments

        # Order matters: complexity must come before rating/users_rated
        ordered_experiments = {}
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            if model_type in sub_model_experiments and sub_model_experiments[model_type]:
                ordered_experiments[model_type] = sub_model_experiments[model_type]

        # Generate predictions from sub-models
        logger.info(f"Generating sub-model predictions for {split_name} set ({len(X)} samples)")
        predictions_df = self.generate_predictions_for_games(X, ordered_experiments)

        if self.mode == "stacking":
            # Stacking mode: only use prediction features
            X_prepared = predictions_df[self.PREDICTION_FEATURES].copy()
        else:
            # Direct mode: combine original features with predictions
            exclude_cols = [
                "game_id", "name", "geek_rating",
                "rating", "users_rated", "complexity", "hurdle",
                "log_users_rated",
            ]
            feature_cols = [c for c in X.columns if c not in exclude_cols]
            X_prepared = X[feature_cols].copy()

            # Add predicted_complexity (always)
            if "predicted_complexity" in predictions_df.columns:
                X_prepared["predicted_complexity"] = predictions_df["predicted_complexity"].values

            # Optionally add other prediction features
            if self.include_predictions:
                for col in self.OPTIONAL_PREDICTION_FEATURES:
                    if col in predictions_df.columns:
                        X_prepared[col] = predictions_df[col].values

            self._direct_feature_columns = list(X_prepared.columns)

        # Handle filtering differently for train vs tune/test
        if split_name == "train":
            # For training: filter to users_rated >= min_ratings AND geek_rating > 0
            users_rated = X["users_rated"] if "users_rated" in X.columns else None
            zero_mask = y == 0

            if users_rated is not None:
                min_ratings_mask = users_rated < self.min_ratings
                combined_mask = zero_mask | min_ratings_mask
                n_zero = zero_mask.sum()
                n_low_ratings = min_ratings_mask.sum()
                n_filtered = combined_mask.sum()
                logger.info(
                    f"Filtering training data: {n_zero} games with geek_rating=0, "
                    f"{n_low_ratings} games with users_rated < {self.min_ratings}, "
                    f"{n_filtered} total removed"
                )
                X_prepared = X_prepared[~combined_mask].copy()
                y = y[~combined_mask].copy()
            elif zero_mask.any():
                # Fallback if users_rated not available
                n_filtered = zero_mask.sum()
                logger.info(f"Filtering {n_filtered} games with geek_rating=0 from training")
                X_prepared = X_prepared[~zero_mask].copy()
                y = y[~zero_mask].copy()
        else:
            # For tune/test: filter to games above hurdle threshold, replace geek_rating=0 with prior (5.5)
            from src.models.score import extract_threshold

            # Get hurdle threshold
            hurdle_threshold = self.hurdle_threshold
            if hurdle_threshold is None and "hurdle" in sub_model_experiments:
                hurdle_threshold = extract_threshold(sub_model_experiments["hurdle"], "hurdle")
            if hurdle_threshold is None:
                hurdle_threshold = 0.5  # Default fallback

            # Filter to games above hurdle threshold
            if "predicted_hurdle_prob" in predictions_df.columns:
                hurdle_mask = predictions_df["predicted_hurdle_prob"] >= hurdle_threshold
                n_before = len(X_prepared)
                n_above_threshold = hurdle_mask.sum()
                logger.info(
                    f"Filtering {split_name} to games above hurdle threshold {hurdle_threshold:.3f}: "
                    f"{n_above_threshold}/{n_before} games"
                )
                X_prepared = X_prepared[hurdle_mask].copy()
                y = y[hurdle_mask].copy()

            # Replace geek_rating=0 with prior (5.5)
            zero_mask = y == 0
            if zero_mask.any():
                n_zeros = zero_mask.sum()
                logger.info(f"Replacing {n_zeros} games with geek_rating=0 -> 5.5 (prior) for {split_name}")
                y = y.copy()
                y[zero_mask] = 5.5

        return X_prepared, y

    def filter_for_refit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        original_X: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter tune data for refit (train + tune combined).

        When refitting on combined train + tune data, we want to apply the same
        filtering as training: only games with users_rated >= min_ratings AND
        geek_rating > 0.

        Args:
            X: Prepared features from tune set (after prepare_features).
            y: Target values from tune set (after prepare_features, so 0s are now 5.5).
            original_X: Original X before prepare_features (has users_rated column).

        Returns:
            Tuple of (filtered_X, filtered_y) for refit.
        """
        # y has already had 0s replaced with 5.5 by prepare_features
        # We need to filter based on original users_rated and identify games that
        # originally had geek_rating=0 (now 5.5 but we should exclude them)

        # For refit, we want games with users_rated >= min_ratings
        # The geek_rating=0 -> 5.5 replacement means we can't easily identify those
        # So we need to filter based on users_rated from original_X

        if "users_rated" not in original_X.columns:
            logger.warning("users_rated not in original_X, cannot filter for refit")
            return X, y

        # Get users_rated values aligned with current X index
        users_rated = original_X.loc[X.index, "users_rated"]
        min_ratings_mask = users_rated >= self.min_ratings

        n_before = len(X)
        X_filtered = X[min_ratings_mask].copy()
        y_filtered = y[min_ratings_mask].copy()
        n_after = len(X_filtered)

        logger.info(
            f"Filtered tune data for refit: {n_before} -> {n_after} "
            f"(removed {n_before - n_after} games with users_rated < {self.min_ratings})"
        )

        return X_filtered, y_filtered

    def create_pipeline(
        self,
        estimator: Any,
        preprocessor: Any,
        algorithm: str,
        args: Any = None,
    ) -> Pipeline:
        """Create pipeline with mode-specific preprocessing.

        Args:
            estimator: The model estimator.
            preprocessor: The preprocessing pipeline (ignored for stacking mode).
            algorithm: Algorithm name.
            args: Optional arguments namespace.

        Returns:
            sklearn Pipeline.
        """
        if self.mode == "stacking":
            # Stacking mode: use PolynomialFeatures for interactions
            return Pipeline([
                ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ])
        else:
            # Direct mode: use standard BGG preprocessor
            return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    def combine_bayesian(
        self, rating: np.ndarray, users_rated: np.ndarray
    ) -> np.ndarray:
        """Combine predictions using Bayesian average formula.

        Args:
            rating: Predicted average ratings.
            users_rated: Predicted number of user ratings.

        Returns:
            Geek rating predictions.
        """
        geek_rating = (
            (rating * users_rated) + (self.prior_rating * self.prior_weight)
        ) / (users_rated + self.prior_weight)
        return geek_rating

    def generate_predictions_for_games(
        self,
        features: pd.DataFrame,
        sub_model_experiments: Dict[str, str],
        base_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Generate sub-model predictions for a set of games.

        Loads trained sub-model pipelines and runs predictions.

        Args:
            features: DataFrame with game features (including embeddings).
            sub_model_experiments: Dict mapping model type to experiment name.
            base_dir: Optional base directory for experiments.

        Returns:
            DataFrame with game_id and predicted_* columns.
        """
        from src.models.score import load_model

        results = pd.DataFrame(index=features.index)
        results["game_id"] = features["game_id"]

        # Load and run each sub-model
        for model_type, experiment_name in sub_model_experiments.items():
            logger.info(f"Loading {model_type} model: {experiment_name}")
            pipeline = load_model(experiment_name, model_type, base_dir=base_dir)

            if model_type == "hurdle":
                # Get probability of positive class
                if hasattr(pipeline, "predict_proba"):
                    proba = pipeline.predict_proba(features)[:, 1]
                else:
                    proba = pipeline.predict(features)
                results["predicted_hurdle_prob"] = proba

            elif model_type == "complexity":
                raw_pred = pipeline.predict(features)
                results["predicted_complexity"] = np.clip(raw_pred, 1, 5)

            elif model_type == "rating":
                # Rating model needs predicted_complexity as input
                features_with_complexity = features.copy()
                features_with_complexity["predicted_complexity"] = results["predicted_complexity"]
                raw_pred = pipeline.predict(features_with_complexity)
                results["predicted_rating"] = np.clip(raw_pred, 1, 10)

            elif model_type == "users_rated":
                # Users rated model needs predicted_complexity as input
                features_with_complexity = features.copy()
                features_with_complexity["predicted_complexity"] = results["predicted_complexity"]
                raw_log_pred = pipeline.predict(features_with_complexity)
                # Transform from log scale
                raw_counts = np.expm1(raw_log_pred)
                results["predicted_users_rated"] = np.maximum(
                    np.round(raw_counts / 50) * 50, 25
                )
                # Log-transformed version for stacking model
                results["predicted_users_rated_log"] = np.log1p(results["predicted_users_rated"])

            logger.info(f"  Generated {len(results)} predictions for {model_type}")

        return results

    def prepare_training_data(
        self,
        sub_model_experiments: Dict[str, str],
        base_dir: str = "./models/experiments",
        tune_start: Optional[int] = None,
        tune_through: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data by loading models and generating predictions.

        Loads the tune set games, runs sub-model predictions on all of them,
        and returns features/target for stacking model training.

        Args:
            sub_model_experiments: Dict mapping model type to experiment name.
            base_dir: Base directory for experiments.
            tune_start: Start year for tune data (defaults to config if not provided).
            tune_through: End year for tune data (defaults to config if not provided).

        Returns:
            Tuple of (features DataFrame, target Series).
        """
        from src.utils.config import load_config
        from src.models.outcomes.data import load_data

        config = load_config()
        tune_start = tune_start if tune_start is not None else config.years.training.tune_start
        tune_through = tune_through if tune_through is not None else config.years.training.tune_through

        # Load tune set games with embeddings (needed for sub-models)
        logger.info(f"Loading tune set games: {tune_start}-{tune_through}")
        data_config = DataConfig(min_ratings=0, supports_embeddings=True)
        df = load_data(
            data_config=data_config,
            start_year=tune_start,
            end_year=tune_through,
            use_embeddings=True,
            apply_filters=False,
        )

        df_pandas = df.to_pandas()
        logger.info(f"Loaded {len(df_pandas)} tune set games")

        # Generate predictions from sub-models
        # Order matters: complexity must come before rating/users_rated
        ordered_experiments = {}
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            if model_type in sub_model_experiments:
                ordered_experiments[model_type] = sub_model_experiments[model_type]

        predictions_df = self.generate_predictions_for_games(
            df_pandas, ordered_experiments, base_dir=base_dir
        )

        # Add actual geek_rating
        predictions_df["geek_rating"] = df_pandas["geek_rating"].values

        # Filter to games with valid geek ratings
        valid_mask = predictions_df["geek_rating"] > 0
        predictions_df = predictions_df[valid_mask].copy()

        logger.info(f"Training data: {len(predictions_df)} games with valid geek ratings")

        # Extract features and target
        X = predictions_df[self.PREDICTION_FEATURES]
        y = predictions_df["geek_rating"]

        return X, y

    def train_stacking_model(
        self,
        sub_model_experiments: Dict[str, str],
        algorithm: str = "ridge",
        base_dir: str = "./models/experiments",
        tune_start: Optional[int] = None,
        tune_through: Optional[int] = None,
    ) -> Pipeline:
        """Train stacking model on sub-model predictions.

        Args:
            sub_model_experiments: Dict mapping model type to experiment name.
            algorithm: Algorithm to use for stacking.
            base_dir: Base directory for experiments.
            tune_start: Start year for tune data (defaults to config if not provided).
            tune_through: End year for tune data (defaults to config if not provided).

        Returns:
            Trained sklearn pipeline.
        """
        # Store experiment references
        self._sub_model_experiments = sub_model_experiments

        # Prepare data
        X, y = self.prepare_training_data(
            sub_model_experiments, base_dir, tune_start=tune_start, tune_through=tune_through
        )

        # Configure model
        model, _ = self.configure_model(algorithm)

        # Build pipeline with interaction terms
        pipeline = Pipeline([
            ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        # Fit
        logger.info(f"Training stacking model with {algorithm} on {len(X)} samples")
        pipeline.fit(X, y)

        self.pipeline = pipeline

        return pipeline

    def prepare_direct_training_data(
        self,
        sub_model_experiments: Dict[str, str],
        base_dir: str = "./models/experiments",
        tune_through: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for direct mode.

        Loads all games with min_ratings, runs sub-model predictions,
        and combines predictions with all original game features.

        Args:
            sub_model_experiments: Dict mapping model type to experiment name.
            base_dir: Base directory for experiments.
            tune_through: End year for training data (defaults to config if not provided).

        Returns:
            Tuple of (features DataFrame, target Series).
        """
        from src.utils.config import load_config
        from src.models.outcomes.data import load_data

        config = load_config()
        tune_through = tune_through if tune_through is not None else config.years.training.tune_through

        # Load all games with enough ratings (not just tune set)
        logger.info(f"Loading games with >= {self.min_ratings} ratings for direct training (through {tune_through})")
        data_config = DataConfig(min_ratings=self.min_ratings, supports_embeddings=True)
        df = load_data(
            data_config=data_config,
            start_year=None,  # All years
            end_year=tune_through,  # Up through tune year
            use_embeddings=True,
            apply_filters=True,
        )

        df_pandas = df.to_pandas()
        logger.info(f"Loaded {len(df_pandas)} games for direct training")

        # Generate predictions from sub-models
        ordered_experiments = {}
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            if model_type in sub_model_experiments:
                ordered_experiments[model_type] = sub_model_experiments[model_type]

        predictions_df = self.generate_predictions_for_games(
            df_pandas, ordered_experiments, base_dir=base_dir
        )

        # Combine original features with sub-model predictions
        # Drop columns that shouldn't be features
        # Keep year_published - needed by BGGPreprocessor for year features
        exclude_cols = [
            "game_id", "name", "geek_rating",
            "rating", "users_rated", "complexity", "hurdle",
            "log_users_rated",  # Target-related columns
        ]
        feature_cols = [c for c in df_pandas.columns if c not in exclude_cols]

        # Build combined feature set
        X = df_pandas[feature_cols].copy()

        # Always add predicted_complexity (part of model chain)
        if "predicted_complexity" in predictions_df.columns:
            X["predicted_complexity"] = predictions_df["predicted_complexity"].values

        # Optionally add rating/users_rated predictions
        if self.include_predictions:
            for col in self.OPTIONAL_PREDICTION_FEATURES:
                if col in predictions_df.columns:
                    X[col] = predictions_df[col].values

        # Store feature columns for prediction time
        self._direct_feature_columns = list(X.columns)

        # Target
        y = df_pandas["geek_rating"]

        # Filter to valid geek ratings
        valid_mask = y > 0
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()

        logger.info(f"Direct training data: {len(X)} games, {len(self._direct_feature_columns)} features")

        return X, y

    def train_direct_model(
        self,
        sub_model_experiments: Dict[str, str],
        algorithm: str = "ridge",
        base_dir: str = "./models/experiments",
        tune_through: Optional[int] = None,
    ) -> Pipeline:
        """Train direct model on sub-model predictions + all game features.

        Args:
            sub_model_experiments: Dict mapping model type to experiment name.
            algorithm: Algorithm to use.
            base_dir: Base directory for experiments.
            tune_through: End year for training data (defaults to config if not provided).

        Returns:
            Trained sklearn pipeline.
        """
        from src.features.preprocessor import create_bgg_preprocessor

        # Store experiment references
        self._sub_model_experiments = sub_model_experiments

        # Prepare data
        X, y = self.prepare_direct_training_data(
            sub_model_experiments, base_dir, tune_through=tune_through
        )

        # Configure model
        model, _ = self.configure_model(algorithm)

        # Build pipeline with BGG preprocessor (handles all feature engineering)
        # Use tree preprocessing for tree-based models, linear for others
        preprocessor_type = "tree" if algorithm in ["catboost", "lightgbm"] else "linear"
        # Always preserve year_published and predicted_complexity
        preserve_cols = ["year_published", "predicted_complexity"]
        # Optionally preserve rating/users_rated predictions
        if self.include_predictions:
            preserve_cols = preserve_cols + self.OPTIONAL_PREDICTION_FEATURES
        preprocessor = create_bgg_preprocessor(
            model_type=preprocessor_type,
            preserve_columns=preserve_cols,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Fit
        logger.info(f"Training direct model with {algorithm} on {len(X)} samples, {len(X.columns)} features")
        pipeline.fit(X, y)

        self.pipeline = pipeline

        return pipeline

    def predict_from_features(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions from pre-computed sub-model predictions.

        Args:
            features: DataFrame with predicted_* columns from sub-models.
                For direct mode, also needs all original game features.

        Returns:
            Geek rating predictions.
        """
        if self.mode == "bayesian":
            return self.combine_bayesian(
                features["predicted_rating"].values,
                features["predicted_users_rated"].values,
            )
        elif self.mode == "stacking":
            if self.pipeline is None:
                raise ValueError("Stacking model not trained. Call train_stacking_model first.")

            X = features[self.PREDICTION_FEATURES]
            predictions = self.pipeline.predict(X)
            return self.post_process_predictions(predictions)
        elif self.mode == "direct":
            if self.pipeline is None:
                raise ValueError("Direct model not trained. Call train_direct_model first.")

            # For direct mode, pass features to pipeline
            # If _direct_feature_columns is set (from training), use those columns
            # Otherwise, let the pipeline's preprocessor handle column selection
            if self._direct_feature_columns:
                X = features[self._direct_feature_columns]
            else:
                X = features
            predictions = self.pipeline.predict(X)
            return self.post_process_predictions(predictions)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def predict_from_sub_models(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions by running sub-models first.

        This is the full prediction pipeline that runs all sub-models
        and then combines their predictions.

        Args:
            features: Raw game features.

        Returns:
            DataFrame with all intermediate and final predictions.
        """
        self._validate_sub_models()

        # Get hurdle threshold
        threshold = self.hurdle_threshold
        if threshold is None and hasattr(self.sub_models["hurdle"], "optimal_threshold"):
            threshold = self.sub_models["hurdle"].optimal_threshold
        if threshold is None:
            threshold = 0.5

        # Initialize results
        results = pd.DataFrame(index=features.index)

        # Preserve identifying fields
        for col in ["game_id", "name", "year_published"]:
            if col in features.columns:
                results[col] = features[col]

        # Step 1: Predict hurdle probability
        hurdle_model = self.sub_models["hurdle"]
        if hasattr(hurdle_model, "predict_proba"):
            hurdle_proba = hurdle_model.predict_proba(features)[:, 1]
        else:
            hurdle_proba = hurdle_model.predict(features)

        results["predicted_hurdle_prob"] = hurdle_proba

        # Identify likely games (above threshold)
        likely_mask = hurdle_proba >= threshold

        # Step 2: For likely games, predict complexity
        complexity_pipeline = self.sub_models["complexity"]
        results["predicted_complexity"] = 1.0  # Default

        if likely_mask.any():
            likely_features = features[likely_mask]
            raw_complexity = complexity_pipeline.predict(likely_features)
            results.loc[likely_mask, "predicted_complexity"] = np.clip(raw_complexity, 1, 5)

        # Step 3: Add complexity to features for rating/users_rated
        features_with_complexity = features.copy()
        features_with_complexity["predicted_complexity"] = results["predicted_complexity"]

        # Step 4: Predict rating and users_rated for likely games
        rating_pipeline = self.sub_models["rating"]
        users_rated_pipeline = self.sub_models["users_rated"]

        # Defaults for unlikely games
        results["predicted_rating"] = 5.5
        results["predicted_users_rated"] = 25.0

        if likely_mask.any():
            likely_features_with_complexity = features_with_complexity[likely_mask]

            # Rating predictions
            raw_rating = rating_pipeline.predict(likely_features_with_complexity)
            results.loc[likely_mask, "predicted_rating"] = np.clip(raw_rating, 1, 10)

            # Users rated predictions (model predicts log1p)
            raw_log_users_rated = users_rated_pipeline.predict(likely_features_with_complexity)
            raw_counts = np.expm1(raw_log_users_rated)
            results.loc[likely_mask, "predicted_users_rated"] = np.maximum(
                np.round(raw_counts / 50) * 50, 25
            )

        # Step 4b: Compute derived features for stacking model
        results["predicted_users_rated_log"] = np.log1p(results["predicted_users_rated"])

        # Step 5: Calculate geek rating using selected mode
        if self.mode == "direct":
            # Direct mode needs full game features plus sub-model predictions
            features_for_geek = features.copy()
            for col in self.PREDICTION_FEATURES:
                features_for_geek[col] = results[col]
            results["predicted_geek_rating"] = self.predict_from_features(features_for_geek)
        else:
            # Bayesian and stacking only need sub-model predictions
            results["predicted_geek_rating"] = self.predict_from_features(results)

        return results

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions.

        If sub_models are loaded, runs full pipeline.
        Otherwise, expects features to contain predicted_* columns.

        Args:
            features: Either raw game features (if sub_models loaded)
                or DataFrame with predicted_* columns.

        Returns:
            DataFrame with predictions.
        """
        # Check if we have sub-model predictions as input
        has_predictions = all(col in features.columns for col in self.PREDICTION_FEATURES)

        if has_predictions and not self.sub_models:
            # Features are sub-model predictions, just combine them
            predictions = self.predict_from_features(features)
            results = features.copy()
            results["predicted_geek_rating"] = predictions
            return results
        elif self.sub_models:
            # Run full pipeline with sub-models
            return self.predict_from_sub_models(features)
        else:
            raise ValueError(
                "Either sub_models must be loaded or features must contain "
                "predicted_* columns from sub-models"
            )

    def _validate_sub_models(self) -> None:
        """Validate that all required sub-models are loaded."""
        required = ["hurdle", "complexity", "rating", "users_rated"]
        missing = [m for m in required if m not in self.sub_models]

        if missing:
            raise ValueError(f"Missing required sub-models: {missing}")

    def load_sub_models(self, experiments: Dict[str, str]) -> None:
        """Load sub-models from experiment names.

        Args:
            experiments: Dictionary mapping model types to experiment names.
        """
        from src.models.score import load_model, extract_threshold

        logger.info("Loading sub-models:")
        for model_type, experiment_name in experiments.items():
            logger.info(f"  {model_type}: {experiment_name}")
            self.sub_models[model_type] = load_model(experiment_name, model_type)

        # Extract optimal threshold from hurdle experiment if not already set
        if self.hurdle_threshold is None and "hurdle" in experiments:
            loaded_threshold = extract_threshold(experiments["hurdle"], "hurdle")
            if loaded_threshold is not None:
                self.hurdle_threshold = loaded_threshold
                logger.info(f"Using threshold from hurdle experiment: {self.hurdle_threshold}")

        self._sub_model_experiments = experiments

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.pipeline is not None:
            pipeline_path = path / "pipeline.pkl"
            joblib.dump(self.pipeline, pipeline_path)
            logger.info(f"Saved pipeline to {pipeline_path}")

        # Save metadata
        metadata = {
            "mode": self.mode,
            "prior_rating": self.prior_rating,
            "prior_weight": self.prior_weight,
            "hurdle_threshold": self.hurdle_threshold,
            "min_ratings": self.min_ratings,
            "include_predictions": self.include_predictions,
            "sub_model_experiments": self._sub_model_experiments,
            "direct_feature_columns": self._direct_feature_columns,
        }

        import json
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Union[str, Path]) -> None:
        """Load a saved model from disk.

        Args:
            path: Path to load the model from.
        """
        path = Path(path)

        # Load pipeline if exists
        pipeline_path = path / "pipeline.pkl"
        if pipeline_path.exists():
            self.pipeline = joblib.load(pipeline_path)
            logger.info(f"Loaded pipeline from {pipeline_path}")

        # Load metadata
        import json
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.mode = metadata.get("mode", "stacking")
            self.prior_rating = metadata.get("prior_rating", 5.5)
            self.prior_weight = metadata.get("prior_weight", 2000)
            self.hurdle_threshold = metadata.get("hurdle_threshold")
            self.min_ratings = metadata.get("min_ratings", 25)
            self.include_predictions = metadata.get("include_predictions", True)
            self._sub_model_experiments = metadata.get("sub_model_experiments", {})
            self._direct_feature_columns = metadata.get("direct_feature_columns", [])

    @classmethod
    def from_experiments(
        cls,
        hurdle_experiment: str,
        complexity_experiment: str,
        rating_experiment: str,
        users_rated_experiment: str,
        mode: str = "bayesian",
        prior_rating: float = 5.5,
        prior_weight: float = 2000,
        hurdle_threshold: Optional[float] = None,
    ) -> "GeekRatingModel":
        """Create GeekRatingModel from experiment names.

        Args:
            hurdle_experiment: Experiment name for hurdle model.
            complexity_experiment: Experiment name for complexity model.
            rating_experiment: Experiment name for rating model.
            users_rated_experiment: Experiment name for users_rated model.
            mode: "bayesian", "stacking", or "direct".
            prior_rating: Prior mean rating for Bayesian average.
            prior_weight: Weight given to prior rating.
            hurdle_threshold: Optional threshold override.

        Returns:
            Initialized GeekRatingModel with loaded sub-models.
        """
        model = cls(
            mode=mode,
            prior_rating=prior_rating,
            prior_weight=prior_weight,
            hurdle_threshold=hurdle_threshold,
        )

        experiments = {
            "hurdle": hurdle_experiment,
            "complexity": complexity_experiment,
            "rating": rating_experiment,
            "users_rated": users_rated_experiment,
        }

        model.load_sub_models(experiments)

        return model

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata including configuration."""
        return {
            "model_type": self.model_type,
            "mode": self.mode,
            "prior_rating": self.prior_rating,
            "prior_weight": self.prior_weight,
            "hurdle_threshold": self.hurdle_threshold,
            "sub_model_experiments": self._sub_model_experiments,
        }


def _create_comparison_plot(
    y_true: np.ndarray,
    bayesian_preds: np.ndarray,
    stacking_preds: np.ndarray,
    output_path: Path,
) -> None:
    """Create comparison plot of Bayesian vs Stacking predictions.

    Args:
        y_true: Actual geek rating values.
        bayesian_preds: Bayesian average predictions.
        stacking_preds: Stacking model predictions.
        output_path: Path to save the plot.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Bayesian predictions vs actual (predicted on x, actual on y)
    ax1 = axes[0]
    ax1.scatter(bayesian_preds, y_true, alpha=0.3, s=10, c="steelblue")
    min_val, max_val = min(y_true.min(), bayesian_preds.min()), max(y_true.max(), bayesian_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax1.set_xlabel("Predicted Geek Rating")
    ax1.set_ylabel("Actual Geek Rating")
    ax1.set_title(f"Bayesian Average\nRMSE: {np.sqrt(mean_squared_error(y_true, bayesian_preds)):.4f}, R²: {r2_score(y_true, bayesian_preds):.4f}")

    # Plot 2: Stacking predictions vs actual (predicted on x, actual on y)
    ax2 = axes[1]
    ax2.scatter(stacking_preds, y_true, alpha=0.3, s=10, c="forestgreen")
    min_val, max_val = min(y_true.min(), stacking_preds.min()), max(y_true.max(), stacking_preds.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax2.set_xlabel("Predicted Geek Rating")
    ax2.set_ylabel("Actual Geek Rating")
    ax2.set_title(f"Stacking Model\nRMSE: {np.sqrt(mean_squared_error(y_true, stacking_preds)):.4f}, R²: {r2_score(y_true, stacking_preds):.4f}")

    # Plot 3: Bayesian vs Stacking predictions
    ax3 = axes[2]
    ax3.scatter(bayesian_preds, stacking_preds, alpha=0.3, s=10, c="purple")
    min_val, max_val = min(bayesian_preds.min(), stacking_preds.min()), max(bayesian_preds.max(), stacking_preds.max())
    ax3.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax3.set_xlabel("Bayesian Prediction")
    ax3.set_ylabel("Stacking Prediction")
    ax3.set_title("Bayesian vs Stacking Predictions")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot to {output_path}")


def main():
    """Train geek rating model using the standard training pipeline."""
    from src.models.outcomes.train import train_model

    train_model(GeekRatingModel)


if __name__ == "__main__":
    main()
