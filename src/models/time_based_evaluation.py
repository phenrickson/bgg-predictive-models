"""Time-based Model Evaluation Pipeline.

Evaluates models using time-based splits, training on historical data
and testing on future years to simulate real-world deployment.
"""

import logging
import argparse
from typing import Dict, Optional, List, Any
from pathlib import Path

import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.models.outcomes.train import get_model_class, train_model
from src.models.outcomes.data import load_training_data, create_data_splits
from src.models.outcomes.geek_rating import GeekRatingModel
from src.models.experiments import ExperimentTracker


logger = logging.getLogger(__name__)


def generate_time_splits(
    start_year: int = 2018,
    end_year: int = 2024,
) -> List[Dict[str, int]]:
    """Generate time-based splits for model evaluation.

    Creates splits where each test year uses all prior data for training.
    Uses a 2-year validation window before the test year.

    Args:
        start_year: First test year to evaluate on.
        end_year: Last test year to evaluate on.

    Returns:
        List of split configurations.
    """
    splits = []

    for test_year in range(start_year, end_year + 1):
        # Train on all data through test_year - 2
        # Tune on test_year - 1
        # Test on test_year
        # This matches config.yaml pattern: train_through=2021, tune=2022, test=2023
        split_config = {
            "train_through": test_year - 2,
            "tune_start": test_year - 1,
            "tune_through": test_year - 1,
            "test_start": test_year,
            "test_through": test_year,
        }
        splits.append(split_config)

    logger.info(f"Generated {len(splits)} time splits for years {start_year}-{end_year}")
    return splits


class TimeBasedEvaluator:
    """Runs time-based evaluation for all models."""

    def __init__(
        self,
        output_dir: str = "./models/experiments",
        use_embeddings: bool = True,
        run_simulation: bool = False,
    ):
        """Initialize evaluator.

        Args:
            output_dir: Base directory for experiments (same as regular training).
            use_embeddings: Whether to include embeddings in features.
            run_simulation: Whether to run simulation-based evaluation after training.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings
        self.run_simulation = run_simulation
        self.config = load_config()

    def run_evaluation(
        self,
        splits: List[Dict[str, int]],
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run time-based evaluation for all model types.

        Args:
            splits: List of time split configurations.
            model_configs: Optional model configurations override.
                Format: {"hurdle": {"algorithm": "logistic"}, ...}

        Returns:
            Dictionary of results DataFrames by model type.
        """
        if model_configs is None:
            model_configs = self._load_model_configs_from_yaml()

        results = {}
        all_geek_rating_results = []

        for split in splits:
            test_year = split["test_through"]
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing split: test_year={test_year}")
            logger.info(f"  Train through: {split['train_through']}")
            logger.info(f"  Tune: {split['tune_start']}-{split['tune_through']}")
            logger.info(f"  Test: {split['test_start']}-{split['test_through']}")
            logger.info(f"{'='*60}")

            # Train complexity first (needed for rating/users_rated)
            complexity_predictions_path = self._train_and_predict_complexity(
                split, model_configs.get("complexity", {})
            )

            # Train other models
            experiment_refs = {"complexity": f"eval-complexity-{test_year}"}

            for model_type in ["hurdle", "rating", "users_rated"]:
                exp_name = f"eval-{model_type}-{test_year}"
                self._train_model(
                    model_type=model_type,
                    split=split,
                    experiment_name=exp_name,
                    model_config=model_configs.get(model_type, {}),
                    complexity_predictions_path=complexity_predictions_path,
                )
                experiment_refs[model_type] = exp_name

            # Train geek_rating model if simulation is requested and mode requires it
            if self.run_simulation:
                sim_config = self.config.simulation
                geek_rating_mode = sim_config.geek_rating_mode if sim_config else "bayesian"
                if geek_rating_mode in ["stacking", "direct"]:
                    geek_rating_exp = f"eval-geek_rating-{test_year}"
                    self._train_geek_rating_model(
                        split=split,
                        experiment_name=geek_rating_exp,
                        experiment_refs=experiment_refs,
                        mode=geek_rating_mode,
                    )
                    experiment_refs["geek_rating"] = geek_rating_exp

            # Calculate geek rating and evaluate
            geek_results = self._evaluate_geek_rating(
                split=split,
                experiment_refs=experiment_refs,
            )
            all_geek_rating_results.append(geek_results)

            # Run simulation evaluation if requested
            if self.run_simulation:
                self._evaluate_simulation(
                    split=split,
                    experiment_refs=experiment_refs,
                )

        # Return aggregated results
        results["geek_rating"] = pd.DataFrame(all_geek_rating_results)

        return results

    def _load_model_configs_from_yaml(self) -> Dict[str, Dict[str, Any]]:
        """Load model configurations from config.yaml."""
        configs = {}

        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            model_config = self.config.models.get(model_type)
            if model_config:
                configs[model_type] = {
                    "algorithm": model_config.type,
                    "use_embeddings": getattr(model_config, "use_embeddings", self.use_embeddings),
                    "use_sample_weights": getattr(model_config, "use_sample_weights", False),
                    "min_ratings": getattr(model_config, "min_ratings", 0),
                }

        return configs

    def _train_model(
        self,
        model_type: str,
        split: Dict[str, int],
        experiment_name: str,
        model_config: Dict[str, Any],
        complexity_predictions_path: Optional[str] = None,
    ) -> None:
        """Train a single model for a split.

        Args:
            model_type: Type of model to train.
            split: Time split configuration.
            experiment_name: Name for the experiment.
            model_config: Model configuration.
            complexity_predictions_path: Path to complexity predictions.
        """
        logger.info(f"Training {model_type} model: {experiment_name}")

        model_class = get_model_class(model_type)

        # Create args namespace
        class Args:
            pass

        args = Args()
        args.model = model_type
        args.algorithm = model_config.get("algorithm")
        args.experiment = experiment_name
        args.description = f"Time-based evaluation for test year {split['test_through']}"
        args.output_dir = str(self.output_dir)
        args.local_data = None
        args.complexity_predictions = complexity_predictions_path
        args.use_embeddings = model_config.get("use_embeddings", self.use_embeddings)
        args.train_through = split["train_through"]
        args.tune_start = split["tune_start"]
        args.tune_through = split["tune_through"]
        args.test_start = split["test_start"]
        args.test_through = split["test_through"]
        args.metric = None
        args.patience = 15
        args.use_sample_weights = model_config.get("use_sample_weights", False)
        args.sample_weight_column = None
        args.preprocessor_type = "auto"
        args.finalize = False
        args.include_count_features = False
        args.algorithm_params = {}

        train_model(model_class, args)

    def _train_and_predict_complexity(
        self,
        split: Dict[str, int],
        model_config: Dict[str, Any],
    ) -> str:
        """Train complexity model and save predictions.

        Args:
            split: Time split configuration.
            model_config: Complexity model configuration.

        Returns:
            Path to complexity predictions file.
        """
        test_year = split["test_through"]
        experiment_name = f"eval-complexity-{test_year}"

        # Train the model
        self._train_model(
            model_type="complexity",
            split=split,
            experiment_name=experiment_name,
            model_config=model_config,
            complexity_predictions_path=None,
        )

        # Load the trained model directly from the experiment directory
        import joblib
        from src.models.outcomes.data import load_data

        # Find the pipeline in the evaluation output directory
        exp_dir = self.output_dir / "complexity" / experiment_name
        pipeline_path = None

        # Look for a version that has a pipeline, starting from latest
        version_dirs = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda x: int(x.name[1:]),
            reverse=True,
        )
        for version_dir in version_dirs:
            candidate_path = version_dir / "pipeline.pkl"
            if candidate_path.exists():
                pipeline_path = candidate_path
                break

        if pipeline_path is None:
            raise FileNotFoundError(f"Could not find pipeline for {experiment_name}")

        model = joblib.load(pipeline_path)
        logger.info(f"Loaded complexity model from {pipeline_path}")

        # Load games that rating/users_rated models will train on
        # Use users_rated data_config (most permissive of the dependent models)
        # so complexity predictions exist for all games they need
        users_rated_data_config = get_model_class("users_rated").data_config
        # But don't require complexity predictions (we're generating them now)
        from src.models.outcomes.base import DataConfig
        prediction_data_config = DataConfig(
            min_ratings=users_rated_data_config.min_ratings,
            min_weights=users_rated_data_config.min_weights,
            requires_complexity_predictions=False,
            supports_embeddings=True,
        )
        df = load_data(
            data_config=prediction_data_config,
            end_year=split["test_through"],
            use_embeddings=model_config.get("use_embeddings", self.use_embeddings),
        )

        # Generate predictions
        df_pandas = df.to_pandas()
        predictions = model.predict(df_pandas)

        # Save predictions
        predictions_dir = self.output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = predictions_dir / f"{experiment_name}.parquet"

        results_df = pl.DataFrame({
            "game_id": df["game_id"],
            "predicted_complexity": predictions,
        })
        results_df.write_parquet(str(predictions_path))

        logger.info(f"Saved complexity predictions to {predictions_path}")
        return str(predictions_path)

    def _train_geek_rating_model(
        self,
        split: Dict[str, int],
        experiment_name: str,
        experiment_refs: Dict[str, str],
        mode: str = "stacking",
    ) -> None:
        """Train a geek_rating model for simulation.

        Args:
            split: Time split configuration.
            experiment_name: Name for the experiment.
            experiment_refs: Dictionary mapping model types to experiment names.
            mode: Training mode - "stacking" or "direct".
        """
        import joblib

        test_year = split["test_through"]
        logger.info(f"Training geek_rating model ({mode} mode): {experiment_name}")

        # Get algorithm from config
        geek_rating_config = self.config.models.get("geek_rating")
        algorithm = "ard" if geek_rating_config is None else geek_rating_config.type
        min_ratings = 25 if geek_rating_config is None else getattr(geek_rating_config, "min_ratings", 25)

        # Build sub_model_experiments dict pointing to eval experiments
        sub_model_experiments = {
            "hurdle": experiment_refs["hurdle"],
            "complexity": experiment_refs["complexity"],
            "rating": experiment_refs["rating"],
            "users_rated": experiment_refs["users_rated"],
        }

        # Create and train the model
        model = GeekRatingModel(mode=mode, min_ratings=min_ratings)

        # Use split years instead of config years
        tune_start = split["tune_start"]
        tune_through = split["tune_through"]

        # Train the model using the sub-model experiments with split-specific years
        if mode == "stacking":
            model.train_stacking_model(
                sub_model_experiments=sub_model_experiments,
                algorithm=algorithm,
                base_dir=str(self.output_dir),
                tune_start=tune_start,
                tune_through=tune_through,
            )
        elif mode == "direct":
            model.train_direct_model(
                sub_model_experiments=sub_model_experiments,
                algorithm=algorithm,
                base_dir=str(self.output_dir),
                tune_through=tune_through,
            )

        # Save the pipeline
        exp_dir = self.output_dir / "geek_rating" / experiment_name / "v1"
        exp_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = exp_dir / "pipeline.pkl"
        joblib.dump(model.pipeline, pipeline_path)
        logger.info(f"Saved geek_rating pipeline to {pipeline_path}")

    def _load_model_from_eval_dir(self, model_type: str, experiment_name: str):
        """Load a model from the evaluation output directory.

        Args:
            model_type: Type of model (hurdle, complexity, rating, users_rated).
            experiment_name: Name of the experiment.

        Returns:
            Loaded sklearn pipeline.
        """
        import joblib

        exp_dir = self.output_dir / model_type / experiment_name
        logger.info(f"Loading model from: {exp_dir}")
        logger.info(f"Directory exists: {exp_dir.exists()}")

        if not exp_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {exp_dir}")

        pipeline_path = None

        # Look for a version that has a pipeline, starting from latest
        all_contents = list(exp_dir.iterdir())
        logger.info(f"Directory contents: {[x.name for x in all_contents]}")

        version_dirs = sorted(
            [d for d in all_contents if d.is_dir() and d.name.startswith("v")],
            key=lambda x: int(x.name[1:]),
            reverse=True,
        )
        logger.info(f"Version dirs found: {[x.name for x in version_dirs]}")

        for version_dir in version_dirs:
            candidate_path = version_dir / "pipeline.pkl"
            logger.info(f"Checking: {candidate_path}, exists: {candidate_path.exists()}")
            if candidate_path.exists():
                pipeline_path = candidate_path
                break

        if pipeline_path is None:
            raise FileNotFoundError(f"Could not find pipeline for {experiment_name}")

        return joblib.load(pipeline_path)

    def _evaluate_geek_rating(
        self,
        split: Dict[str, int],
        experiment_refs: Dict[str, str],
    ) -> Dict[str, Any]:
        """Evaluate geek rating predictions for a split and save as experiment.

        Args:
            split: Time split configuration.
            experiment_refs: Dictionary mapping model types to experiment names.

        Returns:
            Dictionary of evaluation metrics.
        """
        test_year = split["test_through"]
        experiment_name = f"eval-geek_rating-{test_year}"
        logger.info(f"Evaluating geek rating for test year {test_year}")

        # Get geek_rating mode from config
        sim_config = self.config.simulation
        geek_rating_mode = sim_config.geek_rating_mode if sim_config else "bayesian"

        # Load sub-models from evaluation directory
        sub_models = {}
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            sub_models[model_type] = self._load_model_from_eval_dir(
                model_type, experiment_refs[model_type]
            )
            logger.info(f"  Loaded {model_type}: {experiment_refs[model_type]}")

        # Create composite model
        scoring_params = self.config.scoring.parameters
        model = GeekRatingModel(
            mode=geek_rating_mode,
            sub_models=sub_models,
            prior_rating=scoring_params.get("prior_rating", 5.5),
            prior_weight=scoring_params.get("prior_weight", 2000),
            hurdle_threshold=0,  # Predict for all games, don't use hurdle
        )

        # Load trained geek_rating pipeline if using stacking/direct mode
        if geek_rating_mode in ["stacking", "direct"]:
            geek_rating_exp = experiment_refs.get("geek_rating")
            if geek_rating_exp:
                model.pipeline = self._load_model_from_eval_dir("geek_rating", geek_rating_exp)
                logger.info(f"  Loaded geek_rating pipeline: {geek_rating_exp}")

        # Load test data
        from src.models.outcomes.data import load_data
        from src.models.outcomes.base import DataConfig

        # Load ALL games for the test year - no filters
        # GeekRatingModel will predict complexity directly using the sub-model
        data_config = DataConfig(
            min_ratings=0,
            requires_complexity_predictions=False,
            supports_embeddings=True,
        )

        df = load_data(
            data_config=data_config,
            start_year=split["test_start"],
            end_year=split["test_through"],
            use_embeddings=self.use_embeddings,
            apply_filters=False,
        )

        # Generate predictions
        df_pandas = df.to_pandas()
        predictions = model.predict(df_pandas)

        # Calculate metrics for games with valid geek ratings
        actuals = df["geek_rating"].to_numpy()
        pred_values = predictions["predicted_geek_rating"].values

        valid_mask = (actuals > 0) & ~np.isnan(actuals)
        n_valid = valid_mask.sum()

        results = {
            "test_year": test_year,
            "train_through": split["train_through"],
            "n_games": len(df),
            "n_games_with_ratings": int(n_valid),
        }

        # Create experiment tracker and save as experiment
        tracker = ExperimentTracker(
            model_type="geek_rating",
            base_dir=str(self.output_dir),
        )

        # Build metadata for experiment
        experiment_metadata = {
            "split": split,
            "sub_model_experiments": experiment_refs,
            "geek_rating_mode": geek_rating_mode,
            "prior_rating": scoring_params.get("prior_rating", 5.5),
            "prior_weight": scoring_params.get("prior_weight", 2000),
        }

        experiment = tracker.create_experiment(
            name=experiment_name,
            description=f"Time-based evaluation for test year {test_year}",
            metadata=experiment_metadata,
        )

        if n_valid > 0:
            y_true = actuals[valid_mask]
            y_pred = pred_values[valid_mask]

            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
                "n_samples": int(n_valid),
                "n_total": len(df),
            }

            results.update(metrics)

            # Log metrics to experiment (log_metrics expects dict-of-dicts)
            experiment.log_metrics(metrics, "test")

            # Log predictions (convert pandas to polars)
            valid_predictions = predictions[valid_mask].copy()
            valid_predictions["actual"] = y_true
            valid_predictions_pl = pl.from_pandas(valid_predictions)

            experiment.log_predictions(
                predictions=y_pred,
                actuals=y_true,
                df=valid_predictions_pl,
                dataset="test",
            )

            logger.info(f"  RMSE: {metrics['rmse']:.4f}")
            logger.info(f"  MAE:  {metrics['mae']:.4f}")
            logger.info(f"  R²:   {metrics['r2']:.4f}")
            logger.info(f"  n_games evaluated: {n_valid}")
        else:
            logger.warning(f"  No games with valid geek ratings in test year {test_year}")

        logger.info(f"  Saved experiment: {experiment_name}")

        return results

    def _check_bayesian_model(self, pipeline) -> bool:
        """Check if a pipeline contains a Bayesian model that supports simulation.

        Args:
            pipeline: sklearn pipeline to check.

        Returns:
            True if the model has coef_ and sigma_ attributes for posterior sampling.
        """
        model = pipeline.named_steps.get("model")
        if model is None:
            return False
        return hasattr(model, "coef_") and hasattr(model, "sigma_")

    def _evaluate_simulation(
        self,
        split: Dict[str, int],
        experiment_refs: Dict[str, str],
    ) -> None:
        """Run simulation-based evaluation for a split.

        Loads trained models and runs uncertainty propagation simulation.
        Gracefully skips if models don't support Bayesian simulation.

        Args:
            split: Time split configuration.
            experiment_refs: Dictionary mapping model types to experiment names.
        """
        from src.models.outcomes.simulation import (
            simulate_batch,
            precompute_cholesky,
            compute_simulation_metrics,
        )
        from src.models.outcomes.data import load_data
        from src.models.outcomes.base import DataConfig
        from src.pipeline.evaluate_simulation import (
            create_scatter_plots,
            create_top_games_plot,
        )
        import json

        test_year = split["test_through"]
        logger.info(f"\n{'-'*60}")
        logger.info(f"Running simulation evaluation for test year {test_year}")
        logger.info(f"{'-'*60}")

        # Load sub-models from evaluation directory
        pipelines = {}
        for model_type in ["complexity", "rating", "users_rated"]:
            pipelines[model_type] = self._load_model_from_eval_dir(
                model_type, experiment_refs[model_type]
            )

        # Check if models support Bayesian simulation
        unsupported = []
        for model_type, pipeline in pipelines.items():
            if not self._check_bayesian_model(pipeline):
                unsupported.append(model_type)

        if unsupported:
            logger.warning(
                f"  Skipping simulation: {', '.join(unsupported)} model(s) do not support "
                "Bayesian simulation (missing coef_/sigma_ attributes). "
                "Use ARD or similar Bayesian regressors."
            )
            return

        # Get simulation config
        sim_config = self.config.simulation
        n_samples = sim_config.n_samples if sim_config else 500
        geek_rating_mode = sim_config.geek_rating_mode if sim_config else "bayesian"
        random_state = sim_config.random_state if sim_config else 42

        # Load geek_rating pipeline if needed for stacking/direct mode
        geek_rating_pipeline = None
        if geek_rating_mode in ["stacking", "direct"]:
            geek_rating_exp = f"eval-geek_rating-{test_year}"
            try:
                geek_rating_pipeline = self._load_model_from_eval_dir(
                    "geek_rating", geek_rating_exp
                )
                if not self._check_bayesian_model(geek_rating_pipeline):
                    logger.warning(
                        f"  geek_rating model does not support Bayesian simulation. "
                        f"Falling back to bayesian mode."
                    )
                    geek_rating_mode = "bayesian"
                    geek_rating_pipeline = None
            except FileNotFoundError:
                logger.warning(
                    f"  No geek_rating model found for {geek_rating_mode} mode. "
                    f"Falling back to bayesian mode."
                )
                geek_rating_mode = "bayesian"

        logger.info(f"  n_samples: {n_samples}")
        logger.info(f"  geek_rating_mode: {geek_rating_mode}")

        # Load test data
        data_config = DataConfig(
            min_ratings=0,
            requires_complexity_predictions=False,
            supports_embeddings=True,
        )

        df = load_data(
            data_config=data_config,
            start_year=split["test_start"],
            end_year=split["test_through"],
            use_embeddings=self.use_embeddings,
            apply_filters=False,
        )

        df_pandas = df.to_pandas()

        # Filter to games with valid outcomes
        valid_mask = (
            ~df_pandas["rating"].isna()
            & ~df_pandas["users_rated"].isna()
            & (df_pandas["users_rated"] >= 0)
        )
        df_valid = df_pandas[valid_mask].reset_index(drop=True)
        n_games = len(df_valid)

        logger.info(f"  Total games: {len(df_pandas)}")
        logger.info(f"  Valid games: {n_games}")

        if n_games == 0:
            logger.warning(f"  No valid games for simulation")
            return

        # Get scoring params
        scoring_params = self.config.scoring.parameters
        prior_rating = scoring_params.get("prior_rating", 5.5)
        prior_weight = scoring_params.get("prior_weight", 2000)

        # Pre-compute Cholesky decompositions
        logger.info(f"  Pre-computing Cholesky decompositions...")
        cholesky_cache = precompute_cholesky(
            pipelines["complexity"],
            pipelines["rating"],
            pipelines["users_rated"],
            geek_rating_pipeline=geek_rating_pipeline,
        )

        # Run simulation
        logger.info(f"  Running simulation ({n_samples} samples, {n_games} games)...")
        results = simulate_batch(
            df_valid,
            pipelines["complexity"],
            pipelines["rating"],
            pipelines["users_rated"],
            n_samples=n_samples,
            prior_rating=prior_rating,
            prior_weight=prior_weight,
            random_state=random_state,
            cholesky_cache=cholesky_cache,
            geek_rating_mode=geek_rating_mode,
            geek_rating_pipeline=geek_rating_pipeline,
        )

        # Compute metrics
        metrics = compute_simulation_metrics(results)

        # Create output directory
        simulation_dir = self.output_dir / "simulation" / f"eval-{test_year}"
        simulation_dir.mkdir(parents=True, exist_ok=True)

        # Build predictions dataframe for saving and visualization
        predictions_data = []
        for r in results:
            s = r.summary()
            predictions_data.append({
                "game_id": r.game_id,
                "name": r.game_name,
                # Actuals
                "complexity_actual": r.actual_complexity,
                "rating_actual": r.actual_rating,
                "users_rated_actual": s["users_rated"]["actual"],
                "geek_rating_actual": r.actual_geek_rating,
                # Point predictions
                "complexity_point": r.complexity_point,
                "rating_point": r.rating_point,
                "users_rated_point": s["users_rated"]["point"],
                "geek_rating_point": r.geek_rating_point,
                # Simulation median
                "complexity_median": s["complexity"]["median"],
                "rating_median": s["rating"]["median"],
                "users_rated_median": s["users_rated"]["median"],
                "geek_rating_median": s["geek_rating"]["median"],
                # 90% intervals
                "complexity_lower_90": s["complexity"]["interval_90"][0],
                "complexity_upper_90": s["complexity"]["interval_90"][1],
                "rating_lower_90": s["rating"]["interval_90"][0],
                "rating_upper_90": s["rating"]["interval_90"][1],
                "users_rated_lower_90": s["users_rated"]["interval_90"][0],
                "users_rated_upper_90": s["users_rated"]["interval_90"][1],
                "geek_rating_lower_90": s["geek_rating"]["interval_90"][0],
                "geek_rating_upper_90": s["geek_rating"]["interval_90"][1],
                # 50% intervals
                "complexity_lower_50": s["complexity"]["interval_50"][0],
                "complexity_upper_50": s["complexity"]["interval_50"][1],
                "rating_lower_50": s["rating"]["interval_50"][0],
                "rating_upper_50": s["rating"]["interval_50"][1],
                "users_rated_lower_50": s["users_rated"]["interval_50"][0],
                "users_rated_upper_50": s["users_rated"]["interval_50"][1],
                "geek_rating_lower_50": s["geek_rating"]["interval_50"][0],
                "geek_rating_upper_50": s["geek_rating"]["interval_50"][1],
            })

        predictions_df = pl.DataFrame(predictions_data)

        # Save predictions
        predictions_path = simulation_dir / "predictions.parquet"
        predictions_df.write_parquet(predictions_path)
        logger.info(f"  Saved predictions to {predictions_path}")

        # Save metrics
        metrics_path = simulation_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  Saved metrics to {metrics_path}")

        # Create visualizations
        create_scatter_plots(predictions_df, test_year, simulation_dir)
        create_top_games_plot(predictions_df, test_year, simulation_dir)

        # Log summary metrics
        logger.info(f"\n  Simulation Results:")
        for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
            if outcome in metrics and metrics[outcome].get("n", 0) > 0:
                m = metrics[outcome]
                logger.info(f"\n  {outcome.upper()}:")
                logger.info(f"    RMSE (point): {m.get('rmse_point', 'N/A'):.4f}")
                logger.info(f"    RMSE (sim):   {m.get('rmse_sim', 'N/A'):.4f}")
                logger.info(f"    Coverage 90%: {m.get('coverage_90', 0):.1%}")
                logger.info(f"    Coverage 50%: {m.get('coverage_50', 0):.1%}")


def run_time_based_evaluation(
    splits: Optional[List[Dict[str, int]]] = None,
    min_ratings: int = 0,
    output_dir: str = "./models/experiments",
    local_data_path: Optional[str] = None,
    model_args: Optional[Dict[str, Dict[str, Any]]] = None,
    additional_args: Optional[List[str]] = None,
    run_simulation: bool = False,
):
    """Run comprehensive time-based model evaluation pipeline.

    This is the main entry point for backward compatibility with evaluate.py.

    Args:
        splits: Optional list of time splits. If None, generates default splits.
        min_ratings: Minimum number of ratings threshold.
        output_dir: Base directory for experiments (same as regular training).
        local_data_path: Optional path to local data file.
        model_args: Optional dictionary of additional arguments for each model.
        additional_args: Optional list of additional CLI arguments.
        run_simulation: Whether to run simulation-based evaluation after training.
    """
    setup_logging()

    if splits is None:
        config = load_config()
        splits = generate_time_splits(
            start_year=config.years.eval.start,
            end_year=config.years.eval.end,
        )

    # Convert old model_args format to new format
    model_configs = None
    if model_args:
        model_configs = {}
        for model_type, args in model_args.items():
            config = {"algorithm": args.get("model")}
            if args.get("use-sample-weights"):
                config["use_sample_weights"] = True
            if args.get("min-ratings"):
                config["min_ratings"] = args["min-ratings"]
            model_configs[model_type] = config

    evaluator = TimeBasedEvaluator(output_dir=output_dir, run_simulation=run_simulation)
    results = evaluator.run_evaluation(splits=splits, model_configs=model_configs)

    # Print summary
    if "geek_rating" in results:
        summary = results["geek_rating"]
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        print(summary.to_string(index=False))

        if "rmse" in summary.columns:
            logger.info(f"\nMean RMSE: {summary['rmse'].mean():.4f}")
            logger.info(f"Mean MAE:  {summary['mae'].mean():.4f}")
            logger.info(f"Mean R²:   {summary['r2'].mean():.4f}")


def main():
    """CLI entry point for time-based evaluation."""
    parser = argparse.ArgumentParser(description="Run Time-Based Model Evaluation")
    parser.add_argument(
        "--start-year", type=int, default=None, help="First test year for evaluation"
    )
    parser.add_argument(
        "--end-year", type=int, default=None, help="Last test year for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )

    args = parser.parse_args()

    setup_logging()

    config = load_config()
    start_year = args.start_year or config.years.eval.start
    end_year = args.end_year or config.years.eval.end

    splits = generate_time_splits(start_year=start_year, end_year=end_year)

    logger.info(f"Evaluation configuration:")
    logger.info(f"  Test years: {start_year} to {end_year}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Number of splits: {len(splits)}")

    for i, split in enumerate(splits):
        logger.info(
            f"  Split {i+1}: train<={split['train_through']}, "
            f"tune={split['tune_start']}-{split['tune_through']}, "
            f"test={split['test_start']}-{split['test_through']}"
        )

    if args.dry_run:
        logger.info("\nDry run - not executing")
        return

    run_time_based_evaluation(
        splits=splits,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
