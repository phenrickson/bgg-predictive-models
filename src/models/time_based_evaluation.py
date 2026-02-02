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
        # Train on all data through test_year - 3
        # Tune on test_year - 2 to test_year - 1
        # Test on test_year
        split_config = {
            "train_through": test_year - 3,
            "tune_start": test_year - 2,
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
    ):
        """Initialize evaluator.

        Args:
            output_dir: Base directory for experiments (same as regular training).
            use_embeddings: Whether to include embeddings in features.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings
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

            # Calculate geek rating and evaluate
            geek_results = self._evaluate_geek_rating(
                split=split,
                experiment_refs=experiment_refs,
            )
            all_geek_rating_results.append(geek_results)

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

        # Look for the latest version's pipeline
        version_dirs = [
            d for d in exp_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        ]
        if version_dirs:
            latest_version = max(version_dirs, key=lambda x: int(x.name[1:]))
            pipeline_path = latest_version / "pipeline.pkl"

        if pipeline_path is None or not pipeline_path.exists():
            raise FileNotFoundError(f"Could not find pipeline for {experiment_name}")

        model = joblib.load(pipeline_path)
        logger.info(f"Loaded complexity model from {pipeline_path}")

        # Load data for all years through test year
        df = load_data(
            data_config=get_model_class("complexity").data_config,
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
        pipeline_path = None

        # Look for the latest version's pipeline
        version_dirs = [
            d for d in exp_dir.iterdir()
            if d.is_dir() and d.name.startswith("v")
        ]
        if version_dirs:
            latest_version = max(version_dirs, key=lambda x: int(x.name[1:]))
            pipeline_path = latest_version / "pipeline.pkl"

        if pipeline_path is None or not pipeline_path.exists():
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

        # Load sub-models from evaluation directory
        sub_models = {}
        for model_type in ["hurdle", "complexity", "rating", "users_rated"]:
            sub_models[model_type] = self._load_model_from_eval_dir(
                model_type, experiment_refs[model_type]
            )
            logger.info(f"  Loaded {model_type}: {experiment_refs[model_type]}")

        # Create composite model with loaded sub-models
        scoring_params = self.config.scoring.parameters
        model = GeekRatingModel(
            sub_models=sub_models,
            prior_rating=scoring_params.get("prior_rating", 5.5),
            prior_weight=scoring_params.get("prior_weight", 2000),
        )

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



def run_time_based_evaluation(
    splits: Optional[List[Dict[str, int]]] = None,
    min_ratings: int = 0,
    output_dir: str = "./models/experiments",
    local_data_path: Optional[str] = None,
    model_args: Optional[Dict[str, Dict[str, Any]]] = None,
    additional_args: Optional[List[str]] = None,
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

    evaluator = TimeBasedEvaluator(output_dir=output_dir)
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
