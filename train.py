#!/usr/bin/env python3
"""
Training script that replicates the 'make models' functionality.
Trains all model candidates: hurdle, complexity, rating, users_rated, and geek_rating.
"""

import yaml
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
from src.utils.logging import setup_logging
from dotenv import load_dotenv

load_dotenv()

logger = setup_logging()


def load_training_config() -> Dict[str, Any]:
    """Load configuration for training from config.yaml"""
    from src.utils.config import load_config

    config = load_config()

    # Convert config to format expected by training functions
    # All years are inclusive (e.g., train_through=2021 means include 2021)
    return {
        "current_year": config.years.current,
        "train_through": config.years.training.train_through,
        "tune_start": config.years.training.tune_start,
        "tune_through": config.years.training.tune_through,
        "test_start": config.years.training.test_start,
        "test_through": config.years.training.test_through,
        "models": {
            "hurdle": config.models["hurdle"].type,
            "complexity": config.models["complexity"].type,
            "rating": config.models["rating"].type,
            "users_rated": config.models["users_rated"].type,
        },
        "experiments": {
            "hurdle": config.models["hurdle"].experiment_name,
            "complexity": config.models["complexity"].experiment_name,
            "rating": config.models["rating"].experiment_name,
            "users_rated": config.models["users_rated"].experiment_name,
        },
        "model_settings": {
            "complexity": {
                "use_sample_weights": config.models["complexity"].use_sample_weights
            },
            "rating": {
                "use_sample_weights": config.models["rating"].use_sample_weights
            },
            "users_rated": {"min_ratings": config.models["users_rated"].min_ratings},
        },
        "paths": {
            "complexity_predictions": config.models["complexity"].predictions_path
        },
    }


def run_command(cmd: list, description: str) -> None:
    """Run a command and handle errors"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)  # noqa
        logger.info(f"✓ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        logger.info(f"✗ {description} failed with exit code {e.returncode}")
        sys.exit(1)


def train_hurdle(config: Dict[str, Any]) -> None:
    """Train hurdle model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.hurdle",
        "--experiment",
        config["experiments"]["hurdle"],
        "--model",
        config["models"]["hurdle"],
        "--train-through",
        str(config["train_through"]),
        "--tune-start",
        str(config["tune_start"]),
        "--tune-through",
        str(config["tune_through"]),
        "--test-start",
        str(config["test_start"]),
        "--test-through",
        str(config["test_through"]),
    ]
    run_command(cmd, "Training hurdle model")


def finalize_hurdle(config: Dict[str, Any]) -> None:
    """Finalize hurdle model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.finalize_model",
        "--model-type",
        "hurdle",
        "--experiment",
        config["experiments"]["hurdle"],
    ]
    run_command(cmd, "Finalizing hurdle model")


def score_hurdle(config: Dict[str, Any]) -> None:
    """Score hurdle model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.score",
        "--model-type",
        "hurdle",
        "--experiment",
        config["experiments"]["hurdle"],
    ]
    run_command(cmd, "Scoring hurdle model")


def train_complexity(config: Dict[str, Any]) -> None:
    """Train complexity model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.complexity",
        "--model",
        config["models"]["complexity"],
        "--experiment",
        config["experiments"]["complexity"],
        "--train-through",
        str(config["train_through"]),
        "--tune-start",
        str(config["tune_start"]),
        "--tune-through",
        str(config["tune_through"]),
        "--test-start",
        str(config["test_start"]),
        "--test-through",
        str(config["test_through"]),
    ]

    # Add use-sample-weights if specified
    if config["model_settings"]["complexity"]["use_sample_weights"]:
        cmd.append("--use-sample-weights")

    run_command(cmd, "Training complexity model")


def finalize_complexity(config: Dict[str, Any]) -> None:
    """Finalize complexity model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.finalize_model",
        "--model-type",
        "complexity",
        "--experiment",
        config["experiments"]["complexity"],
    ]
    run_command(cmd, "Finalizing complexity model")


def score_complexity(config: Dict[str, Any]) -> None:
    """Score complexity model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.score",
        "--model-type",
        "complexity",
        "--experiment",
        config["experiments"]["complexity"],
    ]
    run_command(cmd, "Scoring complexity model")


def train_rating(config: Dict[str, Any]) -> None:
    """Train rating model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.rating",
        "--model",
        config["models"]["rating"],
        "--complexity-experiment",
        config["experiments"]["complexity"],
        "--local-complexity-path",
        config["paths"]["complexity_predictions"],
        "--experiment",
        config["experiments"]["rating"],
        "--train-through",
        str(config["train_through"]),
        "--tune-start",
        str(config["tune_start"]),
        "--tune-through",
        str(config["tune_through"]),
        "--test-start",
        str(config["test_start"]),
        "--test-through",
        str(config["test_through"]),
    ]

    # Add use-sample-weights if specified
    if config["model_settings"]["rating"]["use_sample_weights"]:
        cmd.append("--use-sample-weights")

    run_command(cmd, "Training rating model")


def finalize_rating(config: Dict[str, Any]) -> None:
    """Finalize rating model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.finalize_model",
        "--model-type",
        "rating",
        "--experiment",
        config["experiments"]["rating"],
    ]
    run_command(cmd, "Finalizing rating model")


def score_rating(config: Dict[str, Any]) -> None:
    """Score rating model"""
    # Use scoring complexity predictions (not training predictions)
    complexity_experiment = config["experiments"]["complexity"]
    scoring_complexity_path = (
        f"data/predictions/complexity/{complexity_experiment}_predictions.parquet"
    )
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.score",
        "--model-type",
        "rating",
        "--experiment",
        config["experiments"]["rating"],
        "--complexity-predictions",
        scoring_complexity_path,
    ]
    run_command(cmd, "Scoring rating model")


def train_users_rated(config: Dict[str, Any]) -> None:
    """Train users_rated model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.users_rated",
        "--model",
        config["models"]["users_rated"],
        "--complexity-experiment",
        config["experiments"]["complexity"],
        "--local-complexity-path",
        config["paths"]["complexity_predictions"],
        "--experiment",
        config["experiments"]["users_rated"],
        "--min-ratings",
        str(config["model_settings"]["users_rated"]["min_ratings"]),
        "--train-through",
        str(config["train_through"]),
        "--tune-start",
        str(config["tune_start"]),
        "--tune-through",
        str(config["tune_through"]),
        "--test-start",
        str(config["test_start"]),
        "--test-through",
        str(config["test_through"]),
    ]
    run_command(cmd, "Training users_rated model")


def finalize_users_rated(config: Dict[str, Any]) -> None:
    """Finalize users_rated model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.finalize_model",
        "--model-type",
        "users_rated",
        "--experiment",
        config["experiments"]["users_rated"],
    ]
    run_command(cmd, "Finalizing users_rated model")


def score_users_rated(config: Dict[str, Any]) -> None:
    """Score users_rated model"""
    # Use scoring complexity predictions (not training predictions)
    complexity_experiment = config["experiments"]["complexity"]
    scoring_complexity_path = (
        f"data/predictions/complexity/{complexity_experiment}_predictions.parquet"
    )
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.score",
        "--model-type",
        "users_rated",
        "--experiment",
        config["experiments"]["users_rated"],
        "--complexity-predictions",
        scoring_complexity_path,
    ]
    run_command(cmd, "Scoring users_rated model")


def train_geek_rating(config: Dict[str, Any]) -> None:
    """Train geek rating model (combines all other models)"""
    cmd = [
        "uv",
        "run",
        "-m",
        "src.models.geek_rating",
        "--start-year",
        str(config["test_start"]),
        "--end-year",
        str(config["test_through"]),
        "--hurdle",
        config["experiments"]["hurdle"],
        "--complexity",
        config["experiments"]["complexity"],
        "--rating",
        config["experiments"]["rating"],
        "--users-rated",
        config["experiments"]["users_rated"],
        "--experiment",
        "estimated-geek-rating",
    ]
    run_command(cmd, "Training geek rating model")


def main():
    """Main training pipeline - replicates 'make models'"""
    logger.info("Starting BGG Model Training Pipeline")
    logger.info("====================================")

    # Load configuration
    try:
        config = load_training_config()
        logger.info("Loaded configuration from config.yaml")
        logger.info(f"Current year: {config['current_year']}")
        logger.info(f"Training period: through {config['train_through']}")
        logger.info(
            f"Tuning period: {config['tune_start']} through {config['tune_through']}"
        )
        logger.info(
            f"Testing period: {config['test_start']} through {config['test_through']}"
        )

        # Training models
        logger.info("Training models")
        logger.info(f"Training hurdle model ({config['models']['hurdle']})")
        train_hurdle(config)
        logger.info(f"Training complexity model ({config['models']['complexity']})")
        train_complexity(config)
        logger.info(f"Training complexity model ({config['models']['rating']})")
        train_rating(config)
        logger.info(f"Training complexity model ({config['models']['users_rated']})")
        train_users_rated(config)

        # finalize models on training set
        finalize_hurdle(config)
        score_hurdle(config)

        # finalize complexity
        finalize_complexity(config)
        score_complexity(config)

        # 3. Rating model
        finalize_rating(config)
        score_rating(config)

        # 4. Users rated model
        finalize_users_rated(config)
        score_users_rated(config)

        # 5. Geek rating model (combines all others)
        logger.info("\n🚀 Training geek rating model")
        train_geek_rating(config)

        logger.info(f"\n{'=' * 60}")
        logger.info("🎉 All models trained successfully!")
        logger.info("Ready for registration with register.py")
        logger.info(f"{'=' * 60}")

    except Exception as e:
        if "Config error" in str(e) or "model_config.yml not found" in str(e):
            logger.info(f"Error loading configuration: {e}")
        else:
            logger.info(f"\n❌ Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
