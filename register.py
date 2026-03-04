#!/usr/bin/env python3
"""
Registration script that replicates the 'make register' functionality.
Registers all trained models to the scoring service.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any
from src.utils.logging import setup_logging
from dotenv import load_dotenv

load_dotenv()

logger = setup_logging()


def load_registration_config() -> Dict[str, Any]:
    """Load configuration for registration from config.yaml"""
    from src.utils.config import load_config

    config = load_config()

    # Convert config to format expected by registration functions
    return {
        "current_year": config.years.current,
        "experiments": {
            "hurdle": config.models["hurdle"].experiment_name,
            "complexity": config.models["complexity"].experiment_name,
            "rating": config.models["rating"].experiment_name,
            "users_rated": config.models["users_rated"].experiment_name,
            "geek_rating": config.models["geek_rating"].experiment_name,
        },
    }


def run_command(cmd: list, description: str) -> None:
    """Run a command and handle errors"""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)  # noqa: F841
        logger.info(f"✓ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        logger.info(f"✗ {description} failed with exit code {e.returncode}")
        sys.exit(1)


def register_complexity(config: Dict[str, Any]) -> None:
    """Register complexity model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "scoring_service.register_model",
        "--model-type",
        "complexity",
        "--experiment",
        config["experiments"]["complexity"],
        "--name",
        f"complexity-v{config['current_year']}",
        "--description",
        f"Production (v{config['current_year']}) model for predicting game complexity",
    ]
    run_command(cmd, "Registering complexity model")


def register_rating(config: Dict[str, Any]) -> None:
    """Register rating model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "scoring_service.register_model",
        "--model-type",
        "rating",
        "--experiment",
        config["experiments"]["rating"],
        "--name",
        f"rating-v{config['current_year']}",
        "--description",
        f"Production (v{config['current_year']}) model for predicting game rating",
    ]
    run_command(cmd, "Registering rating model")


def register_users_rated(config: Dict[str, Any]) -> None:
    """Register users_rated model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "scoring_service.register_model",
        "--model-type",
        "users_rated",
        "--experiment",
        config["experiments"]["users_rated"],
        "--name",
        f"users_rated-v{config['current_year']}",
        "--description",
        f"Production (v{config['current_year']}) model for predicting users_rated",
    ]
    run_command(cmd, "Registering users_rated model")


def register_hurdle(config: Dict[str, Any]) -> None:
    """Register hurdle model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "scoring_service.register_model",
        "--model-type",
        "hurdle",
        "--experiment",
        config["experiments"]["hurdle"],
        "--name",
        f"hurdle-v{config['current_year']}",
        "--description",
        f"Production (v{config['current_year']}) model for predicting whether games will achieve ratings (hurdle)",
    ]
    run_command(cmd, "Registering hurdle model")


def register_geek_rating(config: Dict[str, Any]) -> None:
    """Register geek_rating model"""
    cmd = [
        "uv",
        "run",
        "-m",
        "scoring_service.register_model",
        "--model-type",
        "geek_rating",
        "--experiment",
        config["experiments"]["geek_rating"],
        "--name",
        f"geek_rating-v{config['current_year']}",
        "--description",
        f"Production (v{config['current_year']}) direct model for predicting geek rating",
    ]
    run_command(cmd, "Registering geek_rating model")


def dry_run(config: Dict[str, Any]) -> bool:
    """Validate environment, config, and experiment existence without uploading."""
    from src.utils.config import load_config
    from src.models.experiments import ExperimentTracker

    app_config = load_config()
    environment = app_config.get_environment_prefix()
    bucket_name = app_config.get_bucket_name()
    project_id = os.getenv("ML_PROJECT_ID") or os.getenv("GCP_PROJECT_ID")

    logger.info(f"Environment: {environment}")
    logger.info(f"GCS Bucket:  {bucket_name}")
    logger.info(f"Project ID:  {project_id}")
    logger.info(f"GCS Prefix:  {environment}/models/registered/")
    logger.info(f"Year:        {config['current_year']}")

    all_ok = True
    for model_type, experiment_name in config["experiments"].items():
        registered_name = f"{model_type}-v{config['current_year']}"
        gcs_path = f"gs://{bucket_name}/{environment}/models/registered/{model_type}/{registered_name}/"

        tracker = ExperimentTracker(model_type)
        experiments = tracker.list_experiments()
        matching = [e for e in experiments if e["name"] == experiment_name]

        if matching:
            latest = max(matching, key=lambda x: x["version"])
            logger.info(
                f"  ✓ {model_type}: experiment='{experiment_name}' "
                f"(v{latest['version']}) -> {gcs_path}"
            )
        else:
            logger.info(
                f"  ✗ {model_type}: experiment='{experiment_name}' NOT FOUND locally"
            )
            all_ok = False

    if all_ok:
        logger.info("\nDry run passed. All experiments found.")
    else:
        logger.info("\nDry run failed. Missing experiments listed above.")

    return all_ok


def main():
    """Main registration pipeline - replicates 'make register'"""
    parser = argparse.ArgumentParser(description="Register trained models to scoring service")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, environment, and experiments without uploading",
    )
    args = parser.parse_args()

    logger.info("Starting BGG Model Registration Pipeline")
    logger.info("=======================================")

    # Load configuration and register models
    try:
        config = load_registration_config()
        logger.info("Loaded configuration from config.yaml")
        logger.info(f"Current year: {config['current_year']}")

        if args.dry_run:
            ok = dry_run(config)
            if not ok:
                sys.exit(1)
            return

        logger.info(f"Registering models with version: v{config['current_year']}")

        # Register models in sequence
        logger.info(
            f"\n📝 Registering complexity model ({config['experiments']['complexity']})"
        )
        register_complexity(config)

        logger.info(
            f"\n📝 Registering rating model ({config['experiments']['rating']})"
        )
        register_rating(config)

        logger.info(
            f"\n📝 Registering users_rated model ({config['experiments']['users_rated']})"
        )
        register_users_rated(config)

        logger.info(
            f"\n📝 Registering hurdle model ({config['experiments']['hurdle']})"
        )
        register_hurdle(config)

        logger.info(
            f"\n📝 Registering geek_rating model ({config['experiments']['geek_rating']})"
        )
        register_geek_rating(config)

        logger.info(f"\n{'=' * 60}")
        logger.info("All models registered successfully!")
        logger.info(
            f"Models are now available in the scoring service with version v{config['current_year']}"
        )
        logger.info(f"{'=' * 60}")

    except Exception as e:
        if (
            "Config error" in str(e)
            or "model_config.yml not found" in str(e)
            or "Invalid YAML" in str(e)
        ):
            logger.info(f"Error loading configuration: {e}")
        else:
            logger.info(f"\n❌ Registration pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
