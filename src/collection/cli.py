"""Command-line interface for user collection modeling pipeline."""

import argparse
import json
import logging
import sys

from src.utils.logging import setup_logging
from src.collection.collection_pipeline import CollectionPipeline, PipelineConfig
from src.collection.collection_artifact_storage import ArtifactStorageConfig
from src.collection.collection_split import SplitConfig
from src.collection.collection_model import ModelConfig
from src.collection.collection_analyzer import AnalyzerConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="User Collection Modeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline for a user
  uv run python -m src.collection.cli run --username phenrickson

  # Run without refreshing collection from BGG API
  uv run python -m src.collection.cli run --username phenrickson --no-refresh

  # Refresh predictions only with existing model
  uv run python -m src.collection.cli predict --username phenrickson

  # Check pipeline status
  uv run python -m src.collection.cli status --username phenrickson
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--username", "-u",
        required=True,
        help="BGG username",
    )
    common.add_argument(
        "--environment", "-e",
        default=None,
        help="Environment (dev/prod). Uses config default if not specified.",
    )
    common.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Run full pipeline
    run_parser = subparsers.add_parser(
        "run",
        parents=[common],
        help="Run full pipeline (fetch, train, predict, analyze)",
    )
    run_parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Don't refresh collection from BGG API",
    )
    run_parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "catboost", "logistic"],
        help="Model type to use (default: lightgbm)",
    )
    run_parser.add_argument(
        "--train-through",
        type=int,
        default=None,
        help="Last year to include in training data (inclusive, for time-based splits)",
    )
    run_parser.add_argument(
        "--negative-ratio",
        type=float,
        default=5.0,
        help="Ratio of negative to positive samples (default: 5.0)",
    )
    run_parser.add_argument(
        "--sampling-strategy",
        default="popularity_weighted",
        choices=["random", "popularity_weighted", "uniform"],
        help="Negative sampling strategy (default: popularity_weighted)",
    )

    # Predict only
    predict_parser = subparsers.add_parser(
        "predict",
        parents=[common],
        help="Refresh predictions with existing model",
    )
    predict_parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Model version to use (default: latest)",
    )

    # Status
    status_parser = subparsers.add_parser(
        "status",
        parents=[common],
        help="Check pipeline artifact status",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    return parser


def run_full_pipeline(args: argparse.Namespace) -> int:
    """Run the full pipeline."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running full pipeline for user '{args.username}'")

    # Build configuration
    config = PipelineConfig(
        storage_config=ArtifactStorageConfig(
            environment=args.environment,
        ),
        split_config=SplitConfig(
            negative_sampling_ratio=args.negative_ratio,
            negative_sampling_strategy=args.sampling_strategy,
        ),
        model_config=ModelConfig(
            model_type=args.model_type,
        ),
        analyzer_config=AnalyzerConfig(),
        train_through=args.train_through,
    )

    # Run pipeline
    pipeline = CollectionPipeline(args.username, config)

    try:
        results = pipeline.run_full_pipeline(
            refresh_collection=not args.no_refresh,
        )

        # Print summary
        print("\n" + "=" * 60)
        print(f"Pipeline completed for user: {args.username}")
        print("=" * 60)

        if "steps" in results:
            for step_name, step_data in results["steps"].items():
                status = step_data.get("status", "unknown")
                print(f"\n{step_name.upper()}: {status}")
                for key, value in step_data.items():
                    if key != "status":
                        print(f"  {key}: {value}")

        if "artifacts" in results:
            print("\nARTIFACTS:")
            for artifact_name, path in results["artifacts"].items():
                if isinstance(path, dict):
                    for sub_name, sub_path in path.items():
                        print(f"  {artifact_name}/{sub_name}: {sub_path}")
                else:
                    print(f"  {artifact_name}: {path}")

        if "duration_seconds" in results:
            print(f"\nDuration: {results['duration_seconds']:.1f}s")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


def run_predict_only(args: argparse.Namespace) -> int:
    """Refresh predictions with existing model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Refreshing predictions for user '{args.username}'")

    config = PipelineConfig(
        storage_config=ArtifactStorageConfig(
            environment=args.environment,
        ),
    )

    pipeline = CollectionPipeline(args.username, config)

    try:
        paths = pipeline.refresh_predictions_only(
            model_version=args.model_version,
        )

        print("\nPredictions refreshed successfully!")
        print("\nArtifacts:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

        return 0

    except Exception as e:
        logger.error(f"Prediction refresh failed: {e}")
        return 1


def show_status(args: argparse.Namespace) -> int:
    """Show pipeline artifact status."""
    from src.collection.collection_artifact_storage import CollectionArtifactStorage

    storage = CollectionArtifactStorage(
        args.username,
        ArtifactStorageConfig(environment=args.environment),
    )

    status = storage.get_artifact_status()

    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"\nPipeline Status for: {status['username']}")
        print(f"Base path: {status['base_path']}")
        print(f"\nModel versions: {status['model_versions']}")
        if status.get('latest_model_version'):
            print(f"Latest model: v{status['latest_model_version']}")

        print("\nArtifacts:")
        for name, info in status['artifacts'].items():
            exists = "✓" if info['exists'] else "✗"
            print(f"  [{exists}] {name}")

    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Setup logging
    log_level = logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO
    setup_logging(log_level)

    # Dispatch to command handler
    if args.command == "run":
        return run_full_pipeline(args)
    elif args.command == "predict":
        return run_predict_only(args)
    elif args.command == "status":
        return show_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
