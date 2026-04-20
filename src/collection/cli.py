"""Command-line interface for user collection modeling pipeline."""

import argparse
import json
import logging
import sys

from src.utils.logging import setup_logging
from src.collection.collection_pipeline import CollectionPipeline, PipelineConfig
from src.collection.collection_artifact_storage import ArtifactStorageConfig
from src.collection.collection_split import ClassificationSplitConfig
from src.collection.collection_model import ClassificationModelConfig
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
        "--outcome",
        action="append",
        default=None,
        help="Restrict training to this outcome (repeatable). If omitted, trains all outcomes.",
    )
    run_parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=["lightgbm", "catboost", "logistic"],
        help="Classification model type (default: lightgbm). Regression uses lightgbm.",
    )
    run_parser.add_argument(
        "--train-through",
        type=int,
        default=None,
        help=(
            "Last year to include in training data (inclusive). If set, classification "
            "splits switch to time_based mode."
        ),
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
        help="Refresh predictions with existing models",
    )
    predict_parser.add_argument(
        "--outcome",
        action="append",
        default=None,
        help="Restrict refresh to this outcome (repeatable). If omitted, refreshes all.",
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
    """Run the full pipeline — trains one model per outcome."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running full pipeline for user '{args.username}'")

    split_mode = "time_based" if args.train_through is not None else "stratified_random"
    classification_split = ClassificationSplitConfig(
        split_mode=split_mode,
        train_through=args.train_through,
        negative_sampling_ratio=args.negative_ratio,
        negative_sampling_strategy=args.sampling_strategy,
    )

    config = PipelineConfig(
        storage_config=ArtifactStorageConfig(environment=args.environment),
        classification_split_config=classification_split,
        classification_model_config=ClassificationModelConfig(
            model_type=args.model_type
        ),
    )

    pipeline = CollectionPipeline(args.username, config)

    try:
        results = pipeline.run_full_pipeline(
            refresh_collection=not args.no_refresh,
            outcome_filter=args.outcome,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print(f"Pipeline completed for user: {args.username}")
    print(f"Collection rows: {results.get('collection_rows', 0)}")
    print("=" * 60)

    outcomes = results.get("outcomes", {})
    for name, outcome_result in outcomes.items():
        print(f"\n[{name}]")
        if "error" in outcome_result:
            print(f"  FAILED: {outcome_result['error']}")
            continue
        print(f"  task: {outcome_result['task']}")
        print(f"  version: v{outcome_result['version']}")
        sizes = outcome_result.get("split_sizes", {})
        print(
            f"  splits: train={sizes.get('train')} "
            f"val={sizes.get('val')} test={sizes.get('test')}"
        )
        if outcome_result.get("threshold") is not None:
            print(f"  threshold: {outcome_result['threshold']:.3f}")
        print("  metrics:")
        for k, v in outcome_result.get("metrics", {}).items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    if "duration_seconds" in results:
        print(f"\nDuration: {results['duration_seconds']:.1f}s")

    any_error = any("error" in r for r in outcomes.values())
    return 1 if any_error else 0


def run_predict_only(args: argparse.Namespace) -> int:
    """Refresh predictions for the latest model(s) per outcome."""
    logger = logging.getLogger(__name__)
    logger.info(f"Refreshing predictions for user '{args.username}'")

    config = PipelineConfig(
        storage_config=ArtifactStorageConfig(environment=args.environment),
    )

    pipeline = CollectionPipeline(args.username, config)

    try:
        results = pipeline.refresh_predictions_only(outcome_filter=args.outcome)
    except Exception as e:
        logger.error(f"Prediction refresh failed: {e}")
        return 1

    if not results:
        print("\nNo trained models found to refresh.")
        return 0

    print("\nPredictions refreshed:")
    for name, info in results.items():
        print(f"  {name} v{info['version']}: {info['rows']} rows")

    return 0


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

        collection_mark = "+" if status.get("collection_exists") else "-"
        print(f"\n  [{collection_mark}] collection/latest.parquet")

        outcomes = status.get("outcomes", {})
        if outcomes:
            print("\nOutcomes:")
            for outcome, info in outcomes.items():
                latest = info.get("latest_version")
                versions = info.get("versions", [])
                version_list = ", ".join(f"v{v}" for v in versions)
                print(f"  {outcome}:")
                print(f"    latest version : {f'v{latest}' if latest is not None else 'none'}")
                print(f"    versions       : [{version_list}]")
        else:
            print("\nNo outcome artifacts found.")

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
