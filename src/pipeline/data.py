"""Data retrieval entry point for outcome models.

Usage:
    uv run -m src.pipeline.data --model hurdle
    uv run -m src.pipeline.data --model hurdle --output data/hurdle_training.parquet
    uv run -m src.pipeline.data --model hurdle --use-embeddings
"""

import argparse
import logging
import sys

from src.models.outcomes.data import load_training_data
from src.models.outcomes.train import get_model_class
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Retrieve training data for outcome models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["hurdle", "complexity", "rating", "users_rated"],
        help="Model type to retrieve data for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for parquet file (if not provided, prints summary only)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=False,
        help="Include text embeddings as features",
    )
    parser.add_argument(
        "--complexity-predictions",
        type=str,
        default=None,
        help="Path to complexity predictions parquet (for rating/users_rated models)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local parquet file instead of BigQuery",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last year to include, inclusive (default: from config.years.training.test_through)",
    )

    args = parser.parse_args()

    # Load config defaults
    config = load_config()
    model_config = config.models.get(args.model)

    if args.end_year is None:
        args.end_year = config.years.training.test_through

    if not args.use_embeddings and model_config is not None:
        args.use_embeddings = model_config.use_embeddings

    return args


def main():
    args = parse_arguments()

    # Get model class to access data_config
    model_class = get_model_class(args.model)
    model = model_class()

    logger.info(f"Loading data for {args.model} model")
    logger.info(f"Data config: {model.data_config}")

    df = load_training_data(
        data_config=model.data_config,
        end_year=args.end_year,
        use_embeddings=args.use_embeddings,
        complexity_predictions_path=args.complexity_predictions,
        local_data_path=args.local_data,
    )

    logger.info(f"Retrieved {len(df)} rows with {len(df.columns)} columns")
    logger.info(f"Columns: {df.columns}")
    logger.info(f"Year range: {df['year_published'].min()} - {df['year_published'].max()}")

    if args.output:
        df.write_parquet(args.output)
        logger.info(f"Saved to {args.output}")
    else:
        logger.info("No output path provided, printing first 5 rows:")
        print(df.head())


if __name__ == "__main__":
    main()
