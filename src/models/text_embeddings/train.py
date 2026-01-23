"""CLI entry point for training text embedding models."""

import argparse
import logging

from src.models.training import load_data
from src.utils.config import load_config

from .trainer import TextEmbeddingTrainer


def setup_logging() -> logging.Logger:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train text embedding models from game descriptions"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["pmi", "word2vec"],
        help="Algorithm to use (default: from config.yaml)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dimension (default: from config.yaml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: from config.yaml)",
    )
    parser.add_argument(
        "--document-method",
        type=str,
        default=None,
        choices=["mean", "tfidf", "sif"],
        help="Method for aggregating word vectors (default: from config.yaml)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local parquet file with game features",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Output directory for experiment artifacts",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    logger = setup_logging()

    config = load_config()

    # Load data (use train_end year to fit embeddings on training data only)
    logger.info("Loading game data...")
    df = load_data(
        local_data_path=args.local_data,
        end_train_year=config.years.train_end,
    )

    # Get descriptions
    if "description" not in df.columns:
        raise ValueError("Data does not contain 'description' column")

    descriptions = df["description"].fill_null("").to_list()
    logger.info(f"Loaded {len(descriptions)} game descriptions")

    # Train
    trainer = TextEmbeddingTrainer(config=config, output_dir=args.output_dir)

    result = trainer.train(
        documents=descriptions,
        algorithm=args.algorithm,
        embedding_dim=args.embedding_dim,
        experiment_name=args.experiment,
        document_method=args.document_method,
    )

    logger.info("Training complete!")
    logger.info(f"  Experiment: {result['experiment_dir']}")
    logger.info(f"  Vocab size: {result['vocab_size']}")
    logger.info(f"  Embedding dim: {result['embedding_dim']}")


if __name__ == "__main__":
    main()
