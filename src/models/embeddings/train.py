"""CLI entry point for training embedding models."""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.utils.config import load_config

from .trainer import EmbeddingTrainer


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the training process."""
    # Get root logger and clear any existing handlers to avoid duplication
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    logger = logging.getLogger(__name__)

    if log_file:
        # Check if file handler already exists
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
            for h in logger.handlers
        )
        if not has_file_handler:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            )
            logger.addHandler(file_handler)

    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train embedding models for game representations"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["pca", "svd", "umap", "autoencoder", "vae"],
        help="Embedding algorithm to use (default: from config.yaml)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Target embedding dimension (default: from config.yaml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: {algorithm}-embeddings)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description of the experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/experiments",
        help="Output directory for experiment artifacts",
    )

    # Algorithm-specific arguments
    parser.add_argument(
        "--whiten",
        action="store_true",
        default=True,
        help="(PCA) Whether to whiten the output",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=5,
        help="(SVD) Number of iterations",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="(UMAP) Number of neighbors",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="(UMAP) Minimum distance",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="(UMAP) Distance metric",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="(Autoencoder) Number of training epochs (default: from config.yaml)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="(Autoencoder) Batch size (default: from config.yaml)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="(Autoencoder) Learning rate (default: from config.yaml)",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=None,
        help="Minimum users_rated for training data (default: from config.yaml)",
    )

    return parser.parse_args()


def get_algorithm_params(args: argparse.Namespace) -> dict:
    """Extract algorithm-specific parameters from args.

    Only includes parameters that were explicitly set (not None).
    """
    params = {}
    if args.algorithm == "pca":
        params = {"whiten": args.whiten}
    elif args.algorithm == "svd":
        params = {"n_iter": args.n_iter}
    elif args.algorithm == "umap":
        params = {
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": args.metric,
        }
    elif args.algorithm in ("autoencoder", "vae"):
        params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }
    # Filter out None values so config can provide defaults
    return {k: v for k, v in params.items() if v is not None}


def main():
    """Main entry point for embedding training."""
    args = parse_arguments()
    logger = setup_logging()

    # Load config to get default values
    config = load_config()

    # Resolve algorithm (CLI overrides config)
    if args.algorithm:
        algorithm = args.algorithm
    elif config.embeddings:
        algorithm = config.embeddings.algorithm
    else:
        algorithm = "umap"

    # Resolve embedding dimension (CLI overrides config)
    if args.embedding_dim:
        embedding_dim = args.embedding_dim
    elif config.embeddings:
        embedding_dim = config.embeddings.embedding_dim
    else:
        embedding_dim = 64

    # Determine experiment name
    experiment_name = args.experiment or f"{algorithm}-embeddings"

    # Resolve min_ratings (CLI overrides config)
    if args.min_ratings is not None:
        min_ratings = args.min_ratings
    elif config.embeddings:
        min_ratings = config.embeddings.min_ratings
    else:
        min_ratings = 25

    # Get algorithm parameters (from args or config)
    # Update args.algorithm for get_algorithm_params
    args.algorithm = algorithm
    algorithm_params = get_algorithm_params(args)

    # Override with config values if not specified via CLI
    if config.embeddings:
        config_params = config.embeddings.get_algorithm_params(algorithm)
        for key, value in config_params.items():
            if key not in algorithm_params or algorithm_params[key] is None:
                algorithm_params[key] = value

    logger.info(f"Training {algorithm} embeddings")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Min ratings for training: {min_ratings}")
    logger.info(f"Algorithm parameters: {algorithm_params}")

    # Create trainer and run training
    trainer = EmbeddingTrainer(config=config, output_dir=args.output_dir)

    embedding_model, preprocessor, metrics = trainer.train(
        algorithm=algorithm,
        embedding_dim=embedding_dim,
        experiment_name=experiment_name,
        algorithm_params=algorithm_params,
        description=args.description,
        min_ratings=min_ratings,
    )

    # Log summary
    logger.info("Training complete!")
    logger.info("Metrics summary:")
    for dataset, dataset_metrics in metrics.items():
        logger.info(f"  {dataset}:")
        for metric_name, value in dataset_metrics.items():
            if isinstance(value, float):
                logger.info(f"    {metric_name}: {value:.4f}")
            elif isinstance(value, list) and len(value) <= 5:
                logger.info(f"    {metric_name}: {value}")


if __name__ == "__main__":
    main()
