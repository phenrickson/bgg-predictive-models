"""CLI for scoring games with text embeddings and uploading to BigQuery."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import polars as pl

from src.models.training import load_data
from src.utils.config import load_config

from .document import DocumentEmbedding
from .storage import BigQueryTextEmbeddingStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    config = load_config()
    default_model = f"text-embeddings-v{config.years.current}"

    parser = argparse.ArgumentParser(
        description="Generate text embeddings and upload to BigQuery"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Name of registered text embedding model (default: {default_model})",
    )
    parser.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Specific model version (default: latest)",
    )
    parser.add_argument(
        "--upload-to-bigquery",
        action="store_true",
        help="Upload embeddings to BigQuery",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save embeddings locally (optional)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Start year for games to score (default: all)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="End year for games to score (default: config score_end)",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local data parquet file (optional)",
    )

    return parser.parse_args()


def main():
    """Main entry point for scoring."""
    args = parse_arguments()
    config = load_config()

    # Import here to avoid circular imports
    from text_embeddings_service.registered_model import RegisteredTextEmbeddingModel

    # Load registered model from GCS
    logger.info(f"Loading registered model: {args.model}")
    registered = RegisteredTextEmbeddingModel()

    try:
        word_model, doc_model, registration = registered.load_registered_model(
            args.model, args.model_version
        )
    except ValueError as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    model_info = registration.get("model_info", {})
    version = registration["version"]
    logger.info(f"Loaded model {args.model} v{version}")
    logger.info(f"  Algorithm: {model_info.get('algorithm')}")
    logger.info(f"  Embedding dim: {model_info.get('embedding_dim')}")
    logger.info(f"  Document method: {model_info.get('document_method')}")
    logger.info(f"  Vocab size: {model_info.get('vocab_size')}")

    # Load games data
    end_year = args.end_year or config.years.score_end
    logger.info(f"Loading games data (up to year {end_year})...")

    if args.local_data:
        logger.info(f"Using local data from {args.local_data}")
        df = pl.read_parquet(args.local_data)
    else:
        df = load_data(end_train_year=end_year)

    # Filter by start year if specified
    if args.start_year and "year_published" in df.columns:
        df = df.filter(pl.col("year_published") >= args.start_year)
        logger.info(f"Filtered to games from {args.start_year} onwards")

    if "description" not in df.columns:
        logger.error("Data does not contain 'description' column")
        sys.exit(1)

    logger.info(f"Generating embeddings for {len(df)} games...")

    # Generate embeddings
    descriptions = df["description"].fill_null("").to_list()
    embeddings = doc_model.transform(descriptions)

    # Create output dataframe
    embeddings_df = pl.DataFrame({
        "game_id": df["game_id"],
        "name": df["name"] if "name" in df.columns else None,
        "embedding": [emb.tolist() for emb in embeddings],
    })

    logger.info(f"Generated {len(embeddings_df)} embeddings")

    # Save locally if requested
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.write_parquet(output_path)
        logger.info(f"Saved embeddings to {output_path}")

    # Upload to BigQuery if requested
    if args.upload_to_bigquery:
        logger.info("Uploading to BigQuery...")
        storage = BigQueryTextEmbeddingStorage(config)

        job_id = storage.upload_embeddings(
            embeddings_df=embeddings_df,
            model_name=args.model,
            model_version=version,
            algorithm=model_info.get("algorithm", "pmi"),
            embedding_dim=model_info.get("embedding_dim", 100),
            document_method=model_info.get("document_method"),
        )
        logger.info(f"Uploaded to BigQuery, job_id={job_id}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
