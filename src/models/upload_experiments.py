import os
import argparse
from pathlib import Path
from typing import Optional

from google.cloud import storage
import google.cloud.exceptions
import logging
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def sync_experiments_to_gcs(
    local_dir: str = "models/experiments",
    bucket_name: Optional[str] = None,
    base_prefix: str = "models/experiments",
    config_path: Optional[str] = "config/bigquery.yaml",
    create_bucket: bool = True,
    location: str = "US",
):
    """
    Sync local experiments directory to Google Cloud Storage.

    Args:
        local_dir: Local directory to sync
        bucket_name: GCS bucket name. If None, uses the default from config.
        base_prefix: Base prefix in the bucket for storing experiments
        config_path: Path to configuration file with default bucket info
        create_bucket: Whether to create the bucket if it doesn't exist
        location: GCS location for bucket creation
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)

    # If no bucket specified, try to read from config
    if bucket_name is None and config_path:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                bucket_name = config.get("storage", {}).get("bucket")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found")

    if not bucket_name:
        raise ValueError("No bucket name specified or found in configuration")

    # Initialize GCS client
    storage_client = storage.Client()

    # Check if bucket exists, create if not
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} already exists")
    except google.cloud.exceptions.NotFound:
        if create_bucket:
            logger.info(f"Creating bucket {bucket_name}")
            bucket = storage_client.create_bucket(bucket_name, location=location)
        else:
            raise ValueError(f"Bucket {bucket_name} does not exist")
    except Exception as e:
        raise ValueError(f"Error accessing bucket {bucket_name}: {e}")

    # Walk through local directory
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.error(f"Local directory {local_dir} does not exist")
        return

    # Count files and track upload progress
    total_files = 0
    uploaded_files = 0

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            total_files += 1

            # Construct blob path, preserving directory structure
            relative_path = file_path.relative_to(local_path)
            blob_path = f"{base_prefix}/{relative_path}"

            try:
                # Upload file
                blob = bucket.blob(str(blob_path))
                blob.upload_from_filename(str(file_path))
                uploaded_files += 1
                logger.info(f"Uploaded: {relative_path}")
            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {e}")

    # Summary log
    logger.info(
        f"Upload complete. {uploaded_files}/{total_files} files uploaded successfully to {bucket_name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sync experiments to Google Cloud Storage"
    )
    parser.add_argument(
        "--bucket-name",
        default=None,
        help="GCS bucket name. If not provided, uses default from config.",
    )
    parser.add_argument(
        "--local-dir", default="models/experiments", help="Local directory to sync"
    )
    parser.add_argument(
        "--base-prefix",
        default="models/experiments",
        help="Base prefix in the bucket for storing experiments",
    )
    parser.add_argument(
        "--config-path",
        default="config/bigquery.yaml",
        help="Path to configuration file with default bucket info",
    )
    parser.add_argument(
        "--create-bucket",
        action="store_true",
        help="Create the bucket if it does not exist",
    )
    parser.add_argument(
        "--location",
        default="US",
        help="GCS location for bucket creation",
    )

    args = parser.parse_args()

    sync_experiments_to_gcs(
        local_dir=args.local_dir,
        bucket_name=args.bucket_name,
        base_prefix=args.base_prefix,
        config_path=args.config_path,
        create_bucket=args.create_bucket,
        location=args.location,
    )


if __name__ == "__main__":
    main()
