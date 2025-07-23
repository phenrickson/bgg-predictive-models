import os
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Dict

from google.cloud import storage
import google.cloud.exceptions
import logging
import yaml
from dotenv import load_dotenv
import pathspec

# Load environment variables from .env file
load_dotenv()


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file's contents.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal hash of the file contents
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sync_experiments_to_gcs(
    local_dir: str = "models/experiments",
    bucket_name: Optional[str] = None,
    base_prefix: str = "models/experiments",
    config_path: Optional[str] = "config/bigquery.yaml",
    create_bucket: bool = True,
    location: str = "US",
    download: bool = False,
    gitignore_path: Optional[str] = ".gitignore",
):
    """
    Sync experiments directory with Google Cloud Storage.
    Can upload changed files or download missing files.

    Args:
        local_dir: Local directory to sync
        bucket_name: GCS bucket name. If None, uses the default from config.
        base_prefix: Base prefix in the bucket for storing experiments
        config_path: Path to configuration file with default bucket info
        create_bucket: Whether to create the bucket if it doesn't exist
        location: GCS location for bucket creation
        download: Whether to download missing files from cloud
        gitignore_path: Path to .gitignore file for filtering files
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

    # Ensure local directory exists
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.error(f"Local directory {local_dir} does not exist")
        return

    # Read .gitignore patterns
    gitignore_spec = None
    if gitignore_path and os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            gitignore_spec = pathspec.PathSpec.from_lines("gitignore", f)

    if download:
        # Track download progress
        total_files = 0
        downloaded_files = 0
        skipped_files = 0

        # List all blobs in the bucket with the specified prefix
        blobs = bucket.list_blobs(prefix=base_prefix)

        for blob in blobs:
            # Remove base prefix from blob name to get relative path
            relative_path = blob.name.replace(f"{base_prefix}/", "")
            local_file_path = local_path / relative_path

            # Skip files matching .gitignore patterns
            if gitignore_spec and gitignore_spec.match_file(relative_path):
                logger.info(f"Skipped by .gitignore: {relative_path}")
                continue

            # Skip if not a file (e.g., directory placeholders)
            if relative_path.endswith("/") or relative_path.endswith(".DS_Store"):
                logger.info(f"Skipping system file: {relative_path}")
                continue

            total_files += 1

            try:
                # Create parent directories if they don't exist
                local_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Check if file exists locally
                if not local_file_path.exists():
                    # Download missing file
                    blob.download_to_filename(str(local_file_path))
                    downloaded_files += 1
                    logger.info(f"Downloaded: {relative_path}")
                else:
                    skipped_files += 1
                    logger.info(f"Skipped (already exists): {relative_path}")

            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")

        # Summary log for download
        logger.info(
            f"Download sync complete. "
            f"{downloaded_files} files downloaded, "
            f"{skipped_files} files skipped, "
            f"total {total_files} files processed from bucket {bucket_name}"
        )
    else:
        # Original upload logic
        total_files = 0
        uploaded_files = 0
        skipped_files = 0

        # Walk through local directory
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                # Construct relative path
                relative_path = file_path.relative_to(local_path)

                # # Skip files matching .gitignore patterns
                # if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
                #     logger.info(f"Skipped by .gitignore: {relative_path}")
                #     continue

                total_files += 1

                # Construct blob path, preserving directory structure
                blob_path = f"{base_prefix}/{relative_path}"

                try:
                    # Check if blob exists
                    blob = bucket.blob(str(blob_path))

                    # Calculate local file hash
                    local_file_hash = calculate_file_hash(file_path)

                    # Try to get existing blob metadata
                    try:
                        existing_hash = (
                            blob.metadata.get("file_hash") if blob.metadata else None
                        )
                    except Exception:
                        existing_hash = None

                    # Upload if hash is different or blob doesn't exist
                    if existing_hash != local_file_hash:
                        # Upload file
                        blob.upload_from_filename(str(file_path))

                        # Set metadata after upload
                        blob.metadata = {"file_hash": local_file_hash}
                        blob.patch()

                        uploaded_files += 1
                        logger.info(f"Uploaded: {relative_path}")
                    else:
                        skipped_files += 1
                        logger.info(f"Skipped (unchanged): {relative_path}")

                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")

        # Summary log
        logger.info(
            f"Sync complete. "
            f"{uploaded_files} files uploaded, "
            f"{skipped_files} files skipped, "
            f"total {total_files} files processed in bucket {bucket_name}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Sync experiments to/from Google Cloud Storage"
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
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing files from cloud to local",
    )

    args = parser.parse_args()

    sync_experiments_to_gcs(
        local_dir=args.local_dir,
        bucket_name=args.bucket_name,
        base_prefix=args.base_prefix,
        config_path=args.config_path,
        create_bucket=args.create_bucket,
        location=args.location,
        download=args.download,
    )


if __name__ == "__main__":
    main()
