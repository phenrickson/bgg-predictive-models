import os
import argparse
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def get_file_info(path: Path, base_path: Path) -> Dict[str, str]:
    """
    Get file information including relative path and hash.

    Args:
        path: Path to the file
        base_path: Base path to calculate relative path from

    Returns:
        Dictionary with relative path and hash
    """
    relative_path = str(path.relative_to(base_path))
    file_hash = calculate_file_hash(path)
    return {"path": relative_path, "hash": file_hash}


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

    # Use bucket name as provided (should be environment-specific via GitHub environment)
    logger.info(f"Using bucket: {bucket_name}")

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

    # Get all local files and their hashes
    local_files = {}
    for file_path in local_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative_path = str(file_path.relative_to(local_path))

        # Skip files matching .gitignore patterns, except .pkl files
        if gitignore_spec and gitignore_spec.match_file(relative_path):
            if not relative_path.endswith(".pkl"):
                logger.info(f"Skipped by .gitignore: {relative_path}")
                continue
            else:
                logger.info(f"Including .pkl file despite gitignore: {relative_path}")

        # Skip system files
        if relative_path.endswith(".DS_Store"):
            logger.info(f"Skipping system file: {relative_path}")
            continue

        local_files[relative_path] = calculate_file_hash(file_path)

    # Get all remote files and their hashes
    remote_files = {}
    for blob in bucket.list_blobs(prefix=base_prefix):
        relative_path = blob.name.replace(f"{base_prefix}/", "")

        # Skip directory placeholders
        if relative_path.endswith("/"):
            logger.info(f"Skipping directory placeholder: {relative_path}")
            continue

        try:
            blob.reload()
            remote_files[relative_path] = (
                blob.metadata.get("file_hash") if blob.metadata else None
            )
        except Exception as e:
            logger.warning(f"Failed to get metadata for {relative_path}: {e}")
            remote_files[relative_path] = None

    total_files = 0
    processed_files = 0
    skipped_files = 0

    # Determine which files need transfer
    files_to_transfer: List[Tuple[str, str, bool]] = (
        []
    )  # [(relative_path, hash, is_new)]

    if download:
        # Find files that need downloading
        for relative_path, remote_hash in remote_files.items():
            total_files += 1
            local_file_path = local_path / relative_path

            # Skip if local file exists and hashes match
            if local_file_path.exists():
                local_hash = local_files.get(relative_path)
                if local_hash == remote_hash:
                    skipped_files += 1
                    logger.info(f"Skipped (unchanged): {relative_path}")
                    continue

            # Add to transfer list
            files_to_transfer.append(
                (relative_path, remote_hash, not local_file_path.exists())
            )

        # Create all necessary directories upfront
        all_dirs = {(local_path / path).parent for path, _, _ in files_to_transfer}
        for dir_path in all_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Download files in parallel
        def download_file(args) -> Tuple[str, bool]:
            relative_path, _, is_new = args
            try:
                blob = bucket.blob(f"{base_prefix}/{relative_path}")
                blob.download_to_filename(str(local_path / relative_path))
                logger.info(
                    f"{'Downloaded new' if is_new else 'Updated'}: {relative_path}"
                )
                return relative_path, True
            except Exception as e:
                logger.error(f"Failed to download {relative_path}: {e}")
                return relative_path, False

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(download_file, args) for args in files_to_transfer
            ]
            for future in as_completed(futures):
                _, success = future.result()
                if success:
                    processed_files += 1

    else:
        # Find files that need uploading
        for relative_path, local_hash in local_files.items():
            total_files += 1
            is_new = relative_path not in remote_files

            # Skip if remote file exists and hashes match
            if not is_new:
                remote_hash = remote_files[relative_path]
                if remote_hash == local_hash:
                    skipped_files += 1
                    logger.info(f"Skipped (unchanged): {relative_path}")
                    continue
            else:
                logger.info(f"New file detected: {relative_path}")

            # Add to transfer list
            files_to_transfer.append((relative_path, local_hash, is_new))

        # Upload files in parallel
        def upload_file(args) -> Tuple[str, bool]:
            relative_path, local_hash, is_new = args
            try:
                blob = bucket.blob(f"{base_prefix}/{relative_path}")
                blob.upload_from_filename(str(local_path / relative_path))
                blob.metadata = {"file_hash": local_hash}
                blob.patch()
                logger.info(
                    f"{'Uploaded new' if is_new else 'Updated'}: {relative_path}"
                )
                return relative_path, True
            except Exception as e:
                logger.error(f"Failed to process {relative_path}: {e}")
                return relative_path, False

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_file, args) for args in files_to_transfer]
            for future in as_completed(futures):
                _, success = future.result()
                if success:
                    processed_files += 1

    # Summary log
    operation = "Download" if download else "Upload"
    logger.info(
        f"{operation} sync complete. "
        f"{processed_files} files processed, "
        f"{skipped_files} files skipped, "
        f"total {total_files} files examined in bucket {bucket_name}"
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
