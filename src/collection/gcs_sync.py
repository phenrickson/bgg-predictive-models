"""Prefix-level GCS sync for the collection-training Cloud Run job.

The `collections` image has `google-cloud-storage` but not `gsutil`, so
the job's gs:// boundary uses this instead of `gsutil rsync`. A focused
I/O helper, unit-tested with a mocked storage client (mirrors the
`from google.cloud import storage` pattern in
`src/utils/experiment_loader.py`).

Semantics that matter for correctness:
- `download_prefix` treats an absent/empty prefix as first-run (returns
  0, no error, no files). Real client errors propagate (raise) — a
  transient/real GCS error must NOT be silently treated as first-run.
- `upload_prefix` hard-fails on error: the trained model MUST reach GCS
  or the job must fail (mirrors the old unguarded `gsutil` up-rsync).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from google.cloud import storage

logger = logging.getLogger("src.collection.gcs_sync")


def download_prefix(bucket: str, prefix: str, local_dir: str) -> int:
    """Download every blob under ``prefix/`` into ``local_dir``.

    The sub-path after ``prefix`` is preserved. Returns the number of
    objects downloaded. An empty prefix (first run) returns 0 without
    error; real client errors propagate.
    """
    client = storage.Client()
    bkt = client.bucket(bucket)
    norm = prefix.rstrip("/") + "/"
    count = 0
    for blob in bkt.list_blobs(prefix=norm):
        rel = blob.name[len(norm):]
        if not rel:  # the "directory" placeholder blob, if any
            continue
        dest = Path(local_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))
        count += 1
    return count


def upload_prefix(bucket: str, prefix: str, local_dir: str) -> int:
    """Upload every file under ``local_dir`` (recursively) to ``bucket``
    under ``prefix/<relative path>``.

    Returns the number of objects uploaded. Raises on error (the
    up-sync must hard-fail).
    """
    client = storage.Client()
    bkt = client.bucket(bucket)
    norm = prefix.rstrip("/")
    root = Path(local_dir)
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            bkt.blob(f"{norm}/{rel}").upload_from_filename(str(path))
            count += 1
    return count


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("download", "up"):
        sp = sub.add_parser(name)
        sp.add_argument("--bucket", required=True)
        sp.add_argument("--prefix", required=True)
        sp.add_argument("--local-dir", required=True)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _build_parser().parse_args(argv)
    fn = download_prefix if args.command == "download" else upload_prefix
    try:
        count = fn(args.bucket, args.prefix, args.local_dir)
    except Exception as exc:  # hard-fail: surface and exit non-zero
        logger.error(
            "gcs_sync %s failed for gs://%s/%s: %s",
            args.command, args.bucket, args.prefix, exc,
        )
        return 1
    logger.info(
        "gcs_sync %s: %d object(s) gs://%s/%s <-> %s",
        args.command, count, args.bucket, args.prefix, args.local_dir,
    )
    print(count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
