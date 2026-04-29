"""CLI: list registered collection models in GCS for a user.

Walks ``{env}/services/collections/{username}/`` (optionally narrowed to one
outcome) and prints each registered version's metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List, Optional

from services.scoring.auth import (
    AuthenticationError,
    get_authenticated_storage_client,
)
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def verify_collection_models(
    *,
    username: str,
    outcome: Optional[str] = None,
    bucket_name: Optional[str] = None,
    environment_prefix: Optional[str] = None,
    project_id: Optional[str] = None,
) -> List[dict]:
    if bucket_name is None or environment_prefix is None:
        cfg = load_config()
        bucket_name = bucket_name or cfg.get_bucket_name()
        environment_prefix = environment_prefix or cfg.get_environment_prefix()

    try:
        client = get_authenticated_storage_client(project_id)
    except AuthenticationError as e:
        raise ValueError(f"Authentication failed: {e}")

    bucket = client.bucket(bucket_name)

    if outcome:
        prefix = f"{environment_prefix}/services/collections/{username}/{outcome}/"
    else:
        prefix = f"{environment_prefix}/services/collections/{username}/"

    registrations: List[dict] = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/registration.json"):
            registrations.append(json.loads(blob.download_as_text()))

    registrations.sort(key=lambda r: (r.get("outcome", ""), r.get("version", 0)))
    return registrations


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", required=True)
    p.add_argument("--outcome", help="Optional: narrow to one outcome")
    args = p.parse_args(argv)

    try:
        registrations = verify_collection_models(
            username=args.username, outcome=args.outcome
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if not registrations:
        scope = f"{args.username}/{args.outcome}" if args.outcome else args.username
        print(f"No registered collection models for {scope}")
        return 0

    by_outcome: dict = {}
    for r in registrations:
        by_outcome.setdefault(r.get("outcome", "?"), []).append(r)

    for outcome_name, regs in by_outcome.items():
        print(f"\n{outcome_name} ({len(regs)} version(s)):")
        for r in regs:
            src = r.get("source", {})
            print(
                f"  v{r['version']:<3} "
                f"candidate={src.get('candidate', '?'):<22} "
                f"kind={src.get('pipeline_kind', '?'):<10} "
                f"registered={r.get('registered_at', '?')}"
            )
            if r.get("description"):
                print(f"        description: {r['description']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
