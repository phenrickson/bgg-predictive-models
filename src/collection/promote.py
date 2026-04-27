"""Promote a candidate run to the production-winner path.

Loads the named candidate (latest by default) and copies its pipeline +
threshold + registration into ``{outcome}/v{N}/``, the same path
:class:`src.collection.collection_pipeline.CollectionPipeline` writes.
The new production registration carries a ``promoted_from`` field
pointing back at the candidate run.

Known limitation: concurrent invocations may race on the next production
version.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.utils.logging import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--username", required=True, help="BGG username")
    p.add_argument(
        "--environment",
        default="dev",
        help="Storage environment (default: dev)",
    )
    p.add_argument(
        "--outcome",
        required=True,
        help="Outcome name from config.yaml (e.g. own, rating)",
    )
    p.add_argument(
        "--candidate",
        required=True,
        help="Candidate name from config.collections.candidates",
    )
    p.add_argument(
        "--version",
        default="latest",
        help="Candidate version to promote (default: latest)",
    )
    p.add_argument(
        "--local-root",
        default="models/collections",
        help="Storage root (default: models/collections)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging()

    storage = CollectionArtifactStorage(
        args.username,
        local_root=args.local_root,
        environment=args.environment,
    )

    if args.version == "latest":
        candidate_version = storage.latest_candidate_version(
            args.outcome, args.candidate
        )
        if candidate_version is None:
            print(
                f"No runs found for candidate {args.candidate!r} on outcome "
                f"{args.outcome!r}",
                file=sys.stderr,
            )
            return 1
    else:
        candidate_version = int(args.version)

    pipeline, registration, threshold = storage.load_candidate_run(
        args.outcome, args.candidate, version=candidate_version
    )

    metadata = {
        **registration,
        "promoted_from": {
            "candidate": args.candidate,
            "version": candidate_version,
        },
    }

    production_version = storage.next_version(args.outcome)
    storage.save_model(
        args.outcome,
        pipeline,
        metadata,
        threshold=threshold,
        version=production_version,
    )

    print(f"promoted_to: production v{production_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
