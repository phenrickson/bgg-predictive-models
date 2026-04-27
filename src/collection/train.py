"""Train one candidate against the canonical splits.

Looks up ``--candidate`` by name in ``config.collections.candidates``,
combines it with the named outcome, and runs
:func:`src.collection.candidates.train_candidate`. Splits must already
exist (run ``uv run python -m src.collection.split`` first).

Stdout: a single JSON line with ``version``, ``splits_version``,
``val_metrics``, ``test_metrics`` so callers can parse the result.

Known limitation: concurrent invocations against the same candidate may
race on the next candidate version.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from src.collection.candidates import load_candidates, train_candidate
from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.collection.outcomes import load_outcomes
from src.utils.config import load_config
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
        "--splits-version",
        type=int,
        default=None,
        help="Canonical splits version (default: latest)",
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

    project_config = load_config()
    raw_config = project_config.raw_config

    outcomes = load_outcomes(raw_config)
    if args.outcome not in outcomes:
        print(
            f"Unknown outcome {args.outcome!r}. Available: {sorted(outcomes)}",
            file=sys.stderr,
        )
        return 1
    outcome = outcomes[args.outcome]

    candidates = load_candidates(raw_config)
    if args.candidate not in candidates:
        print(
            f"Unknown candidate {args.candidate!r}. Available: {sorted(candidates)}",
            file=sys.stderr,
        )
        return 1
    candidate = candidates[args.candidate]

    storage = CollectionArtifactStorage(
        args.username,
        local_root=args.local_root,
        environment=args.environment,
    )

    result = train_candidate(
        candidate, outcome, storage, splits_version=args.splits_version
    )

    print(
        json.dumps(
            {
                "candidate": candidate.name,
                "outcome": args.outcome,
                "version": result.version,
                "splits_version": result.splits_version,
                "threshold": result.threshold,
                "val_metrics": result.val_metrics,
                "test_metrics": result.test_metrics,
                "n_train": result.train_n,
                "n_val": result.val_n,
                "n_test": result.test_n,
                "artifact_dir": result.artifact_dir,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
