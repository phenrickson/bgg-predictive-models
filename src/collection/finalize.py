"""Refit a candidate on train+val+test through ``finalize_through``.

Standalone step. ``train`` produces a pipeline tuned and evaluated against
time-based splits (val/test held out). Once a candidate is the chosen
production model, ``finalize`` reloads its hyperparameters, refits on the
union of all splits filtered to ``year_published <= finalize_through``, and
saves the result alongside the original ``model.pkl``.

Stdout: one JSON line with ``candidate``, ``version``, ``finalize_through``,
``finalized_path``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

import polars as pl

from src.collection.candidates import load_candidates
from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.collection.collection_model import CollectionModel
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
        "--version",
        type=int,
        default=None,
        help="Candidate version to finalize (default: latest)",
    )
    p.add_argument(
        "--finalize-through",
        type=int,
        default=None,
        help=(
            "Last year (inclusive) for the refit. "
            "Defaults to collections.finalize_through in config.yaml."
        ),
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

    finalize_through = args.finalize_through
    if finalize_through is None:
        finalize_through = (raw_config.get("collections") or {}).get("finalize_through")
    if finalize_through is None:
        print(
            "finalize_through is required. Set --finalize-through or "
            "collections.finalize_through in config.yaml.",
            file=sys.stderr,
        )
        return 1
    finalize_through = int(finalize_through)

    storage = CollectionArtifactStorage(
        args.username,
        local_root=args.local_root,
        environment=args.environment,
    )

    pipeline, registration, threshold = storage.load_candidate_run(
        args.outcome, candidate.name, version=args.version
    )
    version = registration["version"]

    splits_version = registration.get("splits_version")
    if splits_version is None:
        print(
            f"Run {args.outcome}/{candidate.name}/v{version} has no "
            f"splits_version; cannot reload data to refit on.",
            file=sys.stderr,
        )
        return 1
    splits = storage.load_canonical_splits(args.outcome, version=splits_version)
    if splits is None:
        print(
            f"Splits v{splits_version} for outcome {args.outcome!r} not "
            f"found; cannot finalize.",
            file=sys.stderr,
        )
        return 1

    union = pl.concat(
        [splits["train"], splits["validation"], splits["test"]],
        how="vertical_relaxed",
    )

    # Reuse the candidate's classification/regression configs to
    # reconstruct a CollectionModel, then attach the loaded pipeline so
    # finalize() can read its hyperparams.
    model = CollectionModel(
        username=args.username,
        outcome=outcome,
        classification_config=candidate.classification_config,
        regression_config=candidate.regression_config,
    )
    model.fitted_pipeline = pipeline
    model.threshold = threshold
    model.finalize(
        union,
        finalize_through=finalize_through,
        downsample_ratio=candidate.downsample_negatives_ratio,
        protect_min_ratings=(
            candidate.downsample_protect_min_ratings
            if candidate.downsample_protect_min_ratings is not None
            else 25
        ),
    )

    finalized_path = storage.save_finalized_pipeline(
        outcome=args.outcome,
        candidate=candidate.name,
        version=version,
        pipeline=model.finalized_pipeline,
        finalize_through=finalize_through,
    )

    print(
        json.dumps(
            {
                "candidate": candidate.name,
                "outcome": args.outcome,
                "version": version,
                "finalize_through": finalize_through,
                "n_finalized": int(
                    union.filter(pl.col("year_published") <= finalize_through).height
                ),
                "finalized_path": finalized_path,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
