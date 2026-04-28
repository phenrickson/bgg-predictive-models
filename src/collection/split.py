"""Persist canonical train/val/test splits for one outcome.

Wraps the same processor + splitter pipeline used by
:class:`src.collection.collection_pipeline.CollectionPipeline`, but writes
splits only — does not train a model. The resulting
``{outcome}/_splits/v{N}/`` is the input to candidate training runs, which
share these splits for honest comparison.

Known limitation: concurrent invocations may race on the next splits
version.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from src.collection.collection_artifact_storage import CollectionArtifactStorage
from src.collection.collection_processor import CollectionProcessor
from src.collection.collection_split import (
    ClassificationSplitConfig,
    CollectionSplitter,
    RegressionSplitConfig,
)
from src.collection.outcomes import apply_outcome, load_outcomes
from src.data.loader import BGGDataLoader
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
        "--local-root",
        default="models/collections",
        help="Storage root (default: models/collections)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging()

    project_config = load_config()
    bq_config = project_config.get_bigquery_config()

    outcomes = load_outcomes(project_config.raw_config)
    if args.outcome not in outcomes:
        print(
            f"Unknown outcome {args.outcome!r}. Available: {sorted(outcomes)}",
            file=sys.stderr,
        )
        return 1
    outcome = outcomes[args.outcome]

    storage = CollectionArtifactStorage(
        args.username,
        local_root=args.local_root,
        environment=args.environment,
    )

    processor = CollectionProcessor(
        config=bq_config,
        environment=args.environment,
    )
    processed = processor.process(args.username)
    if processed is None:
        print(
            f"No stored collection for user {args.username!r} in environment "
            f"{args.environment!r}. Refresh first via "
            f"`make refresh-collection USERNAME={args.username}` or "
            f"`uv run python -m src.collection.cli run --username {args.username}`.",
            file=sys.stderr,
        )
        return 1

    universe_df = BGGDataLoader(bq_config).load_features(
        use_predicted_complexity=False,
        use_embeddings=False,
    )

    # Universe (features) ⟕ collection (status). Every game in the
    # universe gets the user's status if any.
    joined = universe_df.join(processed, on="game_id", how="left")
    labeled = apply_outcome(joined, outcome)

    splitter = CollectionSplitter(
        classification_config=ClassificationSplitConfig(),
        regression_config=RegressionSplitConfig(),
    )
    train_df, val_df, test_df = splitter.split(labeled, outcome)

    paths = storage.save_canonical_splits(args.outcome, train_df, val_df, test_df)
    print(f"splits_version: {paths['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
