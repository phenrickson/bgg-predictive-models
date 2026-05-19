"""Single in-process entrypoint: split -> train -> finalize -> register
for one user / one candidate. Pure lifecycle — no gsutil, no CI
dispatch (those are the Cloud Run job wrapper's and the workflow's
job). Reuses each existing module's `main(argv)->int` contract; does
not modify or duplicate their logic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from src.collection.split import main as _split_main
from src.collection.train import main as _train_main
from src.collection.finalize import main as _finalize_main
from services.collections.register_model import main as _register_main

logger = logging.getLogger("src.collection.train_model")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--username", required=True)
    p.add_argument("--outcome", default="own")
    p.add_argument("--candidate", default="logistic_row_norm")
    p.add_argument("--environment", default="dev")
    p.add_argument("--local-root", default="models/collections")
    p.add_argument("--finalize-through", default=None)
    p.add_argument("--description", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _build_parser().parse_args(argv)

    common = [
        "--username", args.username,
        "--outcome", args.outcome,
        "--environment", args.environment,
        "--local-root", args.local_root,
    ]

    # split does not accept --candidate (it persists splits for the
    # whole outcome, not per candidate), so it gets only the common args.
    split_argv = list(common)
    train_argv = common + ["--candidate", args.candidate]
    finalize_argv = common + ["--candidate", args.candidate]
    if args.finalize_through:
        finalize_argv += ["--finalize-through", args.finalize_through]
    description = (
        args.description
        or f"{args.candidate} for {args.username}/{args.outcome}"
    )
    register_argv = common + [
        "--candidate", args.candidate,
        "--description", description,
    ]

    stages = [
        ("split", _split_main, split_argv),
        ("train", _train_main, train_argv),
        ("finalize", _finalize_main, finalize_argv),
        ("register", _register_main, register_argv),
    ]

    for name, fn, stage_argv in stages:
        logger.info("=== %s ===", name)
        rc = fn(stage_argv)
        if rc != 0:
            logger.error("Stage %r failed (rc=%s); aborting.", name, rc)
            return rc
    logger.info(
        "train_model complete: %s/%s/%s",
        args.username, args.outcome, args.candidate,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
