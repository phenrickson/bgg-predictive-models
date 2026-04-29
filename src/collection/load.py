"""Fetch a user's collection from the BGG API and upsert into BigQuery.

Standalone entrypoint for the load step. Use this before `split` /
`train` / `sweep` for a user whose collection has not yet been
persisted.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from src.collection.collection_pipeline import fetch_and_persist
from src.utils.logging import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--username", required=True, help="BGG username")
    p.add_argument(
        "--environment",
        default="dev",
        help="Storage environment (default: dev)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging()

    try:
        rows = fetch_and_persist(args.username, args.environment)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"persisted: {rows} rows for {args.username!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
