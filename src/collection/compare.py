"""Print or write a comparison table over saved candidate runs.

Wraps :func:`src.collection.candidate_comparison.load_candidate_runs` and
:func:`compare_runs`. With no ``--out``, prints the table to stdout. With
``--out path.csv`` or ``--out path.parquet``, writes the table to disk.

Comparison enforces that all runs share the same canonical splits version;
mixed-splits comparisons raise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import polars as pl

from src.collection.candidate_comparison import compare_runs, load_candidate_runs
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
        "--candidates",
        default=None,
        help="Comma-separated candidate names (default: all)",
    )
    p.add_argument(
        "--versions",
        default="latest",
        choices=["latest", "all"],
        help="Which versions to load (default: latest)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output file (.csv or .parquet); default: stdout",
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

    candidate_names: Optional[List[str]] = None
    if args.candidates:
        candidate_names = [c.strip() for c in args.candidates.split(",") if c.strip()]

    storage = CollectionArtifactStorage(
        args.username,
        local_root=args.local_root,
        environment=args.environment,
    )
    runs = load_candidate_runs(
        storage,
        args.outcome,
        candidate_names=candidate_names,
        versions=args.versions,
    )
    df = compare_runs(runs)

    if args.out:
        out_path = Path(args.out)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            df.write_csv(out_path)
        elif suffix == ".parquet":
            df.write_parquet(out_path)
        else:
            print(
                f"--out must end in .csv or .parquet (got {out_path.name!r})",
                file=sys.stderr,
            )
            return 1
        print(f"wrote {out_path} ({df.height} rows)", file=sys.stderr)
    else:
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            print(df)

    print(
        f"compared {len(runs)} candidates, {df.height} rows", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
