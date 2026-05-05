"""CLI driver: shells out to `quarto render` per (user, outcome).

Usage:
    uv run python -m reports.render --username phenrickson --outcome own
    uv run python -m reports.render --username phenrickson --outcome own \
        --source gs://bgg_reports/collections-artifacts/
    uv run python -m reports.render --all-users --source gs://...

Environment:
    BGG_REPORTS_OFFLINE=1
        Stub out BQ-backed fetchers (collection snapshot, games metadata,
        upcoming predictions) with empty DataFrames. Used by the test
        suite and by local renders without GCP creds.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("reports.render")


def _install_offline_stubs() -> None:
    """When BGG_REPORTS_OFFLINE=1, replace BQ-backed fetchers with empty
    DataFrame returns so renders work without GCP creds."""
    import polars as pl

    from src.reports import collection_data

    empty = pl.DataFrame()
    collection_data._fetch_collection_snapshot = lambda username: empty
    collection_data._fetch_games_metadata = lambda: empty
    collection_data._fetch_upcoming_predictions = lambda u, o: empty


def _list_users(source: str) -> list[str]:
    """Discover users by listing the artifact root."""
    if source == "local":
        source = "models/collections"
    if source.startswith("gs://"):
        import fsspec

        fs = fsspec.filesystem("gs")
        prefix = source.rstrip("/").removeprefix("gs://")
        return [Path(p).name for p in fs.ls(prefix) if fs.isdir(p)]
    root = Path(source)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _render_one(
    username: str,
    outcome: str,
    source: str,
    candidate: str | None,
    output_dir: Path,
) -> int:
    """Run quarto render for one (user, outcome). Returns the process
    exit code; 0 = success."""
    project_root = Path(__file__).resolve().parents[1]
    qmd_path = Path(__file__).parent / "collection_report.qmd"
    # Render with a fixed name; move into output_dir after. We pass a
    # relative `--output` (no --output-dir) so the html lands beside the
    # qmd alongside its `_files` support directory — Quarto's bundler
    # gets confused when --output-dir is outside the qmd directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_name = f"{username}.html"
    cmd = [
        "quarto",
        "render",
        str(qmd_path),
        "--output",
        rendered_name,
    ]
    logger.info("Rendering: %s", " ".join(cmd))

    # Quarto resolves the python kernel via QUARTO_PYTHON. Point it at
    # the uv-managed venv so it inherits our deps. The qmd setup chunk
    # reads its params from BGG_REPORT_* environment variables.
    env = os.environ.copy()
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        env["QUARTO_PYTHON"] = str(venv_python)
    env["BGG_PROJECT_ROOT"] = str(project_root)
    env["BGG_REPORT_USERNAME"] = username
    env["BGG_REPORT_OUTCOME"] = outcome
    env["BGG_REPORT_SOURCE"] = source
    if candidate:
        env["BGG_REPORT_CANDIDATE"] = candidate
    else:
        env.pop("BGG_REPORT_CANDIDATE", None)
    proc = subprocess.run(cmd, env=env, cwd=qmd_path.parent)
    if proc.returncode != 0:
        return proc.returncode

    rendered = qmd_path.parent / rendered_name
    if not rendered.exists():
        logger.error("Quarto reported success but %s is missing", rendered)
        return 1
    target = output_dir / rendered_name
    rendered.replace(target)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", help="BGG username (omit with --all-users)")
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--outcome", default="own")
    parser.add_argument("--source", default="local")
    parser.add_argument("--candidate", default=None)
    parser.add_argument(
        "--output-dir",
        default="reports/_output",
        help="Directory to write rendered HTML",
    )
    args = parser.parse_args(argv)

    if os.environ.get("BGG_REPORTS_OFFLINE") == "1":
        _install_offline_stubs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_users and args.username:
        parser.error("Pass --username or --all-users, not both")
    if not args.all_users and not args.username:
        parser.error("Pass --username or --all-users")

    if args.all_users:
        users = _list_users(args.source)
        if not users:
            logger.error("No users found under source=%s", args.source)
            return 1
    else:
        users = [args.username]

    failures: list[str] = []
    for username in users:
        rc = _render_one(
            username=username,
            outcome=args.outcome,
            source=args.source,
            candidate=args.candidate,
            output_dir=output_dir,
        )
        if rc != 0:
            logger.error("Render failed for %s (rc=%s)", username, rc)
            failures.append(username)

    if failures:
        logger.error("Failed users: %s", ", ".join(failures))
        return 1
    logger.info("Rendered %d user(s) successfully", len(users))
    return 0


if __name__ == "__main__":
    sys.exit(main())
