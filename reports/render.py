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
import shutil
import subprocess
import sys
from pathlib import Path

from src.collection.collection_artifact_storage import slugify_username

# Load .env at import time so envvars (including BGG_REPORTS_OFFLINE,
# GOOGLE_APPLICATION_CREDENTIALS, etc.) are available to both this
# driver and the Quarto kernel it spawns.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

logger = logging.getLogger("reports.render")


# Maps --report to its qmd template. The predictions report keeps the
# friendly top-level URL; the model report lives under model/ so the
# two render pipelines never collide on a filename in the shared bundle.
_REPORTS = {
    "predictions": "predictions_report.qmd",
    "model": "model_report.qmd",
}


def _output_rel_path(report: str, username: str) -> Path:
    # Filenames are always the slug so they're filesystem/URL-safe and
    # match the slug-named keys in the user grid (which the index.qmd
    # composes from on-disk dir names). The real username flows through
    # to the qmd via the `username` Quarto param for display.
    slug = slugify_username(username)
    if report == "model":
        return Path("model") / f"{slug}.html"
    return Path(f"{slug}.html")


def _install_offline_stubs() -> None:
    """When BGG_REPORTS_OFFLINE=1, replace BQ-backed fetchers with empty
    DataFrame returns so renders work without GCP creds."""
    from src.reports import collection_data

    collection_data._fetch_collection_snapshot = (
        lambda username: collection_data.empty_offline_frame("collection")
    )
    collection_data._fetch_games_metadata = (
        lambda: collection_data.empty_offline_frame("games")
    )
    collection_data._fetch_upcoming_predictions = (
        lambda u, o: collection_data.empty_offline_frame("upcoming")
    )


def _list_users(source: str) -> list[str]:
    """Discover users by listing the artifact root.

    Directory names on disk are *slugified* (e.g. ``Watch_It_Played``); the
    real BGG username may differ (``"Watch It Played"``) and is recorded in
    each user's ``metadata.json``. Prefer that real username; fall back to
    the directory name when metadata is missing (older artifacts).
    """
    import json

    if source == "local":
        source = "models/collections"
    if source.startswith("gs://"):
        import fsspec

        fs = fsspec.filesystem("gs")
        prefix = source.rstrip("/").removeprefix("gs://")
        names: list[str] = []
        for p in fs.ls(prefix):
            if not fs.isdir(p):
                continue
            dir_name = Path(p).name
            meta_path = f"{p.rstrip('/')}/metadata.json"
            try:
                with fs.open(meta_path) as f:
                    real = json.load(f).get("username")
                names.append(real or dir_name)
            except (FileNotFoundError, OSError):
                names.append(dir_name)
        return sorted(names)
    root = Path(source)
    if not root.exists():
        return []
    names = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        meta = p / "metadata.json"
        if meta.exists():
            try:
                real = json.loads(meta.read_text()).get("username")
                names.append(real or p.name)
                continue
            except (json.JSONDecodeError, OSError):
                pass
        names.append(p.name)
    return names


def _render_one(
    username: str,
    report: str,
    outcome: str,
    source: str,
    candidate: str | None,
    output_dir: Path,
    fixture: bool = False,
) -> int:
    """Run quarto render for one (user, outcome). Returns the process
    exit code; 0 = success."""
    project_root = Path(__file__).resolve().parents[1]
    qmd_path = Path(__file__).parent / _REPORTS[report]
    # Render with a fixed name; move into output_dir after. We pass a
    # relative `--output` (no --output-dir) so the html lands beside the
    # qmd alongside its `_files` support directory — Quarto's bundler
    # gets confused when --output-dir is outside the qmd directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Quarto writes the html alongside the qmd with this name; we then
    # move it into output_dir at rel_out. Both must be slug-named so a
    # username with spaces (e.g. "Watch It Played") produces a clean
    # `Watch_It_Played.html` in the bundle.
    rendered_name = f"{slugify_username(username)}.html"
    rel_out = _output_rel_path(report, username)
    # Quarto params are passed via -P key=value. Booleans are accepted
    # as bare lowercase strings.
    cmd = [
        "quarto",
        "render",
        str(qmd_path),
        "--output",
        rendered_name,
        "-P", f"username={username}",
        "-P", f"outcome={outcome}",
        "-P", f"source={source}",
        "-P", f"candidate={candidate or ''}",
        "-P", f"fixture={'true' if fixture else 'false'}",
    ]
    logger.info("Rendering: %s", " ".join(cmd))

    # Quarto resolves the python kernel via QUARTO_PYTHON. Point it at
    # the uv-managed venv so it inherits our deps.
    env = os.environ.copy()
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        env["QUARTO_PYTHON"] = str(venv_python)
    env["BGG_PROJECT_ROOT"] = str(project_root)
    # If the user's GOOGLE_APPLICATION_CREDENTIALS is a relative path
    # (the convention in this repo), resolve it against the project
    # root so the kernel can find it from cwd=reports/.
    gac = env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not gac:
        # Fall back to the standard repo location.
        candidate_creds = project_root / "credentials" / "service-account-key.json"
        if candidate_creds.exists():
            env["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate_creds)
    elif not Path(gac).is_absolute():
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / gac)
    proc = subprocess.run(cmd, env=env, cwd=qmd_path.parent)
    if proc.returncode != 0:
        return proc.returncode

    rendered = qmd_path.parent / rendered_name
    if not rendered.exists():
        logger.error("Quarto reported success but %s is missing", rendered)
        return 1
    target = output_dir / rel_out
    target.parent.mkdir(parents=True, exist_ok=True)
    # shutil.move (not Path.replace/os.rename): in the container the
    # qmd renders onto the image FS (/app/reports) but output_dir is a
    # bind-mounted volume — a different device — so rename() raises
    # EXDEV. shutil.move falls back to copy+unlink across filesystems
    # and still does a fast rename when same-FS (local runs).
    if target.exists():
        target.unlink()
    shutil.move(str(rendered), str(target))
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
        "--report",
        required=True,
        choices=sorted(_REPORTS),
        help="Which report to render: predictions or model",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/_output",
        help="Directory to write rendered HTML",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help=(
            "Render against synthetic data (src.reports.fixtures) instead "
            "of loading real artifacts. Use for fast styling iteration."
        ),
    )
    args = parser.parse_args(argv)

    if args.fixture:
        # Fixture mode short-circuits BQ entirely; offline stubs match
        # the no-creds story.
        os.environ["BGG_REPORTS_OFFLINE"] = "1"

    if os.environ.get("BGG_REPORTS_OFFLINE") == "1":
        _install_offline_stubs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_users and args.username:
        parser.error("Pass --username or --all-users, not both")
    if args.fixture and not args.username:
        args.username = "fixture_user"
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
        # Pre-flight: confirm the user has finalized artifacts before
        # spinning up Quarto. Skipped in fixture mode (no real artifacts
        # are needed). Works against both local paths and gs:// URIs
        # via fsspec inside select_candidate.
        if not args.fixture:
            from src.reports.collection_data import (
                MissingArtifactsError,
                select_candidate,
            )

            if args.source == "local":
                root = str(
                    Path(__file__).resolve().parents[1] / "models" / "collections"
                )
            else:
                root = args.source
            try:
                select_candidate(
                    root,
                    username,
                    args.outcome,
                    candidate=args.candidate,
                )
            except MissingArtifactsError as exc:
                logger.error("%s", exc)
                failures.append(username)
                continue

        rc = _render_one(
            username=username,
            report=args.report,
            outcome=args.outcome,
            source=args.source,
            candidate=args.candidate,
            output_dir=output_dir,
            fixture=args.fixture,
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
