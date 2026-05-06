"""Generate per-user wrapper qmds under reports/collections/ from the
list of users with finalized artifacts in models/collections/.

Each wrapper is a thin file that just sets `params.username` and
`{{< include >}}`s the canonical collection_report.qmd. Quarto then
builds the website project (reports/_quarto.yml) which wires those
wrappers into the sidebar and produces a single navigable site under
reports/_site/.

Usage:
    uv run python -m reports.build_site
    uv run python -m reports.build_site --no-render   # just generate

Excludes:
    - fixture_user (sandbox-only)
    - any user passed via --exclude

This script is idempotent — running it again rewrites the same wrappers.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("reports.build_site")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
WRAPPERS_DIR = REPORTS_DIR / "collections"
ARTIFACTS_ROOT = PROJECT_ROOT / "models" / "collections"

# Always-excluded usernames (sandbox / synthetic).
ALWAYS_EXCLUDE = {"fixture_user"}

WRAPPER_TEMPLATE = """---
title: "{username}"
subtitle: "Collection Report"
params:
  username: {username}
  outcome: own
  source: local
  candidate: ""
  fixture: false
---

```{{python}}
#| tags: [parameters]
#| include: false
username = "{username}"
outcome = "own"
source = "local"
candidate = ""
fixture = False
```

```{{python}}
#| label: setup
#| include: false
import os, sys
_pr = os.environ.get("BGG_PROJECT_ROOT") or os.getcwd()
if _pr not in sys.path:
    sys.path.insert(0, _pr)

import polars as pl
import pandas as pd
from itables import show as itables_show

from src.reports.collection_data import load
from src.reports.tables import (
    build_status_lookup,
    build_topn_by_year_html,
    format_collection_table,
    format_eval_predictions,
    format_eval_table,
    format_model_details,
    format_predictions_with_images,
)
from src.collection.viz import (
    metrics_table,
    plot_collection_by_category_static,
    plot_collection_by_year_static,
    plot_feature_importance,
    plot_separation_static,
)

USERNAME = username
OUTCOME = outcome
SOURCE = source
CANDIDATE = candidate or None

if os.environ.get("BGG_REPORTS_OFFLINE") == "1":
    from src.reports import collection_data as _cd
    _empty = pl.DataFrame()
    _cd._fetch_collection_snapshot = lambda u: _empty
    _cd._fetch_games_metadata = lambda: _empty
    _cd._fetch_upcoming_predictions = lambda u, o: _empty

candidates = {{OUTCOME: CANDIDATE}} if CANDIDATE else None

if fixture:
    from src.reports.fixtures import build_fake_report_data
    data = build_fake_report_data(username=USERNAME, outcome=OUTCOME)
else:
    data = load(
        username=USERNAME,
        outcomes=OUTCOME,
        source=SOURCE,
        candidates=candidates,
    )
arts = data.outcomes[OUTCOME]
```

{{{{< include ../_collection_body.qmd >}}}}
"""


def _list_users(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _has_finalized_artifacts(user_dir: Path) -> bool:
    """A user is renderable iff some outcome has at least one finalized
    candidate. We check the on-disk presence of finalized.pkl rather than
    importing the loader (which would pull in BQ deps on import path)."""
    if not user_dir.exists():
        return False
    for outcome_dir in user_dir.iterdir():
        if not outcome_dir.is_dir():
            continue
        for cand_dir in outcome_dir.iterdir():
            if not cand_dir.is_dir() or cand_dir.name.startswith("v"):
                continue
            for version_dir in cand_dir.glob("v*"):
                if (version_dir / "finalized.pkl").exists():
                    return True
    return False


def _write_wrapper(username: str) -> Path:
    path = WRAPPERS_DIR / f"{username}.qmd"
    path.write_text(WRAPPER_TEMPLATE.format(username=username))
    return path


def _clean_stale_wrappers(keep_users: set[str]) -> list[Path]:
    """Remove wrapper qmds for users no longer in the artifact tree."""
    removed: list[Path] = []
    if not WRAPPERS_DIR.exists():
        return removed
    for path in WRAPPERS_DIR.glob("*.qmd"):
        if path.stem not in keep_users:
            path.unlink()
            removed.append(path)
    return removed


def _write_sidebar_yml(users: list[str]) -> None:
    """Rewrite reports/_quarto.yml's sidebar to list users explicitly,
    so the labels are usernames (not the included template's title)."""
    qy = REPORTS_DIR / "_quarto.yml"
    text = qy.read_text()
    # Replace the entire sidebar block. Match from `  sidebar:` to the
    # blank line that ends the website section.
    new_sidebar_lines = [
        "  sidebar:",
        "    style: floating",
        "    contents:",
        "      - href: index.qmd",
        "        text: Home",
        '      - section: "Collections"',
        "        contents:",
    ]
    for u in users:
        new_sidebar_lines.append(f"          - href: collections/{u}.qmd")
        new_sidebar_lines.append(f"            text: {u}")
    new_sidebar = "\n".join(new_sidebar_lines) + "\n"

    import re

    # Replace the sidebar: ... block (greedy until the next top-level key
    # or blank line of equal indent).
    pattern = re.compile(
        r"^  sidebar:\n(?:    .*\n|          .*\n|        .*\n|      .*\n)*",
        re.MULTILINE,
    )
    if not pattern.search(text):
        raise SystemExit("Could not find sidebar: block in _quarto.yml")
    new_text = pattern.sub(new_sidebar, text, count=1)
    qy.write_text(new_text)


def _quarto_render() -> int:
    """Invoke `quarto render` on the project (reports/). Honors
    QUARTO_PYTHON / GOOGLE_APPLICATION_CREDENTIALS the same way
    render.py does."""
    cmd = ["quarto", "render"]
    env = os.environ.copy()
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        env["QUARTO_PYTHON"] = str(venv_python)
    env["BGG_PROJECT_ROOT"] = str(PROJECT_ROOT)

    gac = env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not gac:
        creds = PROJECT_ROOT / "credentials" / "service-account-key.json"
        if creds.exists():
            env["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds)
    elif not Path(gac).is_absolute():
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(PROJECT_ROOT / gac)

    logger.info("Running: %s (cwd=%s)", " ".join(cmd), REPORTS_DIR)
    proc = subprocess.run(cmd, env=env, cwd=REPORTS_DIR)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Username to exclude. May be passed multiple times.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Generate wrappers only; skip `quarto render`.",
    )
    args = parser.parse_args(argv)

    excluded = ALWAYS_EXCLUDE | set(args.exclude)

    discovered = _list_users(ARTIFACTS_ROOT)
    if not discovered:
        logger.error("No users found under %s", ARTIFACTS_ROOT)
        return 1

    eligible: list[str] = []
    skipped_no_artifacts: list[str] = []
    for user in discovered:
        if user in excluded:
            continue
        if not _has_finalized_artifacts(ARTIFACTS_ROOT / user):
            skipped_no_artifacts.append(user)
            continue
        eligible.append(user)

    if not eligible:
        logger.error("No eligible users to build site for")
        return 1

    WRAPPERS_DIR.mkdir(parents=True, exist_ok=True)
    written = [_write_wrapper(u) for u in eligible]
    removed = _clean_stale_wrappers(set(eligible))
    _write_sidebar_yml(eligible)

    logger.info("Wrappers: wrote %d, removed %d", len(written), len(removed))
    for u in eligible:
        logger.info("  + %s", u)
    for u in skipped_no_artifacts:
        logger.info("  · skipped %s (no finalized artifacts)", u)
    for u in sorted(excluded & set(discovered)):
        logger.info("  · excluded %s", u)

    if args.no_render:
        return 0

    return _quarto_render()


if __name__ == "__main__":
    sys.exit(main())
