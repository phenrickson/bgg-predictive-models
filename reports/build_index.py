"""Render the index page (reports/index.qmd) into the output directory.

The index is a standalone Quarto page (not a website project) — it
scans the artifact source (local or `gs://`) for users with finalized
models, renders one card per user, and links to the per-user HTMLs
which are expected to be in the same output directory.

Usage:
    uv run python -m reports.build_index --output-dir reports/_output
    uv run python -m reports.build_index \\
        --source gs://bgg-predictive-models/prod/collections \\
        --output-dir reports/_output
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

logger = logging.getLogger("reports.build_index")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="local")
    parser.add_argument("--output-dir", default="reports/_output")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    qmd_path = Path(__file__).parent / "index.qmd"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "quarto",
        "render",
        str(qmd_path),
        "--output",
        "index.html",
        "-P",
        f"source={args.source}",
    ]

    env = os.environ.copy()
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        env["QUARTO_PYTHON"] = str(venv_python)
    env["BGG_PROJECT_ROOT"] = str(project_root)

    gac = env.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not gac:
        creds = project_root / "credentials" / "service-account-key.json"
        if creds.exists():
            env["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds)
    elif not Path(gac).is_absolute():
        env["GOOGLE_APPLICATION_CREDENTIALS"] = str(project_root / gac)

    logger.info("Rendering index: %s", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=qmd_path.parent)
    if proc.returncode != 0:
        return proc.returncode

    rendered = qmd_path.parent / "index.html"
    if not rendered.exists():
        logger.error("Quarto reported success but %s is missing", rendered)
        return 1
    target = output_dir / "index.html"
    rendered.replace(target)
    logger.info("Wrote %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
