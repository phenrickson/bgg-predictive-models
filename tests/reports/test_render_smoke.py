"""End-to-end smoke test for reports/render.py.

Skipped if Quarto is not on PATH. Runs the render driver against the
fixture artifact tree and asserts that an HTML output file is produced.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("quarto") is None, reason="Quarto not installed on PATH"
)
def test_render_smoke(fixture_collection_root: Path, tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "reports.render",
        "--username",
        "phenrickson",
        "--outcome",
        "own",
        "--source",
        str(fixture_collection_root),
        "--output-dir",
        str(output_dir),
        "--candidate",
        "logistic_row_norm",
    ]
    env_extra = {"BGG_REPORTS_OFFLINE": "1"}
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, **env_extra},
    )
    assert result.returncode == 0, f"render failed: {result.stderr}"
    out_html = output_dir / "phenrickson.html"
    assert out_html.exists()
    assert out_html.stat().st_size > 1000  # non-trivial
