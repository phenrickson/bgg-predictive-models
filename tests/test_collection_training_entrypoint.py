from pathlib import Path
import stat
import os
import subprocess


SCRIPT = Path(__file__).parent.parent / "docker" / "collection-training-entrypoint.sh"


def test_entrypoint_exists_and_executable():
    assert SCRIPT.exists()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be executable"


def test_entrypoint_has_required_shape():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    # The `collections` image has no gsutil — the gs:// boundary uses
    # the Python prefix sync, ordered down -> train -> up.
    assert "gsutil" not in text, "image has no gsutil; must not invoke it"
    i_pull = text.index("gcs_sync download")
    i_run = text.index("src.collection.train_model")
    i_push = text.rindex("gcs_sync up")
    assert i_pull < i_run < i_push, "must sync down, run, then sync up"
    for var in ("TRAIN_USERNAME", "ENVIRONMENT", "GCP_PROJECT_ID"):
        assert var in text
    assert "uv run python -m src.collection.train_model" in text


def test_missing_required_var_exits_nonzero():
    env = {k: v for k, v in os.environ.items() if k != "TRAIN_USERNAME"}
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True
    )
    assert result.returncode != 0
    assert b"TRAIN_USERNAME" in result.stderr
