from pathlib import Path
import stat


SCRIPT = Path("docker/collection-training-entrypoint.sh")


def test_entrypoint_exists_and_executable():
    assert SCRIPT.exists()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint must be executable"


def test_entrypoint_has_required_shape():
    text = SCRIPT.read_text()
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    i_pull = text.index("rsync")
    i_run = text.index("src.collection.train_model")
    i_push = text.rindex("rsync")
    assert i_pull < i_run < i_push, "must rsync down, run, then rsync up"
    for var in ("TRAIN_USERNAME", "ENVIRONMENT", "GCP_PROJECT_ID"):
        assert var in text
    assert "uv run python -m src.collection.train_model" in text
