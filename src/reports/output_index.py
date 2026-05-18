"""Discovery of which per-user report HTMLs exist in a rendered bundle.

Pure filesystem inspection, no Quarto. Lives under ``src/reports`` so it
is importable by both ``reports/build_index.py`` (and the ``index.qmd``
it renders) and the test suite, mirroring how ``src.reports.tables`` etc.
are consumed.
"""

from __future__ import annotations

from pathlib import Path


def discover_output_reports(output_dir: Path | str) -> list[dict]:
    """Scan a rendered output directory and report, per user, which of
    the two report HTMLs exist.

    Predictions reports are ``{output_dir}/{user}.html``; model reports
    are ``{output_dir}/model/{user}.html``. ``index.html`` is not a
    user. Returns one dict per user sorted by username with
    ``has_predictions`` / ``has_model`` flags so the index can degrade
    to a single link when one side has not been rendered yet.
    """
    output_dir = Path(output_dir)
    users: dict[str, dict] = {}

    for p in sorted(output_dir.glob("*.html")):
        if p.name == "index.html":
            continue
        users.setdefault(
            p.stem,
            {"username": p.stem, "has_predictions": False, "has_model": False},
        )["has_predictions"] = True

    model_dir = output_dir / "model"
    if model_dir.is_dir():
        for p in sorted(model_dir.glob("*.html")):
            users.setdefault(
                p.stem,
                {"username": p.stem, "has_predictions": False, "has_model": False},
            )["has_model"] = True

    return [users[u] for u in sorted(users)]
