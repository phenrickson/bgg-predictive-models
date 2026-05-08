"""Tests for candidate config resolution."""

from pathlib import Path

import pytest
import yaml

from src.models.candidate_config import find_candidate, list_candidates


def _write_config(tmp_path: Path, contents: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(contents))
    return p


def test_find_candidate_returns_block(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {
        "models": {
            "rating": {
                "candidates": [
                    {"name": "ard-ridge-rating", "algorithm": "ard"},
                    {"name": "catboost-rating", "algorithm": "catboost"},
                ],
            },
        },
    })
    cfg = find_candidate(config_path=p, model_type="rating", candidate="catboost-rating")
    assert cfg["algorithm"] == "catboost"
    assert cfg["name"] == "catboost-rating"


def test_find_candidate_raises_when_missing(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {"models": {"rating": {"candidates": []}}})
    with pytest.raises(KeyError):
        find_candidate(config_path=p, model_type="rating", candidate="nope")


def test_list_candidates(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {
        "models": {
            "rating": {
                "candidates": [
                    {"name": "a", "algorithm": "x"},
                    {"name": "b", "algorithm": "y"},
                ],
            },
        },
    })
    assert list_candidates(config_path=p, model_type="rating") == ["a", "b"]
