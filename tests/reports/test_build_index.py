"""Unit tests for src.reports.output_index.discover_output_reports."""

from __future__ import annotations

from pathlib import Path

from src.reports.output_index import discover_output_reports


def test_detects_both_links(tmp_path: Path):
    (tmp_path / "alice.html").write_text("x")
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "alice.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert rows == [
        {"username": "alice", "has_predictions": True, "has_model": True}
    ]


def test_predictions_only_degrades(tmp_path: Path):
    (tmp_path / "bob.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert rows == [
        {"username": "bob", "has_predictions": True, "has_model": False}
    ]


def test_model_only(tmp_path: Path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "carol.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert rows == [
        {"username": "carol", "has_predictions": False, "has_model": True}
    ]


def test_index_html_is_not_a_user(tmp_path: Path):
    (tmp_path / "index.html").write_text("x")
    (tmp_path / "dave.html").write_text("x")
    rows = discover_output_reports(tmp_path)
    assert [r["username"] for r in rows] == ["dave"]


def test_empty_dir(tmp_path: Path):
    assert discover_output_reports(tmp_path) == []
