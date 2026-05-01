"""Tests for services.collections.registry.CollectionRegistry."""

from unittest.mock import MagicMock
import pytest

from services.collections.registry import CollectionRegistry, RegistryEntry


def _row(username, outcome, version, gcs_path, status="active", year=2025):
    r = MagicMock()
    r.username = username
    r.outcome = outcome
    r.model_version = version
    r.finalize_through_year = year
    r.gcs_path = gcs_path
    r.status = status
    return r


def test_lookup_latest_returns_highest_active_version():
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = [
        _row("alice", "own", 2, "gs://b/v2"),
    ]
    reg = CollectionRegistry("project.raw.collection_models_registry", bq_client)

    entry = reg.lookup_latest("alice", "own")

    assert entry.username == "alice"
    assert entry.outcome == "own"
    assert entry.model_version == 2
    assert entry.gcs_path == "gs://b/v2"
    assert entry.status == "active"


def test_lookup_latest_returns_none_when_no_active_rows():
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = []
    reg = CollectionRegistry("project.raw.collection_models_registry", bq_client)

    assert reg.lookup_latest("ghost", "own") is None


def test_list_active_returns_one_entry_per_user_outcome():
    bq_client = MagicMock()
    bq_client.query.return_value.result.return_value = [
        _row("alice", "own", 2, "gs://b/alice/v2"),
        _row("bob",   "own", 1, "gs://b/bob/v1"),
    ]
    reg = CollectionRegistry("project.raw.collection_models_registry", bq_client)

    entries = reg.list_active(outcome="own")

    assert len(entries) == 2
    assert {e.username for e in entries} == {"alice", "bob"}
    assert all(e.status == "active" for e in entries)
