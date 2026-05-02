"""Tests for services.collections.registry_writer.RegistryWriter."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from services.collections.registry_writer import RegistryWriter


def _build_writer():
    bq = MagicMock()
    return RegistryWriter("proj.raw.collection_models_registry", bq), bq


def test_register_deployment_updates_then_inserts():
    writer, bq = _build_writer()

    writer.register_deployment(
        username="alice",
        outcome="own",
        model_version=1,
        gcs_path="gs://bucket/dev/services/collections/alice/own/v1/",
        finalize_through_year=2025,
        registered_at=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )

    # Two BQ calls: UPDATE then INSERT (in that order)
    assert bq.query.call_count == 2
    update_sql, _ = bq.query.call_args_list[0].args, bq.query.call_args_list[0].kwargs
    insert_sql, _ = bq.query.call_args_list[1].args, bq.query.call_args_list[1].kwargs
    assert "UPDATE" in update_sql[0] and "status = 'inactive'" in update_sql[0]
    assert "INSERT" in insert_sql[0]


def test_register_deployment_threads_correct_parameters():
    writer, bq = _build_writer()

    ts = datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc)
    writer.register_deployment(
        username="alice",
        outcome="own",
        model_version=2,
        gcs_path="gs://bucket/dev/services/collections/alice/own/v2/",
        finalize_through_year=2025,
        registered_at=ts,
    )

    # UPDATE params
    upd_call = bq.query.call_args_list[0]
    upd_params = {p.name: p.value for p in upd_call.kwargs["job_config"].query_parameters}
    assert upd_params == {"username": "alice", "outcome": "own"}

    # INSERT params
    ins_call = bq.query.call_args_list[1]
    ins_params = {p.name: p.value for p in ins_call.kwargs["job_config"].query_parameters}
    assert ins_params == {
        "username": "alice",
        "outcome": "own",
        "model_version": 2,
        "finalize_through_year": 2025,
        "gcs_path": "gs://bucket/dev/services/collections/alice/own/v2/",
        "registered_at": ts,
    }


def test_register_deployment_first_ever_row_still_runs_update():
    """No prior active row → UPDATE matches 0 rows, INSERT still runs."""
    writer, bq = _build_writer()

    writer.register_deployment(
        username="bob",
        outcome="own",
        model_version=1,
        gcs_path="gs://bucket/dev/services/collections/bob/own/v1/",
        finalize_through_year=None,
        registered_at=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert bq.query.call_count == 2  # UPDATE + INSERT regardless


def test_register_deployment_handles_null_finalize_through_year():
    writer, bq = _build_writer()

    writer.register_deployment(
        username="alice",
        outcome="own",
        model_version=1,
        gcs_path="gs://x/v1/",
        finalize_through_year=None,
        registered_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    ins_call = bq.query.call_args_list[1]
    params = {p.name: p.value for p in ins_call.kwargs["job_config"].query_parameters}
    assert params["finalize_through_year"] is None
