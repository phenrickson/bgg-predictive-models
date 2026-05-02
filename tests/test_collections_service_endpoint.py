"""End-to-end test of /predict_own with all GCP calls mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mocked_app():
    """Build the app with all external deps mocked at import time."""
    with patch("services.collections.main.GCPAuthenticator") as auth_cls, \
         patch("services.collections.main.bigquery.Client") as bq_cls, \
         patch("services.collections.main.load_config") as cfg_fn:
        auth = MagicMock()
        auth.project_id = "test-project"
        auth_cls.return_value = auth
        bq_cls.return_value = MagicMock()

        cfg = MagicMock()
        cfg.get_bucket_name.return_value = "test-bucket"
        cfg.get_environment_prefix.return_value = "dev"
        cfg.get_collection_registry_table.return_value = "p.raw.collection_models_registry"
        cfg.get_collection_landing_table.return_value = "p.raw.collection_predictions_landing"
        cfg.get_bigquery_config.return_value = MagicMock()
        cfg_fn.return_value = cfg

        # Force a fresh import so module-level code re-runs under mocks
        import importlib
        from services.collections import main as m
        importlib.reload(m)
        yield m


def test_predict_own_with_explicit_game_ids_returns_predictions(mocked_app):
    m = mocked_app

    # Stub registry.lookup_latest
    entry = MagicMock()
    entry.model_version = 3
    entry.gcs_path = "gs://x/v3"
    m.registry.lookup_latest = MagicMock(return_value=entry)

    # Stub pipeline load
    pipeline = MagicMock()
    pipeline.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4]])
    m._PIPELINE_CACHE.clear()
    with patch.object(m, "RegisteredCollectionModel") as rcm_cls:
        rcm = MagicMock()
        rcm.load.return_value = (pipeline, 0.5, {})
        rcm_cls.return_value = rcm

        # Stub feature loader
        m._loader = MagicMock()
        m._loader.load_features.return_value = pl.DataFrame({"game_id": [10, 11], "x": [1.0, 2.0]})

        # Stub uploader so we don't hit BQ
        with patch("services.collections.main.CollectionPredictionsUploader") as up_cls:
            up = MagicMock()
            up.upload.return_value = 2
            up_cls.return_value = up

            client = TestClient(m.app)
            resp = client.post(
                "/predict_own",
                json={
                    "username": "alice",
                    "game_ids": [10, 11],
                    "use_change_detection": False,
                    "upload_to_data_warehouse": True,
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["username"] == "alice"
    assert body["outcome"] == "own"
    assert body["model_version"] == 3
    assert body["n_scored"] == 2
    assert isinstance(body["job_id"], str) and len(body["job_id"]) > 0
    probs = {p["game_id"]: p["predicted_prob"] for p in body["predictions"]}
    assert probs == {10: 0.8, 11: 0.4}
    labels = {p["game_id"]: p["predicted_label"] for p in body["predictions"]}
    assert labels == {10: True, 11: False}


def test_predict_own_returns_404_when_user_not_registered(mocked_app):
    m = mocked_app
    m.registry.lookup_latest = MagicMock(return_value=None)
    client = TestClient(m.app)

    resp = client.post(
        "/predict_own",
        json={"username": "ghost", "game_ids": [1], "upload_to_data_warehouse": False},
    )
    assert resp.status_code == 404


def test_predict_own_rejects_both_game_ids_and_change_detection(mocked_app):
    m = mocked_app
    entry = MagicMock()
    entry.model_version = 1
    entry.gcs_path = "gs://x"
    m.registry.lookup_latest = MagicMock(return_value=entry)
    client = TestClient(m.app)

    resp = client.post(
        "/predict_own",
        json={
            "username": "alice",
            "game_ids": [1],
            "use_change_detection": True,
            "upload_to_data_warehouse": False,
        },
    )
    assert resp.status_code == 400
