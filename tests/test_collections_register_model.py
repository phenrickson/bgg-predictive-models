"""Tests for services.collections.register_model.register_collection."""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_candidate_dir(tmp_path: Path, *, finalized: bool, train_only: bool):
    """Build a candidate dir with optional finalized.pkl / model.pkl."""
    cand_root = (
        tmp_path / "models" / "collections" / "alice" / "own" / "lgbm_default"
    )
    v1 = cand_root / "v1"
    v1.mkdir(parents=True)

    if finalized:
        (v1 / "finalized.pkl").write_bytes(pickle.dumps("FINALIZED_PIPELINE"))
    if train_only:
        (v1 / "model.pkl").write_bytes(pickle.dumps("TRAIN_PIPELINE"))

    (v1 / "threshold.json").write_text(json.dumps({"threshold": 0.42}))
    (v1 / "registration.json").write_text(
        json.dumps({"splits_version": 7, "finalize_through": 2025})
    )


def test_register_collection_strict_requires_finalized_pkl(tmp_path):
    """Only model.pkl present (no finalized.pkl) → FileNotFoundError."""
    from services.collections import register_model as rm

    _make_candidate_dir(tmp_path, finalized=False, train_only=True)
    local_root = str(tmp_path / "models" / "collections")

    with pytest.raises(FileNotFoundError, match="Finalized artifact not found"):
        rm.register_collection(
            username="alice",
            outcome="own",
            candidate="lgbm_default",
            description="test",
            version="latest",
            environment="dev",
            local_root=local_root,
        )


def test_register_collection_calls_gcs_then_registry(tmp_path):
    """Happy path: finalized.pkl present → GCS register, then registry insert."""
    from services.collections import register_model as rm

    _make_candidate_dir(tmp_path, finalized=True, train_only=False)
    local_root = str(tmp_path / "models" / "collections")

    with patch.object(rm, "RegisteredCollectionModel") as RCM_cls, \
         patch.object(rm, "RegistryWriter") as RW_cls, \
         patch.object(rm, "load_config") as load_cfg:
        rcm = MagicMock()
        rcm.register.return_value = {"version": 3, "username": "alice", "outcome": "own"}
        RCM_cls.return_value = rcm

        rw = MagicMock()
        RW_cls.return_value = rw

        cfg = MagicMock()
        cfg.get_collection_registry_table.return_value = "p.raw.collection_models_registry"
        cfg.get_bucket_name.return_value = "bgg-predictive-models"
        cfg.get_environment_prefix.return_value = "dev"
        load_cfg.return_value = cfg

        rm.register_collection(
            username="alice",
            outcome="own",
            candidate="lgbm_default",
            description="test",
            version="latest",
            environment="dev",
            local_root=local_root,
        )

        # GCS upload happened first
        rcm.register.assert_called_once()
        kwargs = rcm.register.call_args.kwargs
        assert kwargs["pipeline"] == "FINALIZED_PIPELINE"
        assert kwargs["threshold"] == 0.42
        assert kwargs["source_metadata"]["pipeline_kind"] == "finalized"

        # Then registry insert with the version from GCS
        rw.register_deployment.assert_called_once()
        rw_kwargs = rw.register_deployment.call_args.kwargs
        assert rw_kwargs["username"] == "alice"
        assert rw_kwargs["outcome"] == "own"
        assert rw_kwargs["model_version"] == 3
        assert rw_kwargs["finalize_through_year"] == 2025
        assert rw_kwargs["gcs_path"] == (
            "gs://bgg-predictive-models/dev/services/collections/alice/own/v3/"
        )
