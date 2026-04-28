"""Integration tests for BGGDataLoader.load_features against real BigQuery.

Requires ADC (gcloud auth application-default login) and
DATA_WAREHOUSE_PROJECT_ID (or GCP_PROJECT_ID) set.
"""

import pytest

from src.data.loader import BGGDataLoader
from src.utils.config import load_config


pytestmark = pytest.mark.integration


def _make_loader() -> BGGDataLoader:
    cfg = load_config()
    dw = cfg.get_data_warehouse_config()
    return BGGDataLoader(dw)


def test_load_features_real_bq_base_only():
    df = _make_loader().load_features()

    assert df.height > 0
    assert "game_id" in df.columns
    assert "year_published" in df.columns
    assert "predicted_complexity" not in df.columns
    assert "emb_0" not in df.columns
    assert "embedding" not in df.columns


def test_load_features_real_bq_with_predicted_complexity():
    df = _make_loader().load_features(use_predicted_complexity=True)

    assert df.height > 0
    assert "predicted_complexity" in df.columns
    assert df["predicted_complexity"].drop_nulls().len() > 0


def test_load_features_real_bq_with_embeddings():
    df = _make_loader().load_features(use_embeddings=True)

    assert df.height > 0
    assert "embedding" not in df.columns
    assert "emb_0" in df.columns
    assert df["emb_0"].drop_nulls().len() > 0
