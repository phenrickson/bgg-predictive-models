"""Integration tests for CollectionProcessor._load_features against
real BigQuery.

Requires ADC (gcloud auth application-default login) and
DATA_WAREHOUSE_PROJECT_ID (or GCP_PROJECT_ID) set.
"""

import pytest

from src.collection.collection_processor import (
    CollectionProcessor,
    ProcessorConfig,
)
from src.utils.config import load_config


pytestmark = pytest.mark.integration


def _make_processor(processor_config: ProcessorConfig) -> CollectionProcessor:
    """Build a processor without triggering collection-storage init (which
    would require extra setup). `_load_features` only needs `bq_config` and
    `processor_config`.
    """
    cfg = load_config()
    dw = cfg.get_data_warehouse_config()
    proc = CollectionProcessor.__new__(CollectionProcessor)
    proc.bq_config = dw
    proc.environment = "dev"
    proc.processor_config = processor_config
    proc.storage = None
    return proc


def test_load_features_real_bq_base_only():
    proc = _make_processor(
        ProcessorConfig(use_predicted_complexity=False, use_embeddings=False)
    )
    df = proc._load_features()

    assert df.height > 0
    assert "game_id" in df.columns
    assert "year_published" in df.columns
    assert "predicted_complexity" not in df.columns
    assert "emb_0" not in df.columns
    assert "embedding" not in df.columns


def test_load_features_real_bq_with_predicted_complexity():
    proc = _make_processor(
        ProcessorConfig(use_predicted_complexity=True, use_embeddings=False)
    )
    df = proc._load_features()

    assert df.height > 0
    assert "predicted_complexity" in df.columns
    # Some rows are expected to have non-null predicted_complexity.
    assert df["predicted_complexity"].drop_nulls().len() > 0


def test_load_features_real_bq_with_embeddings():
    proc = _make_processor(
        ProcessorConfig(use_predicted_complexity=False, use_embeddings=True)
    )
    df = proc._load_features()

    assert df.height > 0
    assert "embedding" not in df.columns
    assert "emb_0" in df.columns
    # emb_0 should have some non-null values from successful joins.
    assert df["emb_0"].drop_nulls().len() > 0
