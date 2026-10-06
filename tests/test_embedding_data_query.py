# tests/test_embedding_data_query.py
from unittest.mock import MagicMock, patch

from src.models.embeddings.data import EmbeddingDataLoader


def test_build_query_reads_raw_with_bound():
    config = MagicMock()
    config.data_warehouse.project_id = "dw"
    config.data_warehouse.features_dataset = "analytics"
    config.data_warehouse.features_table = "games_features"
    with patch("src.models.embeddings.data.bigquery.Client"), \
         patch("src.models.embeddings.data.complexity_min_score_ts", return_value="2026-03-12 04:00:00+00"):
        loader = EmbeddingDataLoader(config)
        sql = loader._build_query("TRUE", use_embeddings=True)
    assert "raw.complexity_predictions" in sql
    assert "score_ts >= TIMESTAMP('2026-03-12 04:00:00+00')" in sql
    assert "raw.description_embeddings" in sql
    assert "predictions.bgg_" not in sql
