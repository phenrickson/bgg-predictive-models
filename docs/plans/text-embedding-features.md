# Plan: Text Embeddings as Predictive Model Features

## Goal
Integrate document embeddings (from game descriptions) as features in the predictive modeling pipeline by uploading to BigQuery and joining with `games_features`.

---

## Current State

### Text Embeddings Module
- **Location**: `src/models/text_embeddings/`
- **Training**: PMI + SVD word embeddings, SIF document aggregation
- **Model Storage**: `models/experiments/text_embeddings/{experiment}/v{N}/`
- **Generator Class**: `TextEmbeddingGenerator` loads trained model and generates embeddings

### Data Warehouse Pattern
- **Project**: `bgg-data-warehouse`
- **Features table**: `analytics.games_features`
- **Pattern**: Compute features locally, upload to BigQuery, join in SQL/views

---

## Implementation Plan

### Part 1: Generate & Upload Embeddings to BigQuery

**Create**: `src/models/text_embeddings/generate_features.py`

```python
"""Generate document embeddings and upload to BigQuery."""

def main(
    experiment_name: str = "text-embeddings",
    version: Optional[int] = None,
    project_id: str = "bgg-data-warehouse",
    dataset: str = "analytics",
    table: str = "description_embeddings",
):
    """
    1. Load all games from games_features (just game_id + description)
    2. Generate embeddings using trained model
    3. Upload to BigQuery as analytics.description_embeddings

    Schema:
    - game_id: INT64 (primary key)
    - desc_emb_0: FLOAT64
    - desc_emb_1: FLOAT64
    - ...
    - desc_emb_99: FLOAT64
    - embedding_experiment: STRING (e.g., "text-embeddings/v5")
    - created_at: TIMESTAMP
    """
```

**Makefile Target**:
```makefile
description_embeddings:
	uv run -m src.models.text_embeddings.generate_features \
		--experiment text-embeddings \
		--project bgg-data-warehouse \
		--dataset analytics \
		--table description_embeddings
```

---

### Part 2: Create Joined View or Update games_features

**Option A**: Create a view that joins embeddings

```sql
-- analytics.games_features_with_embeddings
SELECT
    gf.*,
    de.desc_emb_0, de.desc_emb_1, ..., de.desc_emb_99
FROM analytics.games_features gf
LEFT JOIN analytics.description_embeddings de
    ON gf.game_id = de.game_id
```

**Option B**: Add embeddings directly to games_features pipeline

If you have a dbt or SQL pipeline that builds `games_features`, add the join there.

---

### Part 3: Update Data Loader

**Modify**: `src/data/loader.py` or `config.yaml`

Option to load from the joined view:
```yaml
data_warehouse:
  features_table: games_features_with_embeddings  # or games_features
```

Or add a flag:
```python
def load_training_data(
    ...,
    include_embeddings: bool = False,
):
    table = "games_features_with_embeddings" if include_embeddings else "games_features"
```

---

## File Changes Summary

| Action | File | Description |
|--------|------|-------------|
| Create | `src/models/text_embeddings/generate_features.py` | Generate embeddings, upload to BQ |
| Create | BigQuery view or table | `analytics.description_embeddings` |
| Modify | `config.yaml` or `loader.py` | Point to joined table/view |
| Modify | `Makefile` | Add `description_embeddings` target |

---

## Workflow

```bash
# 1. Train text embeddings (already done)
make text_embeddings

# 2. Generate embeddings and upload to BigQuery
make description_embeddings

# 3. Training automatically uses embeddings via joined table
make complexity
```

No changes needed to model training scripts - embeddings are just more columns in the feature table.

---

## BigQuery Schema

**Table**: `analytics.description_embeddings`

| Column | Type | Description |
|--------|------|-------------|
| game_id | INT64 | Primary key, join to games_features |
| desc_emb_0 | FLOAT64 | Embedding dimension 0 |
| desc_emb_1 | FLOAT64 | Embedding dimension 1 |
| ... | ... | ... |
| desc_emb_99 | FLOAT64 | Embedding dimension 99 |
| embedding_experiment | STRING | Source experiment (e.g., "text-embeddings/v5") |
| created_at | TIMESTAMP | When embeddings were generated |

---

## Design Decisions

### Why BigQuery?
- Consistent with existing data pipeline
- Join happens at data layer, not in Python
- Easy to version/replace embeddings
- No changes needed to training scripts

### Embedding Versioning
- Store `embedding_experiment` column in the table
- When regenerating, can either replace table or create new version
- Training metadata can query this column

### Leakage Considerations
- Game descriptions are static
- Single embedding model trained once is acceptable
- If needed: filter embeddings table by year for strict separation

---

## Testing Checklist

- [ ] Generate embeddings for all games
- [ ] Upload to BigQuery successfully
- [ ] Verify row count matches games_features
- [ ] Create/update joined view
- [ ] Verify training loads embedding columns
- [ ] Train model, check embedding features in importance
