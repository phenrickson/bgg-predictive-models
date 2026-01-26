# Design: Outcome Models Refactor

## Goal

Refactor the predictive model training code to be class-based, with standardized data loading that supports text embeddings as optional features.

---

## Current State

### Model Training Scripts
- `src/models/hurdle.py` - classification model for predicting if games get rated
- `src/models/complexity.py` - regression model for game complexity
- `src/models/rating.py` - regression model for average rating
- `src/models/users_rated.py` - regression model for number of ratings
- `src/models/geek_rating.py` - orchestrator that combines the above models

### Problems
1. **Code duplication** - Each script has nearly identical structure (~90% shared patterns)
2. **Inconsistent data loading** - Each model handles its own data filtering and joins differently
3. **Hard to add features** - Adding embeddings requires touching each model script
4. **Doesn't match patterns elsewhere** - `embeddings/` and `text_embeddings/` use class-based design

---

## Design

### Two Base Classes with Shared Interface

```
Predictor (protocol/interface)
├── predict(features) -> predictions
├── load()
└── model_type, version, metadata

TrainableModel(Predictor)
├── data_requirements: DataConfig
├── train(), tune(), evaluate()
├── target_column, model_task (classification/regression)
└── HurdleModel, ComplexityModel, RatingModel, UsersRatedModel

CompositeModel(Predictor)
├── sub_models: list[Predictor]
├── combine()
└── GeekRatingModel
```

### Model Data Contracts

Each `TrainableModel` subclass defines what data it needs:

| Model | Filter | Requires Complexity Preds | Supports Embeddings | Target | Task |
|-------|--------|---------------------------|---------------------|--------|------|
| `HurdleModel` | `min_ratings` | No | Yes | `hurdle` | Classification |
| `ComplexityModel` | `min_weights` | No | No | `complexity` | Regression |
| `RatingModel` | `min_ratings` | Yes | Yes | `rating` | Regression |
| `UsersRatedModel` | `min_ratings` | Yes | Yes | `log_users_rated` | Regression |

### GeekRatingModel

Starts as an orchestrator (loads 4 sub-models, combines with Bayesian average), architected to later support:
- **Trained combiner** - learn to combine outputs (stacking/ensemble)
- **End-to-end** - train directly on geek_rating as target

---

## Data Loading

### Centralized Loader

Model classes declare their data requirements. A centralized loader handles fetching:

```python
@dataclass
class DataConfig:
    min_ratings: int | None = None
    min_weights: int | None = None
    requires_complexity_predictions: bool = False
    supports_embeddings: bool = False

class RatingModel(TrainableModel):
    target_column = "rating"
    model_task = "regression"

    data_config = DataConfig(
        min_ratings=5,
        requires_complexity_predictions=True,
        supports_embeddings=True,
    )
```

The loader then fetches what's needed:

```python
df = load_training_data(
    config=model.data_config,
    use_embeddings=args.use_embeddings,  # runtime flag
    complexity_predictions_path=args.complexity_path,
    years=year_config,
)
```

### Embeddings Loading

- **Source**: `predictions.bgg_description_embeddings` in BigQuery (via bgg-data-warehouse Dataform)
- **Dimension**: Full embedding dimension (no reduction)
- **Join**: On `game_id`, concatenated directly to features
- **Control**: `--use-embeddings` flag at training time

---

## Configuration

### Model Classes Define (structural, unchanging)
- Target column
- Model task (classification vs regression)
- Data requirements (what joins it supports)

### config.yaml Defines (tunables)
```yaml
models:
  hurdle:
    type: lightgbm
    experiment_name: lightgbm-hurdle
    min_ratings: 0

  complexity:
    type: catboost
    experiment_name: catboost-complexity
    min_weights: 5
    use_sample_weights: true

  rating:
    type: catboost
    experiment_name: catboost-rating
    min_ratings: 5
    use_sample_weights: true

  users_rated:
    type: ridge
    experiment_name: ridge-users_rated
    min_ratings: 0

embeddings:
  enabled: false  # default off, enable via CLI or here
  source: bigquery  # or local path
  table: predictions.bgg_description_embeddings
```

---

## File Organization

```
src/models/
├── outcomes/                      # NEW - outcome prediction models
│   ├── __init__.py
│   ├── base.py                    # Predictor, TrainableModel, CompositeModel, DataConfig
│   ├── hurdle.py                  # HurdleModel
│   ├── complexity.py              # ComplexityModel
│   ├── rating.py                  # RatingModel
│   ├── users_rated.py             # UsersRatedModel
│   ├── geek_rating.py             # GeekRatingModel
│   ├── data.py                    # Centralized data loading (load_training_data)
│   └── train.py                   # Unified training entry point
│
├── embeddings/                    # Existing - structural embeddings
├── text_embeddings/               # Existing - text embeddings
│
├── experiments.py                 # Keep - experiment tracking
├── training.py                    # Keep - shared utilities (tune_model, evaluate_model)
├── splitting.py                   # Keep - time-based splits
├── score.py                       # Keep - scoring utilities
├── finalize_model.py              # Keep - model finalization
└── time_based_evaluation.py       # Keep - evaluation utilities
```

---

## Training Invocation

### Unified Script (primary)
```bash
python -m src.models.outcomes.train \
    --model hurdle \
    --algo lightgbm \
    --experiment my-hurdle-exp \
    --use-embeddings
```

### Per-Model Scripts (convenience)
```bash
python -m src.models.outcomes.hurdle \
    --algo lightgbm \
    --experiment my-hurdle-exp \
    --use-embeddings
```

Per-model scripts just delegate to the unified trainer:
```python
# src/models/outcomes/hurdle.py
if __name__ == "__main__":
    from src.models.outcomes.train import train_model
    train_model(HurdleModel)
```

---

## Implementation Phases

### Phase 1: Base Classes and Structure
- [ ] Create `src/models/outcomes/` directory
- [ ] Implement `base.py` with `Predictor`, `TrainableModel`, `CompositeModel`, `DataConfig`
- [ ] Implement `data.py` with centralized `load_training_data()`

### Phase 2: Migrate Models
- [ ] Implement `HurdleModel` class
- [ ] Implement `ComplexityModel` class
- [ ] Implement `RatingModel` class
- [ ] Implement `UsersRatedModel` class
- [ ] Implement `GeekRatingModel` class

### Phase 3: Training Scripts
- [ ] Implement unified `train.py` entry point
- [ ] Add per-model script entry points
- [ ] Update config.yaml with model defaults

### Phase 4: Embeddings Integration
- [ ] Add embeddings loading to `data.py`
- [ ] Add `--use-embeddings` CLI flag
- [ ] Test embedding features with hurdle/rating/users_rated models

### Phase 5: Cleanup
- [ ] Deprecate old top-level model scripts
- [ ] Update Makefile targets
- [ ] Update documentation

---

## Migration Strategy

1. Build new structure in `src/models/outcomes/` alongside existing code
2. Verify parity: new classes produce identical results to old scripts
3. Update Makefile to use new entry points
4. Deprecate old scripts (keep for reference initially)
5. Remove old scripts after validation period

---

## Testing Checklist

- [ ] Each model class loads correct data (filters, joins)
- [ ] Training produces identical metrics to old scripts (without embeddings)
- [ ] Embeddings load correctly from BigQuery
- [ ] Embeddings improve/don't hurt model performance
- [ ] GeekRatingModel produces same predictions as old geek_rating.py
- [ ] Unified train script works for all model types
- [ ] Per-model scripts work as shortcuts
