# Model Registration and Scoring Service

This service provides a system for registering trained models and deploying them for scoring new data. It includes:

- Model registration with validation metrics
- Version control for registered models
- FastAPI service for model scoring
- CLI tools for model management

## Model Registration

Models must be registered before they can be used for scoring. The registration process includes:

1. Validation against metric thresholds
2. Version tracking
3. Metadata storage
4. Model artifact storage in Google Cloud Storage

### Registering a Model

Use the `register_model.py` script to register a model:

```bash
python -m services.scoring.register_model \
    --model-type rating \
    --experiment best_rating_model \
    --name production_rating_model \
    --description "Production model for game ratings" \
    --bucket bgg-models
```


## Scoring Service

The scoring service provides a REST API for scoring new data using registered models.

### API Endpoints

#### Simulate Games (Bayesian Posterior Sampling)

```http
POST /simulate_games
```

Generates predictions with credible intervals via posterior sampling.

```json
{
  "hurdle_model_name": "hurdle-v2026",
  "complexity_model_name": "complexity-v2026",
  "rating_model_name": "rating-v2026",
  "users_rated_model_name": "users_rated-v2026",
  "geek_rating_model_name": "geek_rating-v2026",
  "game_ids": [456459],
  "n_samples": 1000,
  "upload_to_data_warehouse": false
}
```

#### Explain Game Predictions

```http
POST /explain_game
```

Returns per-feature contribution breakdowns for all outcomes (complexity, rating, users_rated, geek_rating). Handles the model dependency chain internally.

```json
{
  "game_id": 456459,
  "complexity_model_name": "complexity-v2026",
  "rating_model_name": "rating-v2026",
  "users_rated_model_name": "users_rated-v2026",
  "geek_rating_model_name": "geek_rating-v2026",
  "top_n": 15
}
```

#### Individual Model Endpoints

```http
POST /predict_complexity
POST /predict_hurdle
POST /predict_rating
POST /predict_users_rated
POST /predict_games
```

#### List Models
```http
GET /models
```

Returns all registered models across different types.

#### Get Model Info
```http
GET /model/{model_type}/{model_name}/info?version={version}
```

Returns detailed information about a specific registered model.

### Environment Variables

- `GCS_BUCKET_NAME`: Google Cloud Storage bucket for storing registered models (default: "bgg-models")

## Storage Structure

Registered models are stored in Google Cloud Storage with the following structure:

```
models/registered/
├── {model_type}/
│   ├── {model_name}/
│   │   ├── v1/
│   │   │   ├── registration.json
│   │   │   └── pipeline.pkl
│   │   └── v2/
│   │       ├── registration.json
│   │       └── pipeline.pkl
```

## Example Usage

1. Register a model:
```bash
python -m services.scoring.register_model \
    --model-type rating \
    --experiment best_rating_model \
    --name production_rating_model \
    --description "Production model for game ratings"
```

2. Score new data using the API:
```python
import requests

response = requests.post(
    "http://localhost:8080/score",
    json={
        "model_type": "rating",
        "model_name": "production_rating_model",
        "start_year": 2020,
        "end_year": 2025
    }
)

predictions = response.json()
```

3. List available models:
```python
import requests

response = requests.get("http://localhost:8080/models")
available_models = response.json()
```

4. Get model information:
```python
import requests

response = requests.get(
    "http://localhost:8080/model/rating/production_rating_model/info"
)
model_info = response.json()
```

## Development

### Running the Service Locally

1. Set environment variables:
```bash
export GCS_BUCKET_NAME=bgg-models
```

2. Start the service:
```bash
cd services/scoring
uvicorn main:app --reload
```

### Running Tests

```bash
pytest tests/services/scoring/
```

### Deployment

The service can be deployed to Google Cloud Run using the provided Dockerfile and cloudbuild.yaml:

```bash
gcloud builds submit
```

This will:
1. Build the container image
2. Push it to Container Registry
3. Deploy to Cloud Run
