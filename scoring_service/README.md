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
python -m scoring_service.register_model \
    --model-type rating \
    --experiment best_rating_model \
    --name production_rating_model \
    --description "Production model for game ratings" \
    --bucket bgg-models
```


## Scoring Service

The scoring service provides a REST API for scoring new data using registered models.

### API Endpoints

#### Score Data
```http
POST /score
```

Request body:
```json
{
  "model_type": "rating",
  "model_name": "production_rating_model",
  "model_version": 1,  // optional, uses latest if not specified
  "start_year": 2020,  // optional
  "end_year": 2025,    // optional
  "data_source": "query",
  "output_location": "path/to/save/predictions.parquet"  // optional
}
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
python -m scoring_service.register_model \
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
cd scoring_service
uvicorn main:app --reload
```

### Running Tests

```bash
pytest tests/scoring_service/
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
