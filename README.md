# BGG Predictive Models

Python implementation of predictive models for BoardGameGeek data, focusing on predicting game ratings, complexity, and popularity metrics.

## Overview

This project implements a multi-stage modeling approach to predict various BoardGameGeek metrics:

1. **Hurdle Model**: Predicts if a game will reach a minimum threshold of user ratings (default 25)
2. **Complexity Model**: Predicts game weight/complexity on BGG's 1-5 scale
3. **Rating Model**: Predicts average user rating
4. **Users Rated Model**: Predicts number of user ratings
5. **Bayesaverage**: Combines predictions to calculate BGG's weighted rating system

The models use data from the BGG data warehouse and are designed to work with both existing and upcoming game releases.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/phenrickson/bgg-predictive-models.git
cd bgg-predictive-models
```

2. Create a Python virtual environment using UV:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
uv pip install -e ".[dev]"
```

## Configuration

The project requires access to a BigQuery dataset containing BGG data. Set the following environment variables:

```bash
export BGG_PROJECT_ID="your-gcp-project-id"
export BGG_DATASET="your-bigquery-dataset"
export BGG_CREDENTIALS_PATH="/path/to/service-account-key.json"  # Optional
```

You can also use a `.env` file:

```env
BGG_PROJECT_ID=your-gcp-project-id
BGG_DATASET=your-bigquery-dataset
BGG_CREDENTIALS_PATH=/path/to/service-account-key.json
```

## Usage

### Training Models

Train the full pipeline using the provided script:

```bash
# Basic usage
python -m src.bgg_predictive_models.scripts.train --output-dir models/

# With custom parameters
python -m src.bgg_predictive_models.scripts.train \
    --end-train-year 2021 \
    --valid-years 2 \
    --min-ratings 25 \
    --random-state 42 \
    --output-dir models/ \
    --log-file logs/training.log
```

### Making Predictions

Use trained models to make predictions:

```bash
# Predict for specific games
python -m src.bgg_predictive_models.scripts.predict \
    --model-dir models/ \
    --game-ids 174430 161936 224517 \
    --output-file predictions.csv

# Predict for all games
python -m src.bgg_predictive_models.scripts.predict \
    --model-dir models/ \
    --output-file predictions.csv
```

### Python API

You can also use the models programmatically:

```python
from bgg_predictive_models.data.config import get_config_from_env
from bgg_predictive_models.data.loader import BGGDataLoader
from bgg_predictive_models.models.pipeline import BGGPipeline

# Load data
config = get_config_from_env()
loader = BGGDataLoader(config)
features, targets = loader.load_training_data()

# Train pipeline
pipeline = BGGPipeline()
pipeline.fit(
    X=features,
    y_hurdle=targets["hurdle"],
    y_complexity=targets["complexity"],
    y_rating=targets["rating"],
    y_users_rated=targets["users_rated"],
)

# Make predictions
predictions = pipeline.predict(features)
bayesavg = pipeline.predict_bayesaverage(features)
```

## Project Structure

```
bgg-predictive-models/
├── src/
│   └── bgg_predictive_models/
│       ├── data/
│       │   ├── config.py      # Database configuration
│       │   └── loader.py      # Data loading and preprocessing
│       ├── models/
│       │   ├── base.py        # Base model classes
│       │   ├── hurdle.py      # Rating threshold prediction
│       │   ├── complexity.py  # Game weight prediction
│       │   ├── rating.py      # Average rating prediction
│       │   ├── users_rated.py # Number of ratings prediction
│       │   └── pipeline.py    # Multi-stage model pipeline
│       └── scripts/
│           ├── train.py       # Model training script
│           └── predict.py     # Prediction script
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Model Details

### Hurdle Model
- LightGBM classifier
- Predicts probability of reaching minimum rating threshold
- Features: game attributes, categories, mechanics
- Evaluation: ROC AUC

### Complexity Model
- LightGBM regressor
- Predicts game weight/complexity (1-5 scale)
- Used for imputing missing complexity values
- Evaluation: RMSE, R²

### Rating Model
- Elastic Net regressor
- Predicts average user rating
- Includes predicted complexity as feature
- Evaluation: RMSE, R²

### Users Rated Model
- Elastic Net regressor
- Predicts number of user ratings
- Log-transformed target
- Evaluation: RMSE, R² (original and log scale)

### Bayesaverage Calculation
Implements BGG's weighted rating system:
```python
bayesavg = (rating * n_ratings + prior_rating * prior_weight) / (n_ratings + prior_weight)
```
Where:
- `rating`: Predicted average rating
- `n_ratings`: Predicted number of ratings
- `prior_rating`: Default 5.5
- `prior_weight`: Default 100

## Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/

# Run specific test file
pytest tests/test_models.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Run linter
ruff check src/ tests/

# Type checking
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details
