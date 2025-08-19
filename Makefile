# Makefile for BGG predictive models

# Default settings
RAW_DIR := data/raw

.PHONY: help clean all

help:  ## Show this help message
	@echo 'Usage:'
	@echo '  make help         Show this help message'
	@echo '  make all          Fetch all data from BigQuery'
	@echo '  make clean        Remove generated data files'
	@echo '  make clean_experiments  Remove all experiment subfolders'
	@echo '  make clean_ratings      Remove rating experiment subfolders'
	@echo
	@echo 'Optional arguments:'
	@echo '  OUTPUT_DIR       Directory to save data files (default: data/raw)'
	@echo '  MIN_RATINGS      Minimum number of ratings threshold (default: 25)'
	@echo
	@echo 'Example:'
	@echo '  make all OUTPUT_DIR=custom/path MIN_RATINGS=50'

# requirements
.PHONY: requirements format lint
requirements: 
	uv sync

format: 
	uv run ruff format .

lint:
	uv run ruff check .

## fetch raw data from BigQuery
.PHONY: data
data: 
	uv run -m src.data.get_raw_data

## model types
CURRENT_YEAR = 2025
TRAIN_YEAR = $(CURRENT_YEAR) -4
TUNE_YEAR = $(TEST_YEAR) -2

CATBOOST ?= catboost
LIGHTGBM ?= lightgbm
LINEAR ?= linear
TREE ?= tree
LIGHTGBM_LINEAR ?= lightgbm_linear

## train hurdle moodel
HURDLE_CANDIDATE ?= lightgbm-hurdle

train_hurdle:
	uv run -m src.models.hurdle \
	--experiment $(HURDLE_CANDIDATE) \
	--preprocessor-type linear
	--model logistic

finalize_hurdle: 
	uv run -m src.models.finalize_model \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

score_hurdle: 
	uv run -m src.models.score \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

hurdle: train_hurdle finalize_hurdle score_hurdle


## complexity model
COMPLEXITY_CANDIDATE ?= catboost-complexity
COMPLEXITY_PREDICTIONS ?= models/experiments/predictions/catboost-complexity.parquet
train_complexity:
	uv run -m src.models.complexity \
	--preprocessor-type tree \
	--model catboost \
	--use-sample-weights \
	--experiment $(COMPLEXITY_CANDIDATE)

finalize_complexity: 
	uv run -m src.models.finalize_model \
	--model-type complexity \
	--experiment $(COMPLEXITY_CANDIDATE)

score_complexity: 
	uv run -m src.models.score \
	--model-type complexity \
	--experiment $(COMPLEXITY_CANDIDATE)

complexity: train_complexity finalize_complexity score_complexity

## rating model
RATING_CANDIDATE ?= catboost-rating
train_rating:
	uv run -m src.models.rating \
	--use-sample-weights \
	--min-ratings 5 \
	--complexity-experiment catboost-complexity \
	--local-complexity-path $(COMPLEXITY_PREDICTIONS) \
	--experiment $(RATING_CANDIDATE)

finalize_rating: 
	uv run -m src.models.finalize_model \
	--model-type rating \
	--experiment $(RATING_CANDIDATE)

score_rating:
	uv run -m src.models.score \
	--model-type rating \
	--experiment $(RATING_CANDIDATE) \
	--complexity-predictions $(COMPLEXITY_PREDICTIONS)

rating: train_rating finalize_rating score_rating

## users rated
USERS_RATED_MODEL ?= $(LIGHTGBM)
USERS_RATED_CANDIDATE ?= $(USERS_RATED_MODEL)-users_rated
train_users_rated:
	uv run -m src.models.users_rated \
	--preprocessor-type tree \
	--model $(USERS_RATED_MODEL) \
	--complexity-experiment catboost-complexity \
	--local-complexity-path $(COMPLEXITY_PREDICTIONS) \
	--experiment $(USERS_RATED_CANDIDATE) \
	--min-ratings 0

finalize_users_rated: 
	uv run -m src.models.finalize_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE)

score_users_rated:
	uv run -m src.models.score \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--complexity-predictions $(COMPLEXITY_PREDICTIONS)

users_rated: train_users_rated finalize_users_rated score_users_rated

# run all models_models: complexiity rating users_rated

# predict geek rating given models
geek_rating: 
	uv run -m src.models.geek_rating \
	--start-year 2024 \
	--end-year 2029 \
	--hurdle $(HURDLE_CANDIDATE) \
	--complexity $(COMPLEXITY_CANDIDATE)
	--rating $(RATING_CANDIDATE) \
	--users-rated $(USERS_RATED_CANDIDATE)
	--experiment calculated-geek-rating

# evaluate
OUTPUT_DIR ?= ./models/experiments
.PHONY: train
train:
	uv run -m src.models.time_based_evaluation \
	--start-year 2021 \
	--end-year 2022 \
	--output-dir $(OUTPUT_DIR) \
    --model-args \
        hurdle.preprocessor-type=tree \
        hurdle.model=lightgbm \
        complexity.preprocessor-type=tree \
        complexity.model=catboost \
        complexity.use-sample-weights=true \
        rating.preprocessor-type=tree \
        rating.model=catboost \
        rating.min-ratings=5 \
        rating.use-sample-weights=true \
        users_rated.preprocessor-type=tree \
        users_rated.model=lightgbm_linear \
        users_rated.min-ratings=0

# predictions
predictions: 
	uv run predict.py \
	--start-year 0 \
	--end-year 2029 \
	--hurdle $(HURDLE_CANDIDATE) \
	--complexity $(COMPLEXITY_CANDIDATE) \
	--rating $(RATING_CANDIDATE)
	--users-rated $(USERS_RATED_CANDIDATE)

### register model candidates
# register models
register_complexity:
	uv run -m scoring_service.register_model \
	--model-type complexity \
	--experiment $(COMPLEXITY_CANDIDATE) \
	--name complexity-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting game complexity"

register_rating:
	uv run -m scoring_service.register_model \
	--model-type rating \
	--experiment $(RATING_CANDIDATE) \
	--name rating-$(CURRENT_YEAR)
	--description "Production (v$(CURRENT_YEAR)) model for predicting game rating"

register_users_rated:
	uv run -m scoring_service.register_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--name users_rated-v2025 \
	--description "Production (v$(CURRENT_YEAR)) model for predicting users_rated"

register_hurdle:
	uv run -m scoring_service.register_model \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE) \
	--name hurdle-v2025 \
	--description "Production (v$(CURRENT_YEAR)) model for predicting whether games will achieve ratings (hurdle)"

.PHONY: register_complexity register_rating register_users_rated register_hurdle register
register: register_complexity register_rating register_users_rated register_hurdle

### train model candidates
models: hurdle complexity rating users_rated

## view experiments
experiment_dashboard:
	uv run streamlit run src/monitor/experiment_dashboard.py

# dashboard to look at predicted geek rating
geek_rating_dashboard:
	uv run streamlit run src/monitor/geek_rating_dashboard.py

# dashboard to look at unsupervised learning methods
unsupervised_dashboard:
	uv run streamlit run src/monitor/unsupervised_dashboard.py
	
# remove trained experiments
.PHONY: clean_experiments
clean_experiments:
	@echo "This will delete all subfolders in models/experiments/"
	@read -p "Are you sure? (y/n) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf models/experiments/*/; \
		echo "Subfolders deleted."; \
	else \
		echo "Aborted."; \
	fi

# upload experiments to Google Cloud Storage
.PHONY: upload_experiments
upload_experiments:
	uv run -m src.utils.sync_experiments --create-bucket

.PHONY: download_experiments
download_experiments:
	uv run -m src.utils.sync_experiments --download

# dockerfile training locally
.PHONY: docker-training docker-scoring scoring-service
docker-training:
	docker build -f Dockerfile.training -t bgg-training:test . \
	&& docker run -it \
	--env-file .env \
	bgg-training:test python -c "import os; print('Environment Variables:'); print(f'GCP_PROJECT_ID: {os.getenv(\"GCP_PROJECT_ID\")}'); \print(f'GCS_BUCKET_NAME: {os.getenv(\"GCS_BUCKET_NAME\")}')"

# dockerfile scoring locally
docker-scoring:
	docker build -f Dockerfile.scoring -t bgg-scoring:test . \
	&& docker run -it \
	-p 8080:8080 \
	--env-file .env \
	bgg-scoring:test

# run scoring service locally
scoring-service:
	uv run -m scoring_service.score \
    --service-url http://localhost:8080 \
    --start-year $(CURRENT_YEAR)-1 \
    --end-year $(CURRENT_YEAR)+5 \
    --hurdle-model hurdle-v2025 \
    --complexity-model complexity-v2025 \
    --rating-model rating-v2025 \
    --users-rated-model users_rated-v2025 \
    --download
