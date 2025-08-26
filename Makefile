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

## set years for training, tuning, testing
CURRENT_YEAR = 2025
TRAIN_END_YEAR = $(shell expr $(CURRENT_YEAR) - 4)
TUNE_END_YEAR = $(shell expr $(TRAIN_END_YEAR) + 1)
TEST_START_YEAR = $(shell expr $(TUNE_END_YEAR) + 1)
TEST_END_YEAR = $(shell expr $(TEST_START_YEAR) + 1)

# show years
.PHONY: years
years: 
	@echo "=== Year Configuration for Model Training ==="
	@echo "Current Year: $(CURRENT_YEAR)"
	@echo ""
	@echo "=== Dataset Year Ranges ==="
	@echo "Training Data:   [earliest] to $(TRAIN_END_YEAR) (exclusive)"
	@echo "Tuning Data:     $(TRAIN_END_YEAR) to $(TUNE_END_YEAR) (inclusive)"
	@echo "Testing Data:    $(TEST_START_YEAR) to $(TEST_END_YEAR) (inclusive)"
	@echo ""
	@echo "=== Calculated Values ==="
	@echo "TRAIN_END_YEAR:  $(TRAIN_END_YEAR)  ($(CURRENT_YEAR) - 4)"
	@echo "TUNE_END_YEAR:   $(TUNE_END_YEAR)   ($(TRAIN_END_YEAR) + 1)"
	@echo "TEST_START_YEAR: $(TEST_START_YEAR) ($(TUNE_END_YEAR) + 1)"
	@echo "TEST_END_YEAR:   $(TEST_END_YEAR)   ($(TEST_START_YEAR) + 1)"
	@echo ""
	@echo "=== Usage in Model Scripts ==="
	@echo "• Training uses all data before $(TRAIN_END_YEAR)"
	@echo "• Tuning/validation uses data from $(TRAIN_END_YEAR) to $(TUNE_END_YEAR)"
	@echo "• Testing uses data from $(TEST_START_YEAR) to $(TEST_END_YEAR)"
	@echo "• Final models are trained on combined train+tune data"
	@echo "• Time-based evaluation uses rolling windows with these ranges"


# model types
LINEAR ?= linear
CATBOOST ?= catboost
LIGHTGBM ?= lightgbm
LIGHTGBM_LINEAR ?= lightgbm_linear

# set defaults
HURDLE_MODEL = $(LINEAR)
COMPLEXITY_MODEL = $(LINEAR)
RATING_MODEL ?= $(LINEAR)
USERS_RATED_MODEL ?= $(LINEAR)

## train all model candidates
.PHONY: models
models: hurdle complexity rating users_rated

## register models
.PHONY: register_complexity register_rating register_users_rated register_hurdle register
register: register_complexity register_rating register_users_rated register_hurdle

# train models
hurdle: train_hurdle finalize_hurdle score_hurdle
complexity: train_complexity finalize_complexity score_complexity
rating: train_rating finalize_rating score_rating
users_rated: train_users_rated finalize_users_rated score_users_rated

## train individual models
# hurdle model
HURDLE_CANDIDATE ?= $(HURDLE_MODEL)-hurdle
train_hurdle:
	uv run -m src.models.hurdle \
	--experiment $(HURDLE_CANDIDATE) \
	--model $(HURDLE_MODEL) \
	--train-end-year $(TRAIN_END_YEAR) \
	--tune-start-year $(TRAIN_END_YEAR) \
	--tune-end-year $(TUNE_END_YEAR) \
	--test-start-year $(TEST_START_YEAR) \
	--test-end-year $(TEST_END_YEAR)

finalize_hurdle: 
	uv run -m src.models.finalize_model \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

score_hurdle: 
	uv run -m src.models.score \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

## complexity model
COMPLEXITY_CANDIDATE ?= $(COMPLEXITY_MODEL)-complexity
COMPLEXITY_PREDICTIONS ?= models/experiments/predictions/?(COMPLEXITY_CANDIDATE).parquet
train_complexity:
	uv run -m src.models.complexity \
	--model $(COMPLEXITY_MODEL) \
	--use-sample-weights \
	--experiment $(COMPLEXITY_CANDIDATE) \
	--train-end-year $(TRAIN_END_YEAR) \
	--tune-start-year $(TRAIN_END_YEAR) \
	--tune-end-year $(TUNE_END_YEAR) \
	--test-start-year $(TEST_START_YEAR) \
	--test-end-year $(TEST_END_YEAR)

finalize_complexity: 
	uv run -m src.models.finalize_model \
	--model-type complexity \
	--experiment $(COMPLEXITY_CANDIDATE)

score_complexity: 
	uv run -m src.models.score \
	--model-type complexity \
	--experiment $(COMPLEXITY_CANDIDATE)

# rating model
RATING_CANDIDATE ?= $(RATING_MODEL)-rating
train_rating:
	uv run -m src.models.rating \
	--use-sample-weights \
	--model $(RATING_MODEL) \
	--complexity-experiment $(COMPLEXITY_CANDIDATE) \
	--local-complexity-path $(COMPLEXITY_PREDICTIONS) \
	--experiment $(RATING_CANDIDATE) \
	--train-end-year $(TRAIN_END_YEAR) \
	--tune-start-year $(TRAIN_END_YEAR) \
	--tune-end-year $(TUNE_END_YEAR) \
	--test-start-year $(TEST_START_YEAR) \
	--test-end-year $(TEST_END_YEAR)

finalize_rating: 
	uv run -m src.models.finalize_model \
	--model-type rating \
	--experiment $(RATING_CANDIDATE)

score_rating:
	uv run -m src.models.score \
	--model-type rating \
	--experiment $(RATING_CANDIDATE) \
	--complexity-predictions $(COMPLEXITY_PREDICTIONS)

## users rated
USERS_RATED_CANDIDATE ?= $(USERS_RATED_MODEL)-users_rated

train_users_rated:
	uv run -m src.models.users_rated \
	--model $(USERS_RATED_MODEL) \
	--complexity-experiment catboost-complexity \
	--local-complexity-path $(COMPLEXITY_PREDICTIONS) \
	--experiment $(USERS_RATED_CANDIDATE) \
	--min-ratings 0 \
	--train-end-year $(TRAIN_END_YEAR) \
	--tune-start-year $(TRAIN_END_YEAR) \
	--tune-end-year $(TUNE_END_YEAR) \
	--test-start-year $(TEST_START_YEAR) \
	--test-end-year $(TEST_END_YEAR)

finalize_users_rated: 
	uv run -m src.models.finalize_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE)

score_users_rated:
	uv run -m src.models.score \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--complexity-predictions $(COMPLEXITY_PREDICTIONS)

# predict geek rating given models
geek_rating: 
	uv run -m src.models.geek_rating \
	--start-year $(CURRENT_YEAR) -1 \
	--end-year $(TEST_END_YEAR) \
	--hurdle $(HURDLE_CANDIDATE) \
	--complexity $(COMPLEXITY_CANDIDATE)
	--rating $(RATING_CANDIDATE) \
	--users-rated $(USERS_RATED_CANDIDATE)
	--experiment estimated-geek-rating

# evaluate over time
.PHONY: evaluate
evaluate:
	uv run -m src.models.time_based_evaluation \
	--start-year 2016
	--end-year 2022 \
	--output-dir ./models/experiments
    --model-args \
        hurdle.model=lightgbm \
        complexity.model=catboost \
        complexity.use-sample-weights=true \
        rating.model=catboost \
        rating.min-ratings=5 \
        rating.use-sample-weights=true \
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
	--name rating-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting game rating"

register_users_rated:
	uv run -m scoring_service.register_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--name users_rated-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting users_rated"

register_hurdle:
	uv run -m scoring_service.register_model \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE) \
	--name hurdle-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting whether games will achieve ratings (hurdle)"


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

# remove local predictions
.PHONY: clean_predictions
clean_predictions:
	@echo "This will delete all subfolders in data/predictions/"
	@read -p "Are you sure? (y/n) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf data/predictions/*/; \
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
    --hurdle-model hurdle-v$(CURRENT_YEAR) \
    --complexity-model complexity-v$(CURRENT_YEAR) \
    --rating-model rating-v$(CURRENT_YEAR) \
    --users-rated-model users_rated-v$(CURRENT_YEAR) \
    --download
