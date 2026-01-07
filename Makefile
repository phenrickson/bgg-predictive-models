# Makefile for BGG predictive models

# Default settings
RAW_DIR := data/raw

## set years for training, tuning, testing
CURRENT_YEAR = 2025
TRAIN_END_YEAR = $(shell expr $(CURRENT_YEAR) - 4)
TUNE_END_YEAR = $(shell expr $(TRAIN_END_YEAR) + 1)
TEST_START_YEAR = $(shell expr $(TUNE_END_YEAR) + 1)
TEST_END_YEAR = $(shell expr $(TEST_START_YEAR))

# years for evaluation over time
EVAL_START_YEAR = $(shell expr $(CURRENT_YEAR) -5)
EVAL_END_YEAR = $(shell expr $(EVAL_START_YEAR) +4)

# set years for scoring (including current and previous year)
SCORE_START_YEAR = $(shell expr $(CURRENT_YEAR) - 1)
SCORE_END_YEAR = $(shell expr $(CURRENT_YEAR) + 4)

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
	@echo "=== Usage in Model Scripts ==="
	@echo "• Training uses all data before $(TRAIN_END_YEAR)"
	@echo "• Tuning/validation uses data from $(TRAIN_END_YEAR) to $(TUNE_END_YEAR)"
	@echo "• Testing uses data from $(TEST_START_YEAR) to $(TEST_END_YEAR)"
	@echo "• Time-based evaluation uses rolling windows with these ranges"



.PHONY: help clean all

help:  ## Show this help message
	@echo 'Usage:'
	@echo '  make help                        Show this help message'
	@echo '  make requirements                Install/update Python dependencies'
	@echo '  make format                      Format code using ruff'
	@echo '  make lint                        Lint code using ruff'
	@echo '  make fix                         Fix linting issues using ruff'
	@echo '  make test                        Run tests using pytest'
	@echo '  make data                        Fetch raw data from BigQuery'
	@echo '  make models                      Train all model candidates'
	@echo '  make register                    Register all models to scoring service'
	@echo '  make clean_experiments           Remove all experiment subfolders'
	@echo '  make clean_predictions           Remove data/prediction subfolders'
	@echo '  make years                       Show year configuration for model training'
	@echo '  make evaluate                    Evaluate models over time using config.yaml'
	@echo '  make evaluate-verbose            Run evaluation with verbose logging'
	@echo '  make evaluate-dry-run            Show what evaluation would do without running'
	@echo '  make predictions                 Generate predictions using trained models'
	@echo '  make experiment_dashboard        Launch predictions dashboard'
	@echo '  make predictions_dashboard       Launch geek rating dashboard'
	@echo '  make unsupervised_dashboard      Launch unsupervised learning dashboard'
	@echo '  make upload_experiments          Upload experiments to Google Cloud Storage'
	@echo '  make download_experiments        Download experiments from Google Cloud Storage'
	@echo '  make docker-training             Build and run training Docker image locally'
	@echo '  make docker-scoring              Build and run scoring Docker image locally'
	@echo '  make start-scoring               Start scoring service with credentials'
	@echo '  make stop-scoring                Stop scoring service'
	@echo '  make scoring-service             Build and run scoring service locally'
	@echo '  make scoring-service-upload      Build and run scoring service and upload to BigQuery'
	@echo '  make streamlit-build             Build Streamlit Docker image'
	@echo '  make streamlit-run               Run Streamlit Docker container'
	@echo '  make streamlit-stop              Stop Streamlit container'
	@echo '  make streamlit-test              Build and test Streamlit Docker image interactively'

# requirements
.PHONY: requirements format lint
requirements: 
	uv sync

format: 
	uv run ruff format .

lint:
	uv run ruff check .

fix: 
	uv run ruff check . --fix

test:
	uv run -m pytest tests/

## fetch raw data from BigQuery
.PHONY: data
data: 
	uv run -m src.data.get_raw_data

.PHONY: create-view
create-view:
	uv run -m src.data.create_view



# model types
LINEAR ?= linear
RIDGE ?= ridge
LOGISTIC ?= logistic
CATBOOST ?= catboost
LIGHTGBM ?= lightgbm
LIGHTGBM_LINEAR ?= lightgbm_linear

# set defaults
HURDLE_MODEL = $(LIGHTGBM)
COMPLEXITY_MODEL = $(CATBOOST)
RATING_MODEL ?= $(CATBOOST)
USERS_RATED_MODEL ?= $(RIDGE)

## train all model candidates and predict geek rating
.PHONY: models
models: hurdle complexity rating users_rated geek_rating

## register models
.PHONY: register_complexity register_rating register_users_rated register_hurdle register
register: register_complexity register_rating register_users_rated register_hurdle

# train individual models
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
COMPLEXITY_PREDICTIONS ?= models/experiments/predictions/$(COMPLEXITY_CANDIDATE).parquet
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
	--complexity-experiment $(COMPLEXITY_CANDIDATE) \
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
	--start-year $(TEST_START_YEAR) \
	--end-year $(TEST_END_YEAR) \
	--hurdle $(HURDLE_CANDIDATE) \
	--complexity $(COMPLEXITY_CANDIDATE) \
	--rating $(RATING_CANDIDATE) \
	--users-rated $(USERS_RATED_CANDIDATE) \
	--experiment estimated-geek-rating

# evaluate over time using config.yaml settings
.PHONY: evaluate evaluate-verbose evaluate-dry-run
evaluate:
	uv run python evaluate.py

evaluate-verbose:  ## Run evaluation with verbose logging
	uv run python evaluate.py --verbose

evaluate-dry-run:  ## Show what evaluation would do without running
	uv run python evaluate.py --dry-run --verbose

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

## dashboard
dashboard:
	uv run streamlit run src/streamlit/Home.py

## view experiments
experiment_dashboard:
	uv run streamlit run src/monitor/experiment_dashboard.py

# dashboard to look at predicted geek rating
predictions_dashboard:
	uv run streamlit run src/monitor/predictions_dashboard.py

## view experiments
unsupervised_dashboard:
	uv run streamlit run src/monitor/unsupervised_dashboard.py


clean-experiments:
	@uv run python -c "import shutil; from pathlib import Path; \
		p = Path('models/experiments'); \
		dirs = [d for d in p.iterdir() if d.is_dir()] if p.exists() else []; \
		print(f'This will delete {len(dirs)} subfolders in models/experiments/'); \
		confirm = input('Are you sure? (y/n) ') if dirs else 'n'; \
		[shutil.rmtree(d) for d in dirs] if confirm == 'y' else None; \
		print('Subfolders deleted.' if confirm == 'y' and dirs else 'Aborted.' if dirs else 'No subfolders found.')"

# remove local predictions
.PHONY: clean_predictions
clean-data:
	@uv run python -c "import shutil; from pathlib import Path; \
		p = Path('data/predictions'); \
		dirs = [d for d in p.iterdir() if d.is_dir()] if p.exists() else []; \
		print(f'This will delete {len(dirs)} subfolders in data/predictions/'); \
		confirm = input('Are you sure? (y/n) ') if dirs else 'n'; \
		[shutil.rmtree(d) for d in dirs] if confirm == 'y' else None; \
		print('Subfolders deleted.' if confirm == 'y' and dirs else 'Aborted.' if dirs else 'No subfolders found.')"

# upload experiments to Google Cloud Storage
.PHONY: upload_experiments
upload-experiments:
	uv run -m src.utils.sync_experiments --create-bucket

.PHONY: download_experiments
download-experiments:
	uv run -m src.utils.sync_experiments --download

# dockerfile training locally
.PHONY: docker-training docker-scoring scoring-service
docker-training:
	docker build -f Dockerfile.training -t bgg-training:test . \
	&& docker run -it \
	--env-file .env \
	bgg-training:test python -c "import os; print('Environment Variables:'); print(f'GCP_PROJECT_ID: {os.getenv(\"GCP_PROJECT_ID\")}'); print(f'GCS_BUCKET_NAME: {os.getenv(\"GCS_BUCKET_NAME\")}')"


# run scoring service with credentials mounted
docker-scoring:
	docker build -f Dockerfile.scoring -t bgg-scoring-service . \
	
start-scoring:
	docker run -d \
	-p 8080:8080 \
	-v $(PWD)/credentials:/app/credentials \
	-e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account-key.json \
	--env-file .env \
	bgg-scoring-service

stop-scoring:
	@containers=$$(docker ps -q --filter ancestor=bgg-scoring-service); \
	if [ -n "$$containers" ]; then \
		echo "Stopping scoring service containers: $$containers"; \
		docker stop $$containers; \
	else \
		echo "No running scoring service containers found"; \
	fi

# run scoring service locally
scoring-service:
	uv run -m scoring_service.score \
    --service-url http://localhost:8080 \
    --start-year $(SCORE_START_YEAR) \
    --end-year $(SCORE_END_YEAR) \
    --hurdle-model hurdle-v$(CURRENT_YEAR) \
    --complexity-model complexity-v$(CURRENT_YEAR) \
    --rating-model rating-v$(CURRENT_YEAR) \
    --users-rated-model users_rated-v$(CURRENT_YEAR) \
    --download

scoring-service-upload:
	uv run -m scoring_service.score \
    --service-url http://localhost:8080 \
    --start-year $(SCORE_START_YEAR) \
    --end-year $(SCORE_END_YEAR) \
    --hurdle-model hurdle-v$(CURRENT_YEAR) \
    --complexity-model complexity-v$(CURRENT_YEAR) \
    --rating-model rating-v$(CURRENT_YEAR) \
    --users-rated-model users_rated-v$(CURRENT_YEAR) \
	--upload-to-bigquery \
	--download

# Streamlit targets
.PHONY: streamlit-build streamlit-run streamlit-stop

streamlit-build:  ## Build Streamlit Docker image
	docker build -f Dockerfile.streamlit -t bgg-streamlit:test .

streamlit-run:  ## Run Streamlit Docker container
	docker run -d \
	-p 8080:8080 \
	--env-file .env \
	--name bgg-streamlit-container \
	bgg-streamlit:test
	@echo "Streamlit available at: http://localhost:8080"

streamlit-stop:  ## Stop Streamlit container
	@container=$$(docker ps -q --filter name=bgg-streamlit-container); \
	if [ -n "$$container" ]; then \
		echo "Stopping Streamlit container: $$container"; \
		docker stop $$container && docker rm $$container; \
	else \
		echo "No running Streamlit container found"; \
	fi
