# Makefile for BGG predictive models

# Default settings
RAW_DIR := data/raw

.PHONY: help clean all

help:  ## Show this help message
	@echo 'Usage:'
	@echo '  make help                        Show this help message'
	@echo '  make requirements                Install/update Python dependencies'
	@echo '  make format                      Format code using ruff'
	@echo '  make lint                        Lint code using ruff'
	@echo '  make fix                         Fix linting issues using ruff'
	@echo '  make test                        Run tests using pytest'
	@echo '  make data                        Fetch training data from BigQuery'
	@echo '  make models                      Train all model candidates'
	@echo '  make register                    Register all models to scoring service'
	@echo '  make register_embeddings         Register embeddings model to embeddings service'
	@echo '  make clean_experiments           Remove all experiment subfolders'
	@echo '  make clean_predictions           Remove data/prediction subfolders'
	@echo '  make years                       Show year configuration for model training'
	@echo '  make evaluate                    Evaluate models over time using config.yaml'
	@echo '  make evaluate-verbose            Run evaluation with verbose logging'
	@echo '  make evaluate-dry-run            Show what evaluation would do without running'
	@echo '  make predictions                 Generate predictions using trained models'
	@echo '  make embeddings                  Train all embedding models (pca, svd, umap)'
	@echo '  make embeddings_pca              Train PCA embeddings'
	@echo '  make embeddings_svd              Train SVD embeddings'
	@echo '  make embeddings_umap             Train UMAP embeddings'
	@echo '  make embeddings_autoencoder      Train Autoencoder embeddings (requires torch)'
	@echo '  make text_embeddings             Train text embeddings from descriptions (PMI+SVD)'
	@echo '  make register_text_embeddings   Register text embeddings model to GCS'
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

## fetch training data from BigQuery
.PHONY: data
data:
	uv run -m src.pipeline.data --model hurdle
	uv run -m src.pipeline.data --model complexity

# model types
LINEAR ?= linear
RIDGE ?= ridge
LOGISTIC ?= logistic
CATBOOST ?= catboost
LIGHTGBM ?= lightgbm
LIGHTGBM_LINEAR ?= lightgbm_linear

# set defaults

## train all model candidates and predict geek rating
.PHONY: models
models: hurdle complexity rating users_rated geek_rating

# train individual models
hurdle: train_hurdle
complexity: train_complexity score_complexity
rating: train_rating
users_rated: train_users_rated

## train individual models
# hurdle model
train_hurdle:
	uv run -m src.pipeline.train \
	--model hurdle

score_hurdle:
	uv run -m src.pipeline.score \
	--model hurdle

# complexity
train_complexity: 
	uv run -m src.pipeline.train \
	--model complexity

score_complexity:
	uv run -m src.pipeline.score \
	--model complexity \
	--all-years

# rating
train_rating: 
	uv run -m src.pipeline.train \
	--model rating

# users rated
# rating
train_users_rated: 
	uv run -m src.pipeline.train \
	--model users_rated

# geek rating
geek_rating:
	uv run -m src.models.outcomes.geek_rating

## finalize
finalize:
	uv run -m src.pipeline.finalize

# evaluate models over time periods
.PHONY: evaluate
evalute: 
	uv run -m src.pipeline.evaluate \ 
	--simulate

## embeddings models (settings from config.yaml, data from BigQuery)
.PHONY: embeddings embeddings_pca embeddings_svd embeddings_autoencoder
embeddings: embeddings_pca embeddings_svd embeddings_autoencoder

embeddings_pca:
	uv run -m src.models.embeddings.train --algorithm pca

embeddings_svd:
	uv run -m src.models.embeddings.train --algorithm svd

embeddings_autoencoder:
	uv run -m src.models.embeddings.train --algorithm autoencoder

## text embeddings (word embeddings from descriptions)
.PHONY: text_embeddings text_embeddings_pmi
text_embeddings: text_embeddings_pmi

text_embeddings_pmi:
	uv run -m src.models.text_embeddings.train --algorithm pmi

## text embeddings registration and scoring
TEXT_EMBEDDINGS_CANDIDATE ?= text-embeddings

register_text_embeddings:
	uv run -m text_embeddings_service.register_model \
	--experiment $(TEXT_EMBEDDINGS_CANDIDATE) \
	--name text-embeddings-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) text embeddings for game descriptions"



# evaluate over time using config.yaml settings
.PHONY: evaluate evaluate-verbose evaluate-dry-run evaluate-simulation
evaluate:
	uv run -m src.pipeline.evaluate

evaluate-verbose:  ## Run evaluation with verbose logging
	uv run -m src.pipeline.evaluate --verbose

evaluate-dry-run:  ## Show what evaluation would do without running
	uv run -m src.pipeline.evaluate --dry-run --verbose

evaluate-simulation:  ## Run simulation-based evaluation
	uv run -m src.pipeline.evaluate_simulation --save-predictions

### register model candidates
.PHONY: register_complexity register_rating register_users_rated register_hurdle register_embeddings register_text_embeddings register
register: register_complexity register_rating register_users_rated register_hurdle register_embeddings register_text_embeddings

# register models
register_complexity:
	uv run -m scoring_service.register_model \
	--model complexity \
	--experiment $(COMPLEXITY_CANDIDATE) \
	--name complexity-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting game complexity"

register_rating:
	uv run -m scoring_service.register_model \
	--model rating \
	--experiment $(RATING_CANDIDATE) \
	--name rating-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting game rating"

register_users_rated:
	uv run -m scoring_service.register_model \
	--model users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--name users_rated-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting users_rated"

register_hurdle:
	uv run -m scoring_service.register_model \
	--model hurdle \
	--experiment $(HURDLE_CANDIDATE) \
	--name hurdle-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) model for predicting whether games will achieve ratings (hurdle)"

EMBEDDINGS_CANDIDATE ?= svd-embeddings
register_embeddings:
	uv run -m embeddings_service.register_model \
	--experiment $(EMBEDDINGS_CANDIDATE) \
	--name embeddings-v$(CURRENT_YEAR) \
	--description "Production (v$(CURRENT_YEAR)) SVD embeddings for game similarity"

## dashboard
.PHONY: streamlit dashboard
streamlit dashboard:
	uv run streamlit run src/streamlit/Home.py

## view experiments
experiments:
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
# Use ENVIRONMENT=prod or ENVIRONMENT=dev to specify, or ENVIRONMENT=auto to detect from git branch
.PHONY: upload_experiments
upload-experiments:
	uv run -m src.utils.sync_experiments --create-bucket $(if $(ENVIRONMENT),--environment $(ENVIRONMENT),)

.PHONY: download_experiments
download-experiments:
	uv run -m src.utils.sync_experiments --download $(if $(ENVIRONMENT),--environment $(ENVIRONMENT),)

# Setup git hooks for automatic experiment syncing
.PHONY: setup-hooks
setup-hooks:
	git config core.hooksPath .githooks
	@echo "Git hooks configured to use .githooks directory"

# dockerfile training locally
.PHONY: docker-training docker-scoring scoring-service
docker-training:
	docker build -f docker/training.Dockerfile -t bgg-training:test . \
	&& docker run -it \
	--env-file .env \
	bgg-training:test python -c "import os; print('Environment Variables:'); print(f'GCP_PROJECT_ID: {os.getenv(\"GCP_PROJECT_ID\")}'); print(f'GCS_BUCKET_NAME: {os.getenv(\"GCS_BUCKET_NAME\")}')"


# run scoring service with credentials mounted
docker-scoring:
	docker build -f docker/scoring.Dockerfile -t bgg-scoring-service .

start-scoring:
	docker run -d \
	-p 8087:8080 \
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
	docker build -f docker/streamlit.Dockerfile -t bgg-streamlit:test .

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
