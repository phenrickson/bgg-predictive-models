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

# Target for features file - this is what keeps track of freshness
$(RAW_DIR)/game_features.parquet: src/data/games_features_materialized_view.sql src/data/get_raw_data.py src/data/loader.py
	uv run -m src.data.get_raw_data

## fetch raw data from BigQuery
.PHONY: data
data: $(RAW_DIR)/game_features.parquet

## train hurdle moodel
HURDLE_CANDIDATE ?= linear-hurdle

train_hurdle:
	uv run -m src.models.hurdle \
	--experiment $(HURDLE_CANDIDATE) \
	--preprocessor-type linear \
	--model logistic

finalize_hurdle: 
	uv run -m src.models.finalize_model \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

score_hurdle: 
	uv run -m src.models.score \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

hurdle: train_hurdle finalize_hurdle score_hurdle_tree

## train hurdle moodel
HURDLE_CANDIDATE_TREE ?= lightgbm-hurdle

train_hurdle_tree:
	uv run -m src.models.hurdle \
	--experiment $(HURDLE_CANDIDATE_TREE) \
	--preprocessor-type tree \
	--model lightgbm

finalize_hurdle_tree: 
	uv run -m src.models.finalize_model \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE_TREE)

score_hurdle_tree: 
	uv run -m src.models.score \
	--model-type hurdle
	--experiment $(HURDLE_CANDIDATE_TREE)

hurdle_tree: train_hurdle_tree finalize_hurdle_tree score_hurdle

## complexity model
COMPLEXITY_CANDIDATE ?= test-complexity
train_complexity:
	uv run -m src.models.complexity \
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

## complexity model
COMPLEXITY_TREE_CANDIDATE ?= catboost-complexity
train_complexity_tree:
	uv run -m src.models.complexity \
	--preprocessor-type tree \
	--model catboost \
	--use-sample-weights \
	--experiment $(COMPLEXITY_TREE_CANDIDATE)

finalize_complexity_tree: 
	uv run -m src.models.finalize_model \
	--model-type complexity \
	--experiment $(COMPLEXITY_TREE_CANDIDATE)

score_complexity_tree: 
	uv run -m src.models.score \
	--model-type complexity \
	--experiment $(COMPLEXITY_TREE_CANDIDATE)

complexity_tree: train_complexity_tree finalize_complexity_tree score_complexity_tree

## rating model
RATING_CANDIDATE ?= linear-rating
train_rating:
	uv run -m src.models.rating \
	--use-sample-weights \
	--min-ratings 5 \
	--complexity-experiment test-complexity \
	--local-complexity-path models/experiments/predictions/test-complexity.parquet \
	--experiment $(RATING_CANDIDATE)

finalize_rating: 
	uv run -m src.models.finalize_model \
	--model-type rating \
	--experiment $(RATING_CANDIDATE)

score_rating:
	uv run -m src.models.score \
	--model-type rating \
	--experiment $(RATING_CANDIDATE) \
	--complexity-predictions models/experiments/predictions/test-complexity.parquet

rating: train_rating finalize_rating score_rating

## rating model
RATING_TREE_CANDIDATE = catboost-rating
train_rating_tree:
	uv run -m src.models.rating \
	--use-sample-weights \
	--preprocessor-type tree \
	--model catboost \
	--min-ratings 5 \
	--complexity-experiment catboost-complexity \
	--local-complexity-path models/experiments/predictions/catboost-complexity.parquet \
	--experiment $(RATING_TREE_CANDIDATE)

finalize_rating_tree: 
	uv run -m src.models.finalize_model \
	--model-type rating \
	--experiment $(RATING_TREE_CANDIDATE)

score_rating_tree:
	uv run -m src.models.score \
	--model-type rating \
	--experiment $(RATING_TREE_CANDIDATE) \
	--complexity-predictions models/experiments/predictions/catboost-complexity.parquet
	
rating_tree: train_rating_tree finalize_rating_tree score_rating_tree

## users rated model
USERS_RATED_CANDIDATE ?= linear-users_rated
train_users_rated:
	uv run -m src.models.users_rated \
	--complexity-experiment test-complexity \
	--local-complexity-path models/experiments/predictions/test-complexity.parquet \
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
	--complexity-predictions models/experiments/predictions/test-complexity.parquet

users_rated: train_users_rated finalize_users_rated score_users_rated

## users rated with catboost
## users rated model
USERS_RATED_TREE_CANDIDATE ?= lightgbm-users_rated
train_users_rated_tree:
	uv run -m src.models.users_rated \
	--preprocessor-type tree \
	--model lightgbm \
	--complexity-experiment catboost-complexity \
	--local-complexity-path models/experiments/predictions/catboost-complexity.parquet \
	--experiment $(USERS_RATED_TREE_CANDIDATE) \
	--min-ratings 0

finalize_users_rated_tree: 
	uv run -m src.models.finalize_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_TREE_CANDIDATE)

score_users_rated_tree:
	uv run -m src.models.score \
	--model-type users_rated \
	--experiment $(USERS_RATED_TREE_CANDIDATE) \
	--complexity-predictions models/experiments/predictions/catboost-complexity.parquet

users_rated_tree: train_users_rated_tree finalize_users_rated_tree score_users_rated_tree

# run all models
linear_models: hurdle complexity rating users_rated
tree_models: complexity_tree rating_tree users_rated_tree

# predict geek rating given models
geek_rating: 
	uv run -m src.models.geek_rating \
	--start-year 2024 \
	--end-year 2029 \
	--hurdle linear-hurdle \
	--complexity catboost-complexity \
	--rating catboost-rating \
	--users-rated lightgbm-users_rated \
	--experiment calculated-geek-rating

# predictions
predictions: 
	uv run predict.py \
	--start-year 0 \
	--end-year 2029 \
	--hurdle linear-hurdle \
	--complexity catboost-complexity \
	--rating catboost-rating \
	--users-rated lightgbm-users_rated

# evaluate
evaluation:
	uv run -m src.models.time_based_evaluation \
	--start-year 2016 \
	--end-year 2021 \
    --model-args \
        hurdle.preprocessor-type=linear \
        hurdle.model=logistic \
        complexity.preprocessor-type=tree \
        complexity.model=catboost \
		complexty.model=use-sample-weights \
        rating.preprocessor-type=tree \
        rating.model=catboost \
		rating.min-ratings=5 \
		rating.use-sample-weights \
		users_rated.preprocessor-type=tree \
		users_rated.model=lightgbm \
		users_rated.min-ratings=0

### register model candidates
# register models
register_complexity:
	uv run -m scoring_service.register_model \
	--model-type complexity \
	--experiment catboost-complexity \
	--name complexity-v2025 \
	--description "Production (v2025) model for predicting game complexity"

register_rating:
	uv run -m scoring_service.register_model \
	--model-type rating \
	--experiment catboost-rating \
	--name rating-v2025 \
	--description "Production (v2025) model for predicting game rating"

register_users_rated:
	uv run -m scoring_service.register_model \
	--model-type users_rated \
	--experiment lightgbm-users_rated \
	--name users_rated-v2025 \
	--description "Production (v2025) model for predicting users_rated"

register_hurdle:
	uv run -m scoring_service.register_model \
	--model-type hurdle \
	--experiment lightgbm-hurdle \
	--name hurdle-v2025 \
	--description "Production (v2025) model for predicting whether games will achieve ratings (hurdle)"

.PHONY: register_complexity register_rating register_users_rated register_hurdle register
register: register_complexity register_rating register_users_rated register_hurdle

## train finalize and register models
model_hurdle: train_hurdle_tree finalize_hurdle_tree register_hurdle
model_complexity: train_complexity_tree finalize_complexity_tree register_complexity
model_rating: train_rating_tree finalize_rating_tree register_rating
model_users_rated: train_users_rated_tree finalize_users_rated_tree register_users_rated

### train model candidates
models: model_hurdle model_complexity model_rating model_users_rated

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

# remove rating experiments
.PHONY: clean_ratings
clean_ratings:
	@echo "This will delete rating experiment subfolders in models/experiments/"
	@read -p "Are you sure? (y/n) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf models/experiments/rating/*/; \
		echo "Rating experiment subfolders deleted."; \
	else \
		echo "Aborted."; \
	fi

# remove users rated experiments
.PHONY: clean_users_rated
clean_users_rated:
	@echo "This will delete users rated experiment subfolders in models/experiments/"
	@read -p "Are you sure? (y/n) " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf models/experiments/users_rated/*/; \
		echo "Users rated experiment subfolders deleted."; \
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
make docker-training:
	docker build -f Dockerfile.training -t bgg-training:test . \
	&& docker run -it \
	--env-file .env \
	bgg-training:test python -c "import os; print('Environment Variables:'); print(f'GCP_PROJECT_ID: {os.getenv(\"GCP_PROJECT_ID\")}'); \print(f'GCS_BUCKET_NAME: {os.getenv(\"GCS_BUCKET_NAME\")}')"

# dockerfile scoring locally
make docker-scoring:
	docker build -f Dockerfile.scoring -t bgg-scoring:test . \
	&& docker run -it \
	-p 8080:8080 \
	--env-file .env \
	bgg-scoring:test