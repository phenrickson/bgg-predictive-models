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
.PHONY: requirements
requirements: 
	uv sync

# Target for features file - this is what keeps track of freshness
$(RAW_DIR)/game_features.parquet: src/data/games_features_materialized_view.sql src/data/get_raw_data.py src/data/loader.py
	uv run -m src.data.get_raw_data

## fetch raw data from BigQuery
.PHONY: features_data
features_data: $(RAW_DIR)/game_features.parquet

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

hurdle: train_hurdle finalize_hurdle score_hurdle

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
USERS_RATED_CANDIDATE ?= test-users_rated
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
USERS_RATED_TREE_CANDIDATE ?= lightgbm-linear-users_rated
train_users_rated_tree:
	uv run -m src.models.users_rated \
	--preprocessor-type tree \
	--model lightgbm_linear \
	--complexity-experiment catboost-complexity \
	--local-complexity-path models/experiments/predictions/catboost-complexity.parquet \
	--experiment $(USERS_RATED_CANDIDATE) \
	--min-ratings=0

finalize_users_rated_tree: 
	uv run -m src.models.finalize_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE)

score_users_rated_tree:
	uv run -m src.models.score \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--complexity-predictions models/experiments/predictions/catboost-complexity.parquet

users_rated_tree: train_users_rated_tree finalize_users_rated_tree score_users_rated_tree

# evaluate
evaluation:
	uv run -m src.models.time_based_evaluation \
	--start-year 2014 \
	--end-year 2021 \
    --model-args \
        hurdle.preprocessor-type=linear \
        hurdle.model=logistic \
        complexity.preprocessor-type=tree \
        complexity.model=lightgbm \
        rating.preprocessor-type=linear \
        rating.model=linear \
		rating.min-ratings=5 \
		rating.use-sample-weights \
		users_rated.preprocessor-type=tree \
		users_rated.model=lightgbm_linear \
		users_rated.min-ratings=0

## view experiments
experiment_dashboard:
	uv run streamlit run src/monitor/experiment_dashboard.py

geek_rating_dashboard:
	uv run streamlit run src/monitor/geek_rating_dashboard.py
	
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
	uv run -m src.models.upload_experiments
