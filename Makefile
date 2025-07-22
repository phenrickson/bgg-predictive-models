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
HURDLE_CANDIDATE ?= test-hurdle

train_hurdle:
	uv run -m src.models.hurdle \
	--experiment $(HURDLE_CANDIDATE)

finalize_hurdle: 
	uv run -m src.models.finalize_model \
	--model-type hurdle --experiment $(HURDLE_CANDIDATE)

score_hurdle: 
	uv run -m src.models.score \
	--model-type hurdle \
	--experiment $(HURDLE_CANDIDATE)

hurdle: train_hurdle finalize_hurdle score_hurdle

## complexity model
COMPLEXITY_CANDIDATE ?= test-complexity
train_complexity:
	uv run -m src.models.complexity \
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
RATING_CANDIDATE ?= test-rating
train_rating:
	uv run -m src.models.rating \
	--complexity-experiment test-complexity \
	--local-complexity-path data/estimates/test-complexity_complexity_predictions.parquet \
	--experiment $(RATING_CANDIDATE)

finalize_rating: 
	uv run -m src.models.finalize_model \
	--model-type rating \
	--experiment $(RATING_CANDIDATE)

score_rating:
	uv run -m src.models.score \
	--model-type rating \
	--experiment $(RATING_CANDIDATE) \
	--complexity-predictions data/estimates/test-complexity_complexity_predictions.parquet \


rating: train_rating finalize_rating score_rating

## users rated model
USERS_RATED_CANDIDATE ?= test-users_rated
train_users_rated:
	uv run -m src.models.users_rated \
	--complexity-experiment test-complexity \
	--local-complexity-path data/estimates/test-complexity_complexity_predictions.parquet \
	--experiment $(USERS_RATED_CANDIDATE)

finalize_users_rated: 
	uv run -m src.models.finalize_model \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE)

score_users_rated:
	uv run -m src.models.score \
	--model-type users_rated \
	--experiment $(USERS_RATED_CANDIDATE) \
	--complexity-predictions data/estimates/test-complexity_complexity_predictions.parquet \

users_rated: train_users_rated finalize_users_rated score_users_rated

# evaluate
evaluation:
	uv run -m src.models.time_based_evaluation \
	--start-year 2014 \
	--end-year 2015 \
    --model-args \
        hurdle.preprocessor-type=linear \
        hurdle.model=logistic \
        complexity.preprocessor-type=tree \
        complexity.model=lightgbm \
        rating.preprocessor-type=linear \
        rating.model=linear \
		rating.min-ratings=5 \
		rating.use-sample-weights=True \
		users_rated.preprocessor-type=tree \
		users_rated.model=lightgbm \
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