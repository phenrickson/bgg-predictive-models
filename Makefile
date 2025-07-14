# Makefile for BGG predictive models

# Default settings
RAW_DIR := data/raw

.PHONY: help clean all

help:  ## Show this help message
	@echo 'Usage:'
	@echo '  make help         Show this help message'
	@echo '  make all          Fetch all data from BigQuery'
	@echo '  make clean        Remove generated data files'
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
	uv run -m src.models.hurdle --experiment $(HURDLE_CANDIDATE)

finalize_hurdle: 
	uv run -m src.models.finalize_model --model-type hurdle --experiment $(HURDLE_CANDIDATE)

score_hurdle: 
	uv run -m src.models.score --model-type hurdle --experiment $(HURDLE_CANDIDATE)

hurdle: train_hurdle finalize_hurdle score_hurdle


## complexity model
COMPLEXITY_CANDIDATE ?= test-complexity
train_complexity:
	uv run -m src.models.complexity --experiment $(COMPLEXITY_CANDIDATE)

finalize_complexity: 
	uv run -m src.models.finalize_model --model-type complexity --experiment $(COMPLEXITY_CANDIDATE)

score_complexity: 
	uv run -m src.models.score --model-type complexity --experiment $(COMPLEXITY_CANDIDATE)

complexity: train_complexity finalize_complexity score_complexity


## view experiments
experiment_dashboard:
	uv run streamlit run src/monitor/experiment_dashboard.py

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
