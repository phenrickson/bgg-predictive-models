# Makefile for BGG predictive models

# Default variables
OUTPUT_DIR := data/raw
MIN_RATINGS := 25

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

# Target for features file - this is what keeps track of freshness
$(OUTPUT_DIR)/features.parquet: src/data/get_data.py src/data/config.yaml src/data/config.py
	uv run -m src.data.get_data \
		--output-dir $(OUTPUT_DIR) \
		--min-ratings $(MIN_RATINGS)

# Main target that depends on features file
raw_data: $(OUTPUT_DIR)/features.parquet  ## Fetch all data from BigQuery

clean:  ## Remove generated data files
	rm -rf $(OUTPUT_DIR)/*.parquet
