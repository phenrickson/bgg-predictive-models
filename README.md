## Creating Materialized Views

### Games Features Materialized View

To create the games features materialized view, use `uv run`:

```bash
# Ensure you have set the BGG_DATASET environment variable if not using the default
export BGG_DATASET=your_dataset_name

# Run the script using uv
uv run src/data/create_view.py
```

#### Environment Variables

- `BGG_DATASET`: The BigQuery dataset where the materialized view will be created
  - Default: `bgg_data_dev`
  - Set this to match your project's dataset configuration

#### Prerequisites

- Google Cloud credentials configured
- BigQuery access
- Python dependencies installed via `uv`
