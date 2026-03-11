# Experiments Page: Data Explorer Tab

## Summary

Restructure the Experiments page from 3 tabs to 6 top-level tabs, splitting the monolithic Experiments tab into focused views. Add a new Data tab for visualizing the relationship between features and the outcome in the training data.

## Changes

### Tab restructure

Current: `[Experiments | Models | Metadata]`

New: `[Metrics | Predictions | Coefficients | Data | Models | Metadata]`

The first four tabs share the Outcome + Dataset selectors (moved above the tabs). Metrics, Predictions, and Coefficients contain the existing code extracted from the current Experiments tab.

### Data tab

Toggle between **Raw** and **Preprocessed** views.

#### Raw view (default)

- Dropdown to pick a feature from the experiment's `data/{dataset}.parquet` columns
- Numeric features: scatter plot (feature value vs outcome) with optional LOESS smoother
- List/categorical features (mechanics, categories, etc.): explode the list column, box plot of outcome by category value (top N most frequent)

#### Preprocessed view

- Load `pipeline.pkl`, run only the preprocessor step on the data, cache with `@st.cache_data`
- Feature category filter (reuse existing `FEATURE_CATEGORIES` dict) to narrow the ~2000 engineered features
- Dropdown to pick a feature, scatter/strip plot vs outcome

### Implementation notes

- Add `load_training_data()` to `experiment_loader.py` to load `data/{dataset}.parquet`
- Add `load_preprocessed_data()` that loads the pipeline, runs `pipeline.named_steps['preprocessor'].transform()`, and returns the result
- Both cached with `@st.cache_data`
- Detect column type: if dtype is list → categorical view, if numeric → scatter view
