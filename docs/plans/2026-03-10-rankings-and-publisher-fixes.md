# Rankings Page & Publisher Allow List Fixes

## Summary

Added a Rankings page to the Streamlit dashboard for exploring model coefficient estimates by feature category (designer, publisher, mechanic, etc.). Fixed the publisher allow list to match actual BGG data strings and made filtering case-insensitive.

## Rankings Page

New page: `src/streamlit/pages/6 Rankings.py`

Visualizes top designers, mechanics, publishers, etc. by their estimated effect on each outcome, with uncertainty from Bayesian model coefficients.

### Features
- Select outcome, feature category, and source (finalized model or eval experiments)
- Top N selector (10-500, steps of 10)
- Filter to positive or negative effects only
- **Finalized Model view**: dot plot with 95% CI error bars from posterior std
- **Eval Experiments view**: mean coefficient with min/max range across eval years, per-year dot plot in expander
- Data table below each chart

### Supporting code
- `src/utils/coefficient_rankings.py` — utility for filtering/ranking coefficients by feature category prefix
- `src/utils/experiment_loader.py` — added `load_coefficients()` and `load_all_coefficients()` methods for GCS

## Publisher Allow List Fixes

File: `src/features/transformers.py`

### Problem

The `ALLOWED_PUBLISHER_NAMES` set contained strings that didn't match the actual publisher names in `bgg-data-warehouse.analytics.games_features`. These publishers were silently dropped during `_filter_publishers`, so models never saw them as features.

### Corrections

| Old (wrong)            | New (matches BGG data)       |
|------------------------|------------------------------|
| `Decision Games`       | `Decision Games (I)`         |
| `(web published)`      | `(Web published)`            |
| `Games Workshop Ltd`   | `Games Workshop Ltd.`        |
| `WizKids`              | `WizKids (I)`                |
| `Mattel, Inc`          | `Mattel, Inc.`               |
| `Greater Than Games`   | `Greater Than Games, LLC`    |
| `Renegade Games`       | `Renegade Game Studios`      |
| `The Game Crafter`     | `The Game Crafter, LLC`      |

### Additions

- `The Avalon Hill Game Co` — 253 games in data (existing `Avalon Hill` entry has 48)

### Case-insensitive filtering

`_filter_publishers` now lowercases both the publisher name and the allow list entries when comparing. The original publisher name from the data is preserved in the output — only the comparison is case-insensitive. This prevents future casing mismatches (e.g., `asmodee` vs `Asmodee`).

## Next steps

- Retrain models to pick up the corrected publisher allow list
- Verify `(Web published)` and other corrected publishers appear in new coefficients
- Future: consider switching publisher matching from names to IDs (requires upstream data pipeline change in BigQuery)
