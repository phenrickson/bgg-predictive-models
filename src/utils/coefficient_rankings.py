"""Utilities for extracting and ranking coefficient estimates by feature category."""

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Feature category prefixes matching the one-hot encoded column names
FEATURE_CATEGORIES = {
    "Designer": "designer_",
    "Publisher": "publisher_",
    "Artist": "artist_",
    "Mechanic": "mechanic_",
    "Category": "category_",
    "Family": "family_",
}


def extract_category_coefficients(
    coefficients_df: pd.DataFrame,
    category: str,
    min_abs_coefficient: float = 0.0,
) -> pd.DataFrame:
    """Extract and clean coefficients for a single feature category.

    Args:
        coefficients_df: DataFrame with at least 'feature' and 'coefficient' columns.
        category: One of the keys in FEATURE_CATEGORIES.
        min_abs_coefficient: Filter out coefficients with absolute value below this.

    Returns:
        DataFrame with columns: name, coefficient, std, lower_95, upper_95, etc.
        Sorted by coefficient descending.
    """
    if category not in FEATURE_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Choose from: {list(FEATURE_CATEGORIES.keys())}")

    prefix = FEATURE_CATEGORIES[category]
    mask = coefficients_df["feature"].str.startswith(prefix)
    filtered = coefficients_df[mask].copy()

    if filtered.empty:
        return filtered

    # Strip prefix and clean up names
    filtered["name"] = (
        filtered["feature"]
        .str[len(prefix):]
        .str.replace("_", " ")
        .str.strip()
        .str.title()
    )

    # Filter by minimum absolute coefficient
    if min_abs_coefficient > 0:
        filtered = filtered[filtered["coefficient"].abs() >= min_abs_coefficient]

    # Filter out zero coefficients (ARD shrinks many to exactly zero)
    filtered = filtered[filtered["coefficient"] != 0]

    # Sort by coefficient descending
    filtered = filtered.sort_values("coefficient", ascending=False).reset_index(drop=True)

    return filtered


def rank_coefficients_across_experiments(
    experiment_coefficients: Dict[str, pd.DataFrame],
    category: str,
    top_n: int = 30,
    min_abs_coefficient: float = 0.0,
) -> pd.DataFrame:
    """Rank entities across multiple experiments (e.g., eval years).

    Computes the mean coefficient across experiments for ranking, and
    preserves per-experiment values for showing stability over time.

    Args:
        experiment_coefficients: Dict mapping experiment name to coefficients DataFrame.
        category: Feature category to extract.
        top_n: Number of top entities to return.
        min_abs_coefficient: Minimum absolute coefficient to include.

    Returns:
        DataFrame with columns: name, mean_coefficient, mean_std, n_experiments,
        plus per-experiment coefficient columns.
    """
    frames = []
    for exp_name, coeff_df in experiment_coefficients.items():
        extracted = extract_category_coefficients(
            coeff_df, category, min_abs_coefficient
        )
        if not extracted.empty:
            extracted["experiment"] = exp_name
            frames.append(extracted[["name", "coefficient", "std", "experiment"]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Aggregate across experiments
    summary = (
        combined.groupby("name")
        .agg(
            mean_coefficient=("coefficient", "mean"),
            mean_std=("std", "mean"),
            min_coefficient=("coefficient", "min"),
            max_coefficient=("coefficient", "max"),
            n_experiments=("coefficient", "count"),
        )
        .reset_index()
    )

    # Sort by absolute mean coefficient and take top N
    summary["abs_mean"] = summary["mean_coefficient"].abs()
    summary = summary.nlargest(top_n, "abs_mean").drop(columns=["abs_mean"])
    summary = summary.sort_values("mean_coefficient", ascending=False).reset_index(drop=True)

    return summary


def get_category_summary(
    coefficients_df: pd.DataFrame,
) -> pd.DataFrame:
    """Get a summary of how many features exist per category.

    Args:
        coefficients_df: DataFrame with 'feature' and 'coefficient' columns.

    Returns:
        DataFrame with columns: category, total_features, nonzero_features.
    """
    rows = []
    for category, prefix in FEATURE_CATEGORIES.items():
        mask = coefficients_df["feature"].str.startswith(prefix)
        total = mask.sum()
        nonzero = (mask & (coefficients_df["coefficient"] != 0)).sum()
        rows.append({"category": category, "total_features": total, "nonzero_features": nonzero})

    return pd.DataFrame(rows)
