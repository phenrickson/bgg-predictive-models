"""Compute simulation metrics from predictions DataFrame.

Mirrors the formulas in src/models/outcomes/simulation.py but operates
on flat polars DataFrames (from predictions.parquet) instead of
SimulationResult objects.
"""

from typing import Dict, Any

import numpy as np
import polars as pl

OUTCOMES = ["complexity", "rating", "users_rated", "geek_rating"]


def compute_metrics_for_outcome(
    df: pl.DataFrame, outcome: str
) -> Dict[str, Any]:
    """Compute RMSE, MAE, R2, coverage, and interval width for one outcome.

    Args:
        df: Filtered predictions DataFrame.
        outcome: One of complexity, rating, users_rated, geek_rating.

    Returns:
        Dict with n, rmse_point, rmse_sim, mae_point, mae_sim,
        r2_point, r2_sim, coverage_90, coverage_50,
        interval_width_90, interval_width_50.
    """
    actual_col = f"{outcome}_actual"
    point_col = f"{outcome}_point"
    median_col = f"{outcome}_median"

    # Check columns exist
    required = [actual_col, point_col, median_col]
    if not all(c in df.columns for c in required):
        return {"n": 0}

    # Filter out invalid actuals:
    # complexity/rating/users_rated: skip 0 (means missing)
    # geek_rating: skip where users_rated_actual == 0
    if outcome in ("complexity", "rating", "users_rated"):
        df = df.filter(pl.col(actual_col) != 0)
    elif outcome == "geek_rating":
        if "users_rated_actual" in df.columns:
            df = df.filter(pl.col("users_rated_actual") != 0)

    # Also drop nulls
    df = df.drop_nulls(subset=[actual_col, point_col, median_col])

    n = len(df)
    if n == 0:
        return {"n": 0}

    actuals = df[actual_col].to_numpy()
    points = df[point_col].to_numpy()
    medians = df[median_col].to_numpy()

    # RMSE / MAE / R2 for simulation median
    rmse_sim = float(np.sqrt(np.mean((actuals - medians) ** 2)))
    mae_sim = float(np.mean(np.abs(actuals - medians)))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    ss_res_sim = float(np.sum((actuals - medians) ** 2))
    r2_sim = 1 - ss_res_sim / ss_tot if ss_tot > 0 else 0.0

    # RMSE / MAE / R2 for point estimate
    rmse_point = float(np.sqrt(np.mean((actuals - points) ** 2)))
    mae_point = float(np.mean(np.abs(actuals - points)))
    ss_res_point = float(np.sum((actuals - points) ** 2))
    r2_point = 1 - ss_res_point / ss_tot if ss_tot > 0 else 0.0

    result = {
        "n": n,
        "rmse_point": round(rmse_point, 4),
        "rmse_sim": round(rmse_sim, 4),
        "mae_point": round(mae_point, 4),
        "mae_sim": round(mae_sim, 4),
        "r2_point": round(r2_point, 4),
        "r2_sim": round(r2_sim, 4),
    }

    # Coverage and interval width for 90% and 50%
    for level in [90, 50]:
        lower_col = f"{outcome}_lower_{level}"
        upper_col = f"{outcome}_upper_{level}"
        if lower_col in df.columns and upper_col in df.columns:
            lowers = df[lower_col].to_numpy()
            uppers = df[upper_col].to_numpy()
            in_interval = (actuals >= lowers) & (actuals <= uppers)
            result[f"coverage_{level}"] = round(float(np.mean(in_interval)), 4)
            result[f"interval_width_{level}"] = round(
                float(np.median(uppers - lowers)), 4
            )

    return result


def compute_all_metrics(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute metrics for all outcomes.

    Args:
        df: Filtered predictions DataFrame.

    Returns:
        Dict keyed by outcome name, values are metric dicts.
    """
    return {outcome: compute_metrics_for_outcome(df, outcome) for outcome in OUTCOMES}
