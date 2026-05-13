"""Forest plots for snapshot-aware simulation results.

Reads predictions.parquet from a simulation directory and emits a
forest plot of the top N games by predicted geek_rating. The plot
shows 90% and 50% credible intervals for each outcome (complexity,
rating, users_rated, geek_rating) with filled dots for predicted
medians and open red circles for realized actuals.

CLI::

    uv run python -m src.pipeline.plot_simulation \\
        --snapshot-version 1 \\
        [--simulation-name default] [--simulation-version N] \\
        [--top-n 100]

Output: ``models/bgg/snapshots/v{N}/simulations/{name}/v{M}/top_{top_n}_games.png``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.logging import setup_logging

# Fallback chain for CJK glyphs in game titles. DejaVu Sans (matplotlib default)
# can't render katakana/CJK; Hiragino Sans and Arial Unicode MS both ship with
# macOS and cover what we need. Order = priority.
plt.rcParams["font.family"] = ["DejaVu Sans", "Hiragino Sans", "Arial Unicode MS"]

logger = logging.getLogger(__name__)


def plot_top_games(
    snapshot_version: int,
    simulation_name: str = "default",
    split_name: str = "standard",
    simulation_version: Optional[int] = None,
    top_n: int = 100,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> Path:
    """Plot the top-N games by predicted geek_rating from a simulation run."""
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)

    sim = storage.load_simulation(simulation_name, split_name, version=simulation_version)
    if sim is None:
        raise FileNotFoundError(
            f"No simulation {simulation_name}/{split_name}/v{simulation_version or 'latest'} "
            f"under snapshot v{snapshot_version}"
        )

    # Resolve actual version used by load_simulation (latest if None was passed)
    resolved_version = sim["registration"]["version"]
    sim_dir = storage.simulation_dir(simulation_name, split_name, resolved_version)

    eval_year = sim["registration"].get("eval_year", "unknown")
    df = sim["predictions"].to_pandas()

    df_sorted = df.sort_values("geek_rating_point", ascending=False).head(top_n).reset_index(drop=True)

    df_sorted["label"] = df_sorted["name"].apply(
        lambda n: f"{str(n)[:30]}..." if len(str(n)) > 30 else str(n)
    )

    outcomes = ["complexity", "users_rated", "rating", "geek_rating"]
    n_games = len(df_sorted)

    fig, axes = plt.subplots(1, 4, figsize=(20, max(12, n_games * 0.15)), sharey=True)
    y_positions = np.arange(n_games)

    for i, outcome in enumerate(outcomes):
        ax = axes[i]
        lower_90 = df_sorted[f"{outcome}_q05"]
        upper_90 = df_sorted[f"{outcome}_q95"]
        lower_50 = df_sorted[f"{outcome}_q25"]
        upper_50 = df_sorted[f"{outcome}_q75"]
        median = df_sorted[f"{outcome}_median"]
        actual = df_sorted[f"{outcome}_actual"]

        for j in range(n_games):
            ax.plot(
                [lower_90.iloc[j], upper_90.iloc[j]],
                [y_positions[j], y_positions[j]],
                color="steelblue", linewidth=1, alpha=0.6,
            )
        for j in range(n_games):
            ax.plot(
                [lower_50.iloc[j], upper_50.iloc[j]],
                [y_positions[j], y_positions[j]],
                color="steelblue", linewidth=3, alpha=0.8,
            )

        ax.scatter(median, y_positions, color="steelblue", s=25, zorder=5, label="Predicted")
        ax.scatter(
            actual, y_positions,
            facecolors="none", edgecolors="red", s=40, linewidths=1.5, zorder=6,
            label="Actual",
        )

        ax.set_xlabel(outcome.replace("_", " ").title(), fontsize=11)
        ax.set_title(outcome.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(df_sorted["label"], fontsize=7)
    axes[0].set_ylabel("Game", fontsize=11)
    axes[0].invert_yaxis()

    plt.suptitle(
        f"Top {top_n} Games by Predicted Geek Rating - {eval_year} ({split_name})\n"
        "(line = 90% interval, thick = 50% interval, dot = predicted, circle = actual)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    plot_path = sim_dir / f"top_{top_n}_games.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {plot_path}")
    return plot_path


def plot_predicted_vs_actual(
    snapshot_version: int,
    simulation_name: str = "default",
    split_name: str = "standard",
    simulation_version: Optional[int] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
) -> Path:
    """Scatter predicted vs actual for each outcome (2x2 grid).

    Drops rows where actual is null (games without ratings/votes yet).
    """
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)

    sim = storage.load_simulation(simulation_name, split_name, version=simulation_version)
    if sim is None:
        raise FileNotFoundError(
            f"No simulation {simulation_name}/{split_name}/v{simulation_version or 'latest'} "
            f"under snapshot v{snapshot_version}"
        )

    resolved_version = sim["registration"]["version"]
    sim_dir = storage.simulation_dir(simulation_name, split_name, resolved_version)
    eval_year = sim["registration"].get("eval_year", "unknown")
    df = sim["predictions"].to_pandas()

    outcomes = ["complexity", "users_rated", "rating", "geek_rating"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Drop "no data" rows so metrics + scatter reflect rows with real signal:
    # - complexity / rating: actual != 0
    # - users_rated: keep all (0 is real)
    # - geek_rating: drop rows whose users_rated_actual == 0; color the
    #   remaining ones orange when users_rated < 25 (low-data, dominated by
    #   the bayes prior) and steelblue otherwise. Title carries both n/r/rmse
    #   pairs (overall and rated-only).
    #
    # users_rated_actual in the predictions parquet is log-scale (log1p),
    # so the 25-rating threshold maps to log1p(25) ≈ 3.258.
    MIN_RATINGS = 25
    LOG_MIN_RATINGS = float(np.log1p(MIN_RATINGS))

    for ax, outcome in zip(axes.ravel(), outcomes):
        sub = df[[f"{outcome}_point", f"{outcome}_actual"]].dropna()
        if outcome in ("complexity", "rating"):
            sub = sub[sub[f"{outcome}_actual"] != 0]
        elif outcome == "geek_rating":
            ur_actual = df.loc[sub.index, "users_rated_actual"]
            sub = sub[ur_actual != 0]
        pred = sub[f"{outcome}_point"].to_numpy()
        actual = sub[f"{outcome}_actual"].to_numpy()
        n = len(sub)

        if n == 0:
            ax.text(0.5, 0.5, "no actuals", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(outcome.replace("_", " ").title(), fontsize=12, fontweight="bold")
            continue

        # Color rule applies to complexity / rating / geek_rating: orange when
        # users_rated_actual < log1p(25), blue otherwise. users_rated panel
        # has no second axis (the x-axis already IS users_rated) so leave it
        # uncolored.
        if outcome in ("complexity", "rating", "geek_rating"):
            ur_actual_arr = df.loc[sub.index, "users_rated_actual"].to_numpy()
            low_mask = ur_actual_arr < LOG_MIN_RATINGS
            rated_mask = ~low_mask
            ax.scatter(
                pred[low_mask], actual[low_mask],
                alpha=0.4, s=12, color="darkorange", edgecolors="none",
                label=f"<{MIN_RATINGS} ratings (n={low_mask.sum()})",
            )
            ax.scatter(
                pred[rated_mask], actual[rated_mask],
                alpha=0.5, s=15, color="steelblue", edgecolors="none",
                label=f"≥{MIN_RATINGS} ratings (n={rated_mask.sum()})",
            )
        else:
            ax.scatter(pred, actual, alpha=0.4, s=15, color="steelblue", edgecolors="none")
            rated_mask = None

        lo = float(min(pred.min(), actual.min()))
        hi = float(max(pred.max(), actual.max()))
        ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, linestyle="--",
                alpha=0.7, label="y = x")

        corr_all = float(np.corrcoef(pred, actual)[0, 1]) if n > 1 else float("nan")
        rmse_all = float(np.sqrt(np.mean((pred - actual) ** 2)))

        # OLS fit through all points; if we have a rated subset, also fit
        # through just those. Draw both lines so the user can see how the
        # low-information points pull the overall fit away from the y=x.
        xs = np.linspace(lo, hi, 100)
        if n > 1:
            slope_all, intercept_all = np.polyfit(pred, actual, 1)
            ax.plot(
                xs, slope_all * xs + intercept_all,
                color="darkorange", linewidth=1.5, alpha=0.9,
                label=f"fit all (slope={slope_all:.2f})",
            )

        if rated_mask is not None and rated_mask.sum() > 1:
            r_rated = float(np.corrcoef(pred[rated_mask], actual[rated_mask])[0, 1])
            rmse_rated = float(np.sqrt(np.mean((pred[rated_mask] - actual[rated_mask]) ** 2)))
            slope_r, intercept_r = np.polyfit(pred[rated_mask], actual[rated_mask], 1)
            ax.plot(
                xs, slope_r * xs + intercept_r,
                color="steelblue", linewidth=1.5, alpha=0.9,
                label=f"fit ≥{MIN_RATINGS} (slope={slope_r:.2f})",
            )
            title = (
                f"{outcome.replace('_', ' ').title()}  "
                f"(n={n}, r={corr_all:.3f}, rmse={rmse_all:.3f})\n"
                f"≥{MIN_RATINGS} only (n={rated_mask.sum()}, r={r_rated:.3f}, rmse={rmse_rated:.3f})"
            )
        else:
            title = (
                f"{outcome.replace('_', ' ').title()}  "
                f"(n={n}, r={corr_all:.3f}, rmse={rmse_all:.3f})"
            )

        ax.legend(loc="lower right", fontsize=7, framealpha=0.7)

        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Predicted vs Actual - {eval_year} ({split_name})",
        fontsize=13, y=1.00,
    )
    plt.tight_layout()

    plot_path = sim_dir / "predicted_vs_actual.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {plot_path}")
    return plot_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--split", type=str, default="standard")
    parser.add_argument("--simulation-name", type=str, default=None,
                        help="Override config.simulation.experiment_name")
    parser.add_argument("--simulation-version", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    simulation_name = args.simulation_name
    if simulation_name is None:
        from src.utils.config import load_config
        cfg = load_config()
        simulation_name = cfg.simulation.experiment_name if cfg.simulation else "default"

    path = plot_top_games(
        snapshot_version=args.snapshot_version,
        simulation_name=simulation_name,
        split_name=args.split,
        simulation_version=args.simulation_version,
        top_n=args.top_n,
        base_dir=args.base_dir,
    )
    print(f"plot: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
