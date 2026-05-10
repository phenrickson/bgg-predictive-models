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

Output: ``_snapshots/v{N}/simulations/{name}/v{M}/top_{top_n}_games.png``
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

    outcomes = ["complexity", "rating", "users_rated", "geek_rating"]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--split", type=str, default="standard")
    parser.add_argument("--simulation-name", type=str, default="default")
    parser.add_argument("--simulation-version", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    path = plot_top_games(
        snapshot_version=args.snapshot_version,
        simulation_name=args.simulation_name,
        split_name=args.split,
        simulation_version=args.simulation_version,
        top_n=args.top_n,
        base_dir=args.base_dir,
    )
    print(f"plot: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
