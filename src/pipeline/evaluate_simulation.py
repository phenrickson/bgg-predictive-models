"""Snapshot-aware simulation evaluator (per-split).

Loads per-split pipelines for the four-model chain (complexity → rating
+ users_rated → geek_rating), runs the chained-Bayesian simulation on
the year immediately after the split's test fold, and writes per-game
predictions plus end-to-end metrics under
``models/bgg/snapshots/v{N}/simulations/{name}/{split_name}/v{M}/``.

Eval year is derived from the split's ``test_through + 1``. This is the
genuinely-held-out year — the chain is being asked to predict games it
has not seen at any stage. Repeat across all YoY splits to get a
year-over-year picture of methodology performance.

CLI::

    uv run python -m src.pipeline.evaluate_simulation \\
        --snapshot-version 1 --split standard \\
        [--simulation-name default] \\
        [--candidates complexity=ard-complexity,...] \\
        [--n-samples 500]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl

from src.models.candidate_config import list_candidates
from src.models.outcomes.simulation import (
    simulate_batch,
    compute_simulation_metrics,
    precompute_cholesky,
)
from src.models.snapshot_storage import DEFAULT_BASE_DIR, SnapshotStorage
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def evaluate_simulation(
    snapshot_version: int,
    split_name: str = "standard",
    simulation_name: Optional[str] = None,
    candidates: Optional[Dict[str, str]] = None,
    n_samples: int = 500,
    geek_rating_mode: Optional[str] = None,
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    random_state: int = 42,
) -> int:
    """Run chain simulation on the year after the split's test fold."""
    if geek_rating_mode is None or simulation_name is None:
        from src.utils.config import load_config
        cfg = load_config()
        if geek_rating_mode is None:
            geek_rating_mode = (
                cfg.simulation.geek_rating_mode if cfg.simulation else "bayesian"
            )
        if simulation_name is None:
            simulation_name = (
                cfg.simulation.experiment_name if cfg.simulation else "default"
            )
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    split = storage.load_split(split_name)
    if split is None:
        raise FileNotFoundError(f"Split {split_name!r} not found in v{snapshot_version}")
    split_meta = split.get("metadata") or {}
    test_through = split_meta.get("test_through")
    if test_through is None:
        raise ValueError(
            f"Split {split_name!r} has no test_through in its metadata"
        )
    eval_year = int(test_through) + 1

    if candidates is None:
        candidates = {}
    for model_type in ["complexity", "rating", "users_rated", "geek_rating"]:
        if model_type not in candidates:
            cands = list_candidates(model_type)
            if not cands:
                raise ValueError(f"No candidates for {model_type} in config.yaml")
            candidates[model_type] = cands[0]

    # Load per-split FINALIZED pipelines (refit on train+tune+test). The
    # finalize step must have been run for this split first.
    pipelines: Dict[str, Any] = {}
    candidate_versions: Dict[str, int] = {}
    for model_type, cand in candidates.items():
        versions = storage.list_candidate_versions(model_type, cand)
        if not versions:
            raise FileNotFoundError(
                f"No versions for {model_type}/{cand} in v{snapshot_version}"
            )
        v = versions[-1]
        finalized = storage.load_finalized_pipeline(model_type, cand, v, split_name)
        if finalized is None:
            raise FileNotFoundError(
                f"No finalized.pkl for {model_type}/{cand}/v{v} on split {split_name!r} — "
                f"run `just bgg-finalize-yoy` (or `just bgg-finalize`) before simulating"
            )
        pipelines[model_type] = finalized
        candidate_versions[model_type] = v

    eval_df = universe.filter(pl.col("year_published") == eval_year)
    if eval_df.height == 0:
        raise ValueError(
            f"No games in universe with year_published == {eval_year} "
            f"(split {split_name}'s test_through + 1)"
        )
    eval_pd = eval_df.to_pandas()
    logger.info(
        f"Simulating {eval_df.height} games from {eval_year} "
        f"(split={split_name}, test_through={test_through})"
    )

    cholesky_cache = precompute_cholesky(
        complexity_pipeline=pipelines["complexity"],
        rating_pipeline=pipelines["rating"],
        users_rated_pipeline=pipelines["users_rated"],
        geek_rating_pipeline=(
            pipelines["geek_rating"] if geek_rating_mode != "bayesian" else None
        ),
    )

    results = simulate_batch(
        games=eval_pd,
        complexity_pipeline=pipelines["complexity"],
        rating_pipeline=pipelines["rating"],
        users_rated_pipeline=pipelines["users_rated"],
        n_samples=n_samples,
        random_state=random_state,
        cholesky_cache=cholesky_cache,
        geek_rating_mode=geek_rating_mode,
        geek_rating_pipeline=pipelines.get("geek_rating"),
    )

    predictions_rows = []
    for r in results:
        s = r.summary()
        row = {"game_id": r.game_id, "name": r.game_name}
        for outcome in ["complexity", "rating", "users_rated", "geek_rating"]:
            o = s[outcome]
            row.update({
                f"{outcome}_actual": o["actual"],
                f"{outcome}_point": o["point"],
                f"{outcome}_median": o["median"],
                f"{outcome}_mean": o["mean"],
                f"{outcome}_std": o["std"],
                f"{outcome}_q05": o["interval_90"][0],
                f"{outcome}_q95": o["interval_90"][1],
                f"{outcome}_q25": o["interval_50"][0],
                f"{outcome}_q75": o["interval_50"][1],
            })
        predictions_rows.append(row)
    predictions = pl.DataFrame(predictions_rows, infer_schema_length=None)

    metrics = compute_simulation_metrics(results)

    sim_version = storage.next_simulation_version(simulation_name, split_name)
    registration = {
        "snapshot_version": snapshot_version,
        "split_name": split_name,
        "simulation_name": simulation_name,
        "version": sim_version,
        "created_at": datetime.now().isoformat(),
        "candidates": candidates,
        "candidate_versions": candidate_versions,
        "test_through": int(test_through),
        "eval_year": eval_year,
        "n_samples": n_samples,
        "geek_rating_mode": geek_rating_mode,
        "n_eval_games": len(results),
    }
    storage.save_simulation(simulation_name, split_name, sim_version, registration, metrics, predictions)
    logger.info(f"Wrote simulation {simulation_name}/{split_name}/v{sim_version}")

    # Emit plots alongside the artifacts.
    try:
        from src.pipeline.plot_simulation import plot_predicted_vs_actual, plot_top_games
        plot_top_games(
            snapshot_version=snapshot_version,
            simulation_name=simulation_name,
            split_name=split_name,
            simulation_version=sim_version,
            top_n=100,
            base_dir=base_dir,
        )
        plot_predicted_vs_actual(
            snapshot_version=snapshot_version,
            simulation_name=simulation_name,
            split_name=split_name,
            simulation_version=sim_version,
            base_dir=base_dir,
        )
    except Exception as e:
        logger.warning(f"Skipped plot for {simulation_name}/{split_name}/v{sim_version}: {e}")

    return sim_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--split", type=str, default="standard",
                        help="Split name. Eval year = split's test_through + 1.")
    parser.add_argument("--simulation-name", type=str, default=None,
                        help="Override config.simulation.experiment_name")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Comma-separated overrides like 'rating=catboost-rating'")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--geek-rating-mode", type=str, default=None,
                        choices=["bayesian", "stacking", "direct"],
                        help="Override config.simulation.geek_rating_mode")
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    candidates: Dict[str, str] = {}
    if args.candidates:
        for pair in args.candidates.split(","):
            k, v = pair.split("=", 1)
            candidates[k.strip()] = v.strip()

    simulation_name = args.simulation_name
    if simulation_name is None:
        from src.utils.config import load_config
        cfg = load_config()
        simulation_name = cfg.simulation.experiment_name if cfg.simulation else "default"

    version = evaluate_simulation(
        snapshot_version=args.snapshot_version,
        split_name=args.split,
        simulation_name=simulation_name,
        candidates=candidates,
        n_samples=args.n_samples,
        geek_rating_mode=args.geek_rating_mode,
        base_dir=args.base_dir,
    )
    print(f"simulation: {simulation_name}/{args.split}/v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
