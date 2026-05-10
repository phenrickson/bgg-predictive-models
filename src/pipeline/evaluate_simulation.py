"""Snapshot-aware simulation evaluator.

Loads finalized pipelines for the four-model chain (complexity → rating
+ users_rated → geek_rating) from a snapshot, runs the chained-Bayesian
simulation on the year following ``finalize_through``, and writes per-
game predictions plus end-to-end metrics under
``_snapshots/v{N}/simulations/{name}/v{M}/``.

Eval year is derived from the complexity candidate's
``finalize_through`` (recorded in registration.json by pipeline.finalize).
The simulation evaluates the deployed-style chain on the immediately-
following year — even if that year's actual values aren't fully
realized yet, the per-game predictions are useful to inspect.

CLI::

    uv run python -m src.pipeline.evaluate_simulation \\
        --snapshot-version 1 \\
        [--simulation-name default] \\
        [--candidates complexity=ard-complexity,rating=ard-ridge-rating,...] \\
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
    simulation_name: str = "default",
    candidates: Optional[Dict[str, str]] = None,
    n_samples: int = 500,
    geek_rating_mode: str = "bayesian",
    base_dir: Union[str, Path] = DEFAULT_BASE_DIR,
    random_state: int = 42,
) -> int:
    """Run simulation on the year immediately following finalize_through.

    Returns the simulation version assigned.
    """
    storage = SnapshotStorage(snapshot_version=snapshot_version, base_dir=base_dir)
    universe = storage.load_universe()
    if universe is None:
        raise FileNotFoundError(f"No snapshot v{snapshot_version}")

    if candidates is None:
        candidates = {}
    for model_type in ["complexity", "rating", "users_rated", "geek_rating"]:
        if model_type not in candidates:
            cands = list_candidates(model_type)
            if not cands:
                raise ValueError(f"No candidates for {model_type} in config.yaml")
            candidates[model_type] = cands[0]

    # Load finalized pipelines + figure out eval year from each model's
    # registration. We require all four to share the same finalize_through
    # so the eval year is unambiguous.
    pipelines: Dict[str, Any] = {}
    finalize_throughs: Dict[str, int] = {}
    for model_type, cand in candidates.items():
        versions = storage.list_candidate_versions(model_type, cand)
        if not versions:
            raise FileNotFoundError(
                f"No versions for {model_type}/{cand} in v{snapshot_version}"
            )
        v = versions[-1]
        pipeline = storage.load_finalized_pipeline(model_type, cand, v)
        if pipeline is None:
            raise FileNotFoundError(
                f"No finalized.pkl for {model_type}/{cand}/v{v} — run pipeline.finalize first"
            )
        pipelines[model_type] = pipeline
        reg = storage.load_candidate_registration(model_type, cand, v) or {}
        ft = reg.get("finalize_through")
        if ft is None:
            raise ValueError(
                f"{model_type}/{cand}/v{v} has no finalize_through in its registration. "
                f"Re-finalize with --finalize-through to set it."
            )
        finalize_throughs[model_type] = int(ft)

    distinct = set(finalize_throughs.values())
    if len(distinct) > 1:
        raise ValueError(
            f"Models have inconsistent finalize_through values: {finalize_throughs}. "
            f"Re-finalize all models to a common cutoff."
        )
    finalize_through = next(iter(distinct))
    eval_year = finalize_through + 1

    # Filter universe to the eval year
    eval_df = universe.filter(pl.col("year_published") == eval_year)
    if eval_df.height == 0:
        raise ValueError(
            f"No games in universe with year_published == {eval_year} "
            f"(finalize_through+1). Either bump the snapshot to include "
            f"newer games, or re-finalize at a lower cutoff."
        )
    eval_pd = eval_df.to_pandas()
    logger.info(
        f"Evaluating {eval_df.height} games from year {eval_year} "
        f"(finalize_through={finalize_through})"
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
    predictions = pl.DataFrame(predictions_rows)

    metrics = compute_simulation_metrics(results)

    sim_version = storage.next_simulation_version(simulation_name)
    registration = {
        "snapshot_version": snapshot_version,
        "simulation_name": simulation_name,
        "version": sim_version,
        "created_at": datetime.now().isoformat(),
        "candidates": candidates,
        "finalize_through": finalize_through,
        "eval_year": eval_year,
        "n_samples": n_samples,
        "geek_rating_mode": geek_rating_mode,
        "n_eval_games": len(results),
    }
    storage.save_simulation(simulation_name, sim_version, registration, metrics, predictions)
    logger.info(f"Wrote simulation {simulation_name}/v{sim_version}")

    # Emit the top-N forest plot alongside the artifacts. Lazy import so
    # matplotlib isn't a hard dependency for callers that only consume
    # the metrics/predictions programmatically.
    try:
        from src.pipeline.plot_simulation import plot_top_games
        plot_top_games(
            snapshot_version=snapshot_version,
            simulation_name=simulation_name,
            simulation_version=sim_version,
            top_n=100,
            base_dir=base_dir,
        )
    except Exception as e:
        logger.warning(f"Skipped plot for {simulation_name}/v{sim_version}: {e}")

    return sim_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--snapshot-version", type=int, required=True)
    parser.add_argument("--simulation-name", type=str, default="default")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Comma-separated overrides like 'rating=catboost-rating'")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--geek-rating-mode", type=str, default="bayesian",
                        choices=["bayesian", "stacking", "direct"])
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()

    setup_logging()

    candidates: Dict[str, str] = {}
    if args.candidates:
        for pair in args.candidates.split(","):
            k, v = pair.split("=", 1)
            candidates[k.strip()] = v.strip()

    version = evaluate_simulation(
        snapshot_version=args.snapshot_version,
        simulation_name=args.simulation_name,
        candidates=candidates,
        n_samples=args.n_samples,
        geek_rating_mode=args.geek_rating_mode,
        base_dir=args.base_dir,
    )
    print(f"simulation: {args.simulation_name}/v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
