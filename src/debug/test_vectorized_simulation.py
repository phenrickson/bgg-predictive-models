"""Test vectorized simulate_batch() performance."""

import time
import numpy as np
import pandas as pd
from pathlib import Path

# Load models from existing experiments
from src.models.outcomes.simulation import (
    simulate_batch,
    precompute_cholesky,
    sample_posterior,
    compute_cholesky,
)
from src.models.outcomes.data import load_data
from src.models.outcomes.base import DataConfig
import joblib


def load_pipeline(model_type: str, experiment_name: str) -> object:
    """Load a trained pipeline from an experiment."""
    base_dir = Path("models/experiments")
    exp_dir = base_dir / model_type / experiment_name

    # Find latest version
    versions = sorted(exp_dir.glob("v*"), key=lambda x: int(x.name[1:]))
    if not versions:
        raise FileNotFoundError(f"No versions found in {exp_dir}")

    # Try pipeline.pkl first, then model.joblib
    model_path = versions[-1] / "pipeline.pkl"
    if not model_path.exists():
        model_path = versions[-1] / "model.joblib"

    return joblib.load(model_path)


def main():
    print("=" * 60)
    print("Testing Vectorized simulate_batch()")
    print("=" * 60)

    # Load models from 2023 evaluation
    print("\nLoading models...")
    complexity_pipeline = load_pipeline("complexity", "eval-complexity-2023")
    rating_pipeline = load_pipeline("rating", "eval-rating-2023")
    users_rated_pipeline = load_pipeline("users_rated", "eval-users_rated-2023")
    print("  Loaded complexity, rating, users_rated pipelines")

    # Load test data
    print("\nLoading test data...")
    data_config = DataConfig(
        min_ratings=0,
        requires_complexity_predictions=False,
        supports_embeddings=True,
    )

    df = load_data(
        data_config=data_config,
        start_year=2023,
        end_year=2023,
        use_embeddings=True,
        apply_filters=False,
    )

    df_pandas = df.to_pandas()

    # Filter to games with valid ratings
    valid_mask = (
        ~df_pandas["rating"].isna()
        & ~df_pandas["users_rated"].isna()
        & (df_pandas["users_rated"] > 0)
    )
    df_valid = df_pandas[valid_mask].reset_index(drop=True)

    print(f"  Total games: {len(df_pandas)}")
    print(f"  Valid games (with ratings): {len(df_valid)}")

    # Test with different sizes
    n_samples = 500
    test_sizes = [100, 500, 1000, len(df_valid)]

    print("\n" + "-" * 60)
    print(f"Benchmarking simulate_batch() with n_samples={n_samples}")
    print("-" * 60)

    # Pre-compute Cholesky once
    cholesky_cache = precompute_cholesky(
        complexity_pipeline,
        rating_pipeline,
        users_rated_pipeline,
    )
    print("Pre-computed Cholesky decompositions")

    for n_games in test_sizes:
        if n_games > len(df_valid):
            continue

        subset = df_valid.head(n_games).reset_index(drop=True)

        start = time.perf_counter()
        results = simulate_batch(
            subset,
            complexity_pipeline,
            rating_pipeline,
            users_rated_pipeline,
            n_samples=n_samples,
            prior_rating=5.5,
            prior_weight=2000,
            random_state=42,
            cholesky_cache=cholesky_cache,
        )
        elapsed = time.perf_counter() - start

        # Quick sanity check
        r = results[0]

        print(f"\n{n_games} games:")
        print(f"  Time: {elapsed:.2f}s ({elapsed/n_games*1000:.1f}ms/game)")
        print(f"  Results: {len(results)} SimulationResult objects")
        print(f"  Sample game: {r.game_name[:40]}...")
        print(f"    complexity: {r.complexity_point:.2f} (median: {np.median(r.complexity_samples):.2f})")
        print(f"    rating: {r.rating_point:.2f} (median: {np.median(r.rating_samples):.2f})")
        print(f"    geek_rating: {r.geek_rating_point:.2f} (median: {np.median(r.geek_rating_samples):.2f})")

    # Compare with individual model simulation
    print("\n" + "-" * 60)
    print("Comparison: sample_posterior() for individual models")
    print("-" * 60)

    n_games = 1000
    subset = df_valid.head(n_games).reset_index(drop=True)

    # Complexity model
    cholesky_L = compute_cholesky(complexity_pipeline.named_steps["model"])
    start = time.perf_counter()
    complexity_samples = sample_posterior(
        complexity_pipeline,
        subset,
        n_samples=n_samples,
        include_noise=True,
        random_state=42,
        cholesky_L=cholesky_L,
    )
    elapsed_complexity = time.perf_counter() - start

    print(f"\nComplexity only ({n_games} games, {n_samples} samples):")
    print(f"  Time: {elapsed_complexity:.2f}s")
    print(f"  Shape: {complexity_samples.shape}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
