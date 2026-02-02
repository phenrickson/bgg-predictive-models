#!/usr/bin/env python3
"""
Test script demonstrating simulation-based uncertainty estimation for geek rating.

This script shows how to:
1. Sample from the complexity model's posterior
2. For each complexity sample, get conditional predictions from rating/users_rated
3. Combine to get a distribution of geek rating predictions

Uses models trained to predict 2024 games (eval-*-2024 experiments).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import joblib

from src.models.experiments import ExperimentTracker
from src.models.outcomes.data import load_data
from src.models.outcomes.complexity import ComplexityModel
from src.utils.config import load_config


@dataclass
class SimulationResult:
    """Results from a geek rating simulation."""
    game_id: int
    game_name: str
    n_samples: int

    # Actual values
    actual_complexity: Optional[float]
    actual_rating: Optional[float]
    actual_users_rated: Optional[float]
    actual_geek_rating: Optional[float]

    # Simulation samples (arrays)
    complexity_samples: np.ndarray
    rating_samples: np.ndarray
    users_rated_samples: np.ndarray  # log scale
    geek_rating_samples: np.ndarray

    # Point estimates
    complexity_point: float
    rating_point: float
    users_rated_point: float
    geek_rating_point: float

    @property
    def users_rated_count_samples(self) -> np.ndarray:
        """Users rated in count scale."""
        return np.maximum(np.expm1(self.users_rated_samples), 25)

    def percentile(self, arr: np.ndarray, q: float) -> float:
        """Get percentile of samples."""
        return float(np.percentile(arr, q))

    def interval(self, arr: np.ndarray, level: float = 0.90) -> Tuple[float, float]:
        """Get credible interval."""
        lower = (1 - level) / 2 * 100
        upper = (1 + level) / 2 * 100
        return self.percentile(arr, lower), self.percentile(arr, upper)

    def in_interval(self, actual: float, arr: np.ndarray, level: float = 0.90) -> bool:
        """Check if actual value falls within credible interval."""
        lower, upper = self.interval(arr, level)
        return lower <= actual <= upper

    def summary(self) -> Dict:
        """Return summary statistics."""
        return {
            "game_id": self.game_id,
            "game_name": self.game_name,
            "complexity": {
                "actual": self.actual_complexity,
                "point": self.complexity_point,
                "mean": float(self.complexity_samples.mean()),
                "std": float(self.complexity_samples.std()),
                "interval_90": self.interval(self.complexity_samples, 0.90),
                "in_interval_90": self.in_interval(self.actual_complexity, self.complexity_samples, 0.90) if self.actual_complexity else None,
            },
            "rating": {
                "actual": self.actual_rating,
                "point": self.rating_point,
                "mean": float(self.rating_samples.mean()),
                "std": float(self.rating_samples.std()),
                "interval_90": self.interval(self.rating_samples, 0.90),
                "in_interval_90": self.in_interval(self.actual_rating, self.rating_samples, 0.90) if self.actual_rating else None,
            },
            "users_rated": {
                "actual": self.actual_users_rated,
                "point": self.users_rated_point,
                "mean": float(self.users_rated_count_samples.mean()),
                "std": float(self.users_rated_count_samples.std()),
                "interval_90": self.interval(self.users_rated_count_samples, 0.90),
                "in_interval_90": self.in_interval(self.actual_users_rated, self.users_rated_count_samples, 0.90) if self.actual_users_rated else None,
            },
            "geek_rating": {
                "actual": self.actual_geek_rating,
                "point": self.geek_rating_point,
                "mean": float(self.geek_rating_samples.mean()),
                "std": float(self.geek_rating_samples.std()),
                "interval_90": self.interval(self.geek_rating_samples, 0.90),
                "in_interval_90": self.in_interval(self.actual_geek_rating, self.geek_rating_samples, 0.90) if self.actual_geek_rating else None,
            },
        }

    def print_summary(self):
        """Print formatted summary."""
        s = self.summary()
        print(f"\nGame: {s['game_name']} (ID: {s['game_id']})")
        print("=" * 60)

        for metric in ["complexity", "rating", "users_rated", "geek_rating"]:
            m = s[metric]
            print(f"\n{metric.upper()}:")
            print(f"  Actual:     {m['actual']:.3f}" if m['actual'] else "  Actual:     N/A")
            print(f"  Point:      {m['point']:.3f}")
            print(f"  Mean±Std:   {m['mean']:.3f} ± {m['std']:.3f}")
            low, high = m['interval_90']
            print(f"  90% CI:     [{low:.3f}, {high:.3f}]")
            if m['in_interval_90'] is not None:
                print(f"  In 90% CI:  {m['in_interval_90']}")


def load_eval_model(model_type: str, test_year: int = 2024):
    """Load the eval model for a specific test year."""
    tracker = ExperimentTracker(model_type)
    experiment_name = f"eval-{model_type}-{test_year}"

    experiments = tracker.list_experiments()
    matching = [e for e in experiments if e["name"] == experiment_name]
    if not matching:
        raise ValueError(f"No experiment found: {experiment_name}")

    latest = max(matching, key=lambda x: x["version"])
    experiment = tracker.load_experiment(latest["name"], latest["version"])

    pipeline_path = experiment.exp_dir / "pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"No pipeline found at {pipeline_path}")

    pipeline = joblib.load(pipeline_path)
    return pipeline, experiment


def get_scaler_params_for_column(pipeline, column_name: str) -> Tuple[float, float, int]:
    """Extract mean, std, and index for a column from the preprocessor's scaler."""
    preprocessor = pipeline.named_steps["preprocessor"]
    scaler = preprocessor.named_steps["scaler"]

    feature_names = None
    for name, step in reversed(list(preprocessor.named_steps.items())):
        if name == "scaler":
            continue
        try:
            feature_names = list(step.get_feature_names_out())
            break
        except (AttributeError, TypeError):
            continue

    if feature_names is None:
        raise ValueError("Could not extract feature names from preprocessor")

    if column_name not in feature_names:
        raise ValueError(f"Column {column_name} not found in features")

    idx = feature_names.index(column_name)
    mean = scaler.mean_[idx]
    std = scaler.scale_[idx]

    return mean, std, idx


def build_full_sigma(model) -> np.ndarray:
    """Build full covariance matrix handling ARDRegression's pruned features."""
    coef = model.coef_
    sigma = model.sigma_

    if sigma.shape[0] == len(coef):
        return sigma

    n_features = len(coef)
    full_sigma = np.zeros((n_features, n_features))

    if hasattr(model, "lambda_"):
        active_mask = model.lambda_ < getattr(model, "threshold_lambda", np.inf)
        active_indices = np.where(active_mask)[0]
        for i, ai in enumerate(active_indices):
            for j, aj in enumerate(active_indices):
                full_sigma[ai, aj] = sigma[i, j]
    else:
        n_active = sigma.shape[0]
        full_sigma[:n_active, :n_active] = sigma

    return full_sigma


def compute_cholesky(model) -> np.ndarray:
    """Pre-compute Cholesky decomposition of the covariance matrix.

    This avoids recomputing it on every call to multivariate_normal.
    """
    sigma = build_full_sigma(model)
    # Add small regularization for numerical stability
    return np.linalg.cholesky(sigma + 1e-10 * np.eye(sigma.shape[0]))


def sample_weights_fast(
    coef: np.ndarray,
    cholesky_L: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample weight vectors using pre-computed Cholesky decomposition.

    Much faster than multivariate_normal which recomputes Cholesky each time.
    """
    z = rng.standard_normal((n_samples, len(coef)))
    return coef + z @ cholesky_L.T


def sample_posterior(
    pipeline,
    features: pd.DataFrame,
    n_samples: int = 1000,
    include_noise: bool = True,
    random_state: int = 42,
    cholesky_L: np.ndarray = None,
) -> np.ndarray:
    """Sample from a Bayesian model's posterior predictive distribution.

    Args:
        pipeline: Fitted sklearn pipeline with preprocessor and model.
        features: Input features as DataFrame.
        n_samples: Number of posterior samples.
        include_noise: Whether to add observation noise.
        random_state: Random seed.
        cholesky_L: Pre-computed Cholesky decomposition. If None, computed on the fly.

    Returns:
        Array of shape (n_games, n_samples) with posterior samples.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not (hasattr(model, "coef_") and hasattr(model, "sigma_")):
        raise ValueError(f"Model {type(model).__name__} does not support posterior sampling")

    X_transformed = preprocessor.transform(features)
    if hasattr(X_transformed, "values"):
        X_transformed = X_transformed.values

    rng = np.random.default_rng(random_state)

    # Use pre-computed Cholesky if provided, otherwise compute
    if cholesky_L is None:
        cholesky_L = compute_cholesky(model)

    weight_samples = sample_weights_fast(model.coef_, cholesky_L, n_samples, rng)

    predictions = X_transformed @ weight_samples.T
    if hasattr(model, "intercept_"):
        predictions += model.intercept_

    if include_noise and hasattr(model, "alpha_"):
        noise_std = 1.0 / np.sqrt(model.alpha_)
        predictions += rng.normal(0, noise_std, size=predictions.shape)

    return predictions


def sample_conditional_on_complexity(
    pipeline,
    features: pd.DataFrame,
    complexity_samples: np.ndarray,
    sample_posterior_weights: bool = True,
    include_noise: bool = True,
    random_state: int = 42,
    cholesky_L: np.ndarray = None,
) -> np.ndarray:
    """
    Sample from rating/users_rated posterior conditional on complexity samples.

    Args:
        pipeline: Fitted sklearn pipeline
        features: Base features (without predicted_complexity)
        complexity_samples: Shape (n_games, n_samples)
        sample_posterior_weights: If True, sample from weight posterior
        include_noise: If True, add observation noise
        random_state: Random seed
        cholesky_L: Pre-computed Cholesky decomposition. If None, computed on the fly.

    Returns:
        predictions: Shape (n_games, n_samples)
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    n_games, n_samples = complexity_samples.shape
    rng = np.random.default_rng(random_state)

    # Get scaler params for predicted_complexity
    complexity_mean, complexity_std, complexity_idx = get_scaler_params_for_column(
        pipeline, "predicted_complexity"
    )

    # Transform base features once (with placeholder complexity)
    features_with_complexity = features.copy()
    features_with_complexity["predicted_complexity"] = complexity_samples[:, 0]
    X_base = preprocessor.transform(features_with_complexity)
    if hasattr(X_base, "values"):
        X_base = X_base.values

    predictions = np.zeros((n_games, n_samples))

    if sample_posterior_weights and hasattr(model, "sigma_"):
        # Use pre-computed Cholesky if provided
        if cholesky_L is None:
            cholesky_L = compute_cholesky(model)

        weight_samples = sample_weights_fast(model.coef_, cholesky_L, n_samples, rng)

        for i in range(n_samples):
            scaled_complexity = (complexity_samples[:, i] - complexity_mean) / complexity_std
            X_sample = X_base.copy()
            X_sample[:, complexity_idx] = scaled_complexity

            pred = X_sample @ weight_samples[i]
            if hasattr(model, "intercept_"):
                pred += model.intercept_
            predictions[:, i] = pred

        if include_noise and hasattr(model, "alpha_"):
            noise_std = 1.0 / np.sqrt(model.alpha_)
            predictions += rng.normal(0, noise_std, size=predictions.shape)
    else:
        for i in range(n_samples):
            scaled_complexity = (complexity_samples[:, i] - complexity_mean) / complexity_std
            X_sample = X_base.copy()
            X_sample[:, complexity_idx] = scaled_complexity
            predictions[:, i] = model.predict(X_sample)

    return predictions


def compute_geek_rating(
    rating_samples: np.ndarray,
    users_rated_log_samples: np.ndarray,
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
) -> np.ndarray:
    """Compute geek rating using Bayesian average formula."""
    users_rated_count = np.maximum(np.expm1(users_rated_log_samples), 25)
    rating_clipped = np.clip(rating_samples, 1, 10)

    geek_rating = (
        (rating_clipped * users_rated_count) + (prior_rating * prior_weight)
    ) / (users_rated_count + prior_weight)

    return geek_rating


def simulate_geek_rating(
    game: pd.DataFrame,
    complexity_pipeline,
    rating_pipeline,
    users_rated_pipeline,
    n_samples: int = 1000,
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
    include_noise: bool = True,
    random_state: int = 42,
    cholesky_cache: dict = None,
) -> SimulationResult:
    """
    Run full simulation for a single game.

    Args:
        game: Single-row DataFrame with game features
        complexity_pipeline: Fitted complexity model pipeline
        rating_pipeline: Fitted rating model pipeline
        users_rated_pipeline: Fitted users_rated model pipeline
        n_samples: Number of posterior samples
        prior_rating: Bayesian average prior rating
        prior_weight: Bayesian average prior weight
        include_noise: Whether to include observation noise
        random_state: Random seed for reproducibility
        cholesky_cache: Pre-computed Cholesky decompositions keyed by model name.
                       If None, computed on the fly.

    Returns:
        SimulationResult with all samples and summaries
    """
    game_id = int(game["game_id"].iloc[0])
    game_name = str(game["name"].iloc[0])

    # Extract actuals
    actual_complexity = float(game["complexity"].iloc[0]) if "complexity" in game.columns else None
    actual_rating = float(game["rating"].iloc[0]) if "rating" in game.columns else None
    actual_users_rated = float(game["users_rated"].iloc[0]) if "users_rated" in game.columns else None

    actual_geek_rating = None
    if actual_rating is not None and actual_users_rated is not None:
        actual_geek_rating = (
            (actual_rating * actual_users_rated) + (prior_rating * prior_weight)
        ) / (actual_users_rated + prior_weight)

    # Get pre-computed Cholesky if available
    complexity_L = cholesky_cache.get("complexity") if cholesky_cache else None
    rating_L = cholesky_cache.get("rating") if cholesky_cache else None
    users_rated_L = cholesky_cache.get("users_rated") if cholesky_cache else None

    # Step 1: Sample complexity
    complexity_samples = sample_posterior(
        complexity_pipeline, game, n_samples=n_samples,
        include_noise=include_noise, random_state=random_state,
        cholesky_L=complexity_L
    )
    complexity_samples = np.clip(complexity_samples, 1, 5)

    # Step 2: Sample rating conditional on complexity
    rating_samples = sample_conditional_on_complexity(
        rating_pipeline, game, complexity_samples,
        sample_posterior_weights=True, include_noise=include_noise,
        random_state=random_state + 1, cholesky_L=rating_L
    )
    rating_samples = np.clip(rating_samples, 1, 10)

    # Step 3: Sample users_rated conditional on complexity
    users_rated_samples = sample_conditional_on_complexity(
        users_rated_pipeline, game, complexity_samples,
        sample_posterior_weights=True, include_noise=include_noise,
        random_state=random_state + 2, cholesky_L=users_rated_L
    )

    # Step 4: Compute geek rating
    geek_rating_samples = compute_geek_rating(
        rating_samples, users_rated_samples,
        prior_rating=prior_rating, prior_weight=prior_weight
    )

    # Point estimates
    complexity_point = float(np.clip(complexity_pipeline.predict(game)[0], 1, 5))

    game_with_complexity = game.copy()
    game_with_complexity["predicted_complexity"] = complexity_point

    rating_point = float(np.clip(rating_pipeline.predict(game_with_complexity)[0], 1, 10))
    users_rated_point_log = float(users_rated_pipeline.predict(game_with_complexity)[0])
    users_rated_point = max(np.expm1(users_rated_point_log), 25)

    geek_rating_point = (
        (rating_point * users_rated_point) + (prior_rating * prior_weight)
    ) / (users_rated_point + prior_weight)

    return SimulationResult(
        game_id=game_id,
        game_name=game_name,
        n_samples=n_samples,
        actual_complexity=actual_complexity,
        actual_rating=actual_rating,
        actual_users_rated=actual_users_rated,
        actual_geek_rating=actual_geek_rating,
        complexity_samples=complexity_samples.flatten(),
        rating_samples=rating_samples.flatten(),
        users_rated_samples=users_rated_samples.flatten(),
        geek_rating_samples=geek_rating_samples.flatten(),
        complexity_point=complexity_point,
        rating_point=rating_point,
        users_rated_point=users_rated_point,
        geek_rating_point=geek_rating_point,
    )


def precompute_cholesky(
    complexity_pipeline,
    rating_pipeline,
    users_rated_pipeline,
) -> dict:
    """Pre-compute Cholesky decompositions for all models.

    Call this once at startup, then pass the result to simulate_geek_rating
    for significant speedup (~40x faster sampling).
    """
    return {
        "complexity": compute_cholesky(complexity_pipeline.named_steps["model"]),
        "rating": compute_cholesky(rating_pipeline.named_steps["model"]),
        "users_rated": compute_cholesky(users_rated_pipeline.named_steps["model"]),
    }


def simulate_batch(
    games: pd.DataFrame,
    complexity_pipeline,
    rating_pipeline,
    users_rated_pipeline,
    n_samples: int = 1000,
    prior_rating: float = 5.5,
    prior_weight: float = 2000,
    random_state: int = 42,
    cholesky_cache: dict = None,
) -> list[SimulationResult]:
    """Run simulation for multiple games.

    Args:
        games: DataFrame with multiple games.
        complexity_pipeline: Fitted complexity model pipeline.
        rating_pipeline: Fitted rating model pipeline.
        users_rated_pipeline: Fitted users_rated model pipeline.
        n_samples: Number of posterior samples per game.
        prior_rating: Bayesian average prior rating.
        prior_weight: Bayesian average prior weight.
        random_state: Random seed.
        cholesky_cache: Pre-computed Cholesky decompositions. If None, computed once here.

    Returns:
        List of SimulationResult, one per game.
    """
    # Pre-compute Cholesky if not provided
    if cholesky_cache is None:
        cholesky_cache = precompute_cholesky(
            complexity_pipeline, rating_pipeline, users_rated_pipeline
        )

    results = []
    for i in range(len(games)):
        game = games.iloc[[i]]
        result = simulate_geek_rating(
            game, complexity_pipeline, rating_pipeline, users_rated_pipeline,
            n_samples=n_samples, prior_rating=prior_rating, prior_weight=prior_weight,
            random_state=random_state + i * 10, cholesky_cache=cholesky_cache
        )
        results.append(result)
    return results


def compute_coverage(results: list[SimulationResult], level: float = 0.90) -> Dict:
    """Compute interval coverage across multiple games."""
    coverage = {
        "complexity": [],
        "rating": [],
        "users_rated": [],
        "geek_rating": [],
    }

    for r in results:
        s = r.summary()
        for metric in coverage.keys():
            if s[metric]["in_interval_90"] is not None:
                coverage[metric].append(s[metric]["in_interval_90"])

    return {
        metric: {
            "coverage": sum(vals) / len(vals) if vals else None,
            "n": len(vals),
            "expected": level,
        }
        for metric, vals in coverage.items()
    }


def plot_simulation_results(result: SimulationResult, save_path: Optional[str] = None):
    """Create a 2x2 plot showing posterior distributions for each outcome."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{result.game_name}", fontsize=14, fontweight='bold')

    # Complexity
    ax = axes[0, 0]
    ax.hist(result.complexity_samples, bins=50, density=True, alpha=0.7, color='steelblue', label='Posterior')
    ax.axvline(result.complexity_point, color='darkblue', linestyle='--', linewidth=2, label=f'Point: {result.complexity_point:.2f}')
    if result.actual_complexity:
        ax.axvline(result.actual_complexity, color='red', linestyle='-', linewidth=2, label=f'Actual: {result.actual_complexity:.2f}')
    low, high = result.interval(result.complexity_samples, 0.90)
    ax.axvspan(low, high, alpha=0.2, color='steelblue', label=f'90% CI: [{low:.2f}, {high:.2f}]')
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_xlim(1, 5)

    # Rating
    ax = axes[0, 1]
    ax.hist(result.rating_samples, bins=50, density=True, alpha=0.7, color='forestgreen', label='Posterior')
    ax.axvline(result.rating_point, color='darkgreen', linestyle='--', linewidth=2, label=f'Point: {result.rating_point:.2f}')
    if result.actual_rating:
        ax.axvline(result.actual_rating, color='red', linestyle='-', linewidth=2, label=f'Actual: {result.actual_rating:.2f}')
    low, high = result.interval(result.rating_samples, 0.90)
    ax.axvspan(low, high, alpha=0.2, color='forestgreen', label=f'90% CI: [{low:.2f}, {high:.2f}]')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_xlim(1, 10)

    # Users Rated (count scale, log x-axis)
    ax = axes[1, 0]
    users_count = result.users_rated_count_samples
    # Use log-spaced bins for better visualization
    bins = np.logspace(np.log10(max(users_count.min(), 1)), np.log10(users_count.max()), 50)
    ax.hist(users_count, bins=bins, density=True, alpha=0.7, color='darkorange', label='Posterior')
    ax.axvline(result.users_rated_point, color='orangered', linestyle='--', linewidth=2, label=f'Point: {result.users_rated_point:.0f}')
    if result.actual_users_rated:
        ax.axvline(result.actual_users_rated, color='red', linestyle='-', linewidth=2, label=f'Actual: {result.actual_users_rated:.0f}')
    low, high = result.interval(users_count, 0.90)
    ax.axvspan(low, high, alpha=0.2, color='darkorange', label=f'90% CI: [{low:.0f}, {high:.0f}]')
    ax.set_xlabel('Users Rated')
    ax.set_ylabel('Density')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Geek Rating
    ax = axes[1, 1]
    ax.hist(result.geek_rating_samples, bins=50, density=True, alpha=0.7, color='purple', label='Posterior')
    ax.axvline(result.geek_rating_point, color='darkviolet', linestyle='--', linewidth=2, label=f'Point: {result.geek_rating_point:.2f}')
    if result.actual_geek_rating:
        ax.axvline(result.actual_geek_rating, color='red', linestyle='-', linewidth=2, label=f'Actual: {result.actual_geek_rating:.2f}')
    low, high = result.interval(result.geek_rating_samples, 0.90)
    ax.axvspan(low, high, alpha=0.2, color='purple', label=f'90% CI: [{low:.2f}, {high:.2f}]')
    ax.set_xlabel('Geek Rating')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Simulation-based uncertainty for geek rating")
    parser.add_argument("--year", type=int, default=2023, help="Test year (models trained to predict this year)")
    parser.add_argument("--game-index", type=int, default=None, help="Index of game to use (random if not specified)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of posterior samples")
    parser.add_argument("--plot", action="store_true", help="Show plot of simulation results")
    parser.add_argument("--save-plot", type=str, default=None, help="Path to save plot")
    parser.add_argument("--batch", type=int, default=0, help="Run batch simulation on N games (0 to skip)")
    args = parser.parse_args()

    config = load_config()
    prior_rating = config.scoring.parameters.get("prior_rating", 5.5)
    prior_weight = config.scoring.parameters.get("prior_weight", 2000)

    print(f"=== Loading Models (test year: {args.year}) ===")
    complexity_pipeline, _ = load_eval_model("complexity", args.year)
    rating_pipeline, _ = load_eval_model("rating", args.year)
    users_rated_pipeline, _ = load_eval_model("users_rated", args.year)

    print(f"\n=== Loading Test Data ({args.year}) ===")
    complexity_model = ComplexityModel()
    df = load_data(
        data_config=complexity_model.data_config,
        start_year=args.year,
        end_year=args.year,
        use_embeddings=True,
        apply_filters=True,
    )
    print(f"Loaded {len(df)} games from {args.year}")

    # Select game
    if args.game_index is not None:
        game_idx = args.game_index
    else:
        # Random game
        rng = np.random.default_rng()
        game_idx = rng.integers(0, len(df))

    game = df.slice(game_idx, 1).to_pandas()
    print(f"\nSelected game index: {game_idx}")

    print("\n=== Running Simulation ===")
    result = simulate_geek_rating(
        game, complexity_pipeline, rating_pipeline, users_rated_pipeline,
        n_samples=args.n_samples, prior_rating=prior_rating, prior_weight=prior_weight
    )
    result.print_summary()

    # Plot if requested
    if args.plot or args.save_plot:
        plot_simulation_results(result, save_path=args.save_plot)

    # Batch simulation if requested
    if args.batch > 0:
        print(f"\n\n=== Batch Simulation ({args.batch} games) ===")
        games = df.head(args.batch).to_pandas()
        results = simulate_batch(
            games, complexity_pipeline, rating_pipeline, users_rated_pipeline,
            n_samples=min(args.n_samples, 500), prior_rating=prior_rating, prior_weight=prior_weight
        )

        coverage = compute_coverage(results, level=0.90)
        print("\n90% Interval Coverage:")
        print("-" * 40)
        for metric, stats in coverage.items():
            if stats["coverage"] is not None:
                print(f"  {metric:15s}: {stats['coverage']:.1%} (n={stats['n']}, expected={stats['expected']:.0%})")


if __name__ == "__main__":
    main()
