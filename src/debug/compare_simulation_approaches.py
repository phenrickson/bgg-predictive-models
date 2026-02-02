"""Compare independent vs conditional simulation approaches.

This script demonstrates the difference between:
1. Independent sampling: Each model sampled independently (ignoring dependencies)
2. Conditional sampling: rating/users_rated conditioned on complexity samples

The key question: Does conditioning on simulated complexity values
produce different distributions than independent sampling?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from src.models.outcomes.simulation import (
    sample_posterior,
    sample_conditional_on_complexity,
    compute_geek_rating,
    compute_cholesky,
    precompute_cholesky,
)
from src.models.outcomes.data import load_data
from src.models.outcomes.base import DataConfig


def load_pipeline(model_type: str, experiment_name: str) -> object:
    """Load a trained pipeline from an experiment."""
    base_dir = Path("models/experiments")
    exp_dir = base_dir / model_type / experiment_name

    versions = sorted(exp_dir.glob("v*"), key=lambda x: int(x.name[1:]))
    if not versions:
        raise FileNotFoundError(f"No versions found in {exp_dir}")

    model_path = versions[-1] / "pipeline.pkl"
    if not model_path.exists():
        model_path = versions[-1] / "model.joblib"

    return joblib.load(model_path)


def simulate_independent(
    game: pd.DataFrame,
    complexity_pipeline,
    rating_pipeline,
    users_rated_pipeline,
    n_samples: int = 1000,
    random_state: int = 42,
    cholesky_cache: dict = None,
) -> dict:
    """Simulate each model independently (no conditioning).

    This samples:
    - complexity from complexity model posterior
    - rating from rating model posterior (using POINT complexity as feature)
    - users_rated from users_rated model posterior (using POINT complexity as feature)
    - geek_rating computed from independent rating/users_rated samples
    """
    # Get point complexity prediction
    complexity_point = float(np.clip(complexity_pipeline.predict(game)[0], 1, 5))

    # Add point complexity to features for rating/users_rated
    game_with_complexity = game.copy()
    game_with_complexity["predicted_complexity"] = complexity_point

    # Sample complexity independently
    complexity_samples = sample_posterior(
        complexity_pipeline,
        game,
        n_samples=n_samples,
        include_noise=True,
        random_state=random_state,
        cholesky_L=cholesky_cache.get("complexity") if cholesky_cache else None,
    )
    complexity_samples = np.clip(complexity_samples, 1, 5)

    # Sample rating independently (using POINT complexity, not sampled)
    rating_samples = sample_posterior(
        rating_pipeline,
        game_with_complexity,
        n_samples=n_samples,
        include_noise=True,
        random_state=random_state + 1,
        cholesky_L=cholesky_cache.get("rating") if cholesky_cache else None,
    )
    rating_samples = np.clip(rating_samples, 1, 10)

    # Sample users_rated independently (using POINT complexity, not sampled)
    users_rated_samples = sample_posterior(
        users_rated_pipeline,
        game_with_complexity,
        n_samples=n_samples,
        include_noise=True,
        random_state=random_state + 2,
        cholesky_L=cholesky_cache.get("users_rated") if cholesky_cache else None,
    )

    # Compute geek rating from independent samples
    geek_rating_samples = compute_geek_rating(
        rating_samples,
        users_rated_samples,
        prior_rating=5.5,
        prior_weight=2000,
    )

    return {
        "complexity": complexity_samples.flatten(),
        "rating": rating_samples.flatten(),
        "users_rated": users_rated_samples.flatten(),
        "geek_rating": geek_rating_samples.flatten(),
    }


def simulate_conditional(
    game: pd.DataFrame,
    complexity_pipeline,
    rating_pipeline,
    users_rated_pipeline,
    n_samples: int = 1000,
    random_state: int = 42,
    cholesky_cache: dict = None,
) -> dict:
    """Simulate with proper conditioning.

    This samples:
    - complexity from complexity model posterior
    - rating from rating model posterior, CONDITIONAL on sampled complexity
    - users_rated from users_rated model posterior, CONDITIONAL on sampled complexity
    - geek_rating computed from the conditional rating/users_rated samples
    """
    # Step 1: Sample complexity
    complexity_samples = sample_posterior(
        complexity_pipeline,
        game,
        n_samples=n_samples,
        include_noise=True,
        random_state=random_state,
        cholesky_L=cholesky_cache.get("complexity") if cholesky_cache else None,
    )
    complexity_samples = np.clip(complexity_samples, 1, 5)

    # Step 2: Sample rating CONDITIONAL on complexity samples
    rating_samples = sample_conditional_on_complexity(
        rating_pipeline,
        game,
        complexity_samples,
        sample_posterior_weights=True,
        include_noise=True,
        random_state=random_state + 1,
        cholesky_L=cholesky_cache.get("rating") if cholesky_cache else None,
    )
    rating_samples = np.clip(rating_samples, 1, 10)

    # Step 3: Sample users_rated CONDITIONAL on complexity samples
    users_rated_samples = sample_conditional_on_complexity(
        users_rated_pipeline,
        game,
        complexity_samples,
        sample_posterior_weights=True,
        include_noise=True,
        random_state=random_state + 2,
        cholesky_L=cholesky_cache.get("users_rated") if cholesky_cache else None,
    )

    # Step 4: Compute geek rating from conditional samples
    geek_rating_samples = compute_geek_rating(
        rating_samples,
        users_rated_samples,
        prior_rating=5.5,
        prior_weight=2000,
    )

    return {
        "complexity": complexity_samples.flatten(),
        "rating": rating_samples.flatten(),
        "users_rated": users_rated_samples.flatten(),
        "geek_rating": geek_rating_samples.flatten(),
    }


def print_comparison(name: str, independent: np.ndarray, conditional: np.ndarray):
    """Print comparison statistics."""
    print(f"\n{name}:")
    print(f"  Independent:  mean={independent.mean():.3f}, std={independent.std():.3f}, "
          f"90% CI=[{np.percentile(independent, 5):.3f}, {np.percentile(independent, 95):.3f}]")
    print(f"  Conditional:  mean={conditional.mean():.3f}, std={conditional.std():.3f}, "
          f"90% CI=[{np.percentile(conditional, 5):.3f}, {np.percentile(conditional, 95):.3f}]")
    print(f"  Std ratio (conditional/independent): {conditional.std() / independent.std():.3f}")


def plot_comparison(
    independent: dict,
    conditional: dict,
    game_name: str,
    actuals: dict,
    output_path: Path,
):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    outcomes = ["complexity", "rating", "users_rated", "geek_rating"]

    # Row 1: Histograms comparing distributions
    for i, outcome in enumerate(outcomes):
        ax = axes[0, i]

        ind_samples = independent[outcome]
        cond_samples = conditional[outcome]

        # For users_rated, convert from log scale to count scale
        if outcome == "users_rated":
            ind_samples = np.maximum(np.expm1(ind_samples), 25)
            cond_samples = np.maximum(np.expm1(cond_samples), 25)
            # Use log-spaced bins for users_rated
            all_samples = np.concatenate([ind_samples, cond_samples])
            bins = np.logspace(np.log10(all_samples.min()), np.log10(all_samples.max()), 50)
            ax.set_xscale("log")
        else:
            # Determine bins
            all_samples = np.concatenate([ind_samples, cond_samples])
            bins = np.linspace(all_samples.min(), all_samples.max(), 50)

        ax.hist(ind_samples, bins=bins, alpha=0.5, label="Independent", density=True)
        ax.hist(cond_samples, bins=bins, alpha=0.5, label="Conditional", density=True)

        # Add actual value if available
        if actuals.get(outcome) is not None:
            ax.axvline(actuals[outcome], color="red", linestyle="--", linewidth=2, label="Actual")

        ax.set_xlabel(outcome.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.set_title(f"{outcome.replace('_', ' ').title()}")

    # Row 2: Scatter plots showing correlation structure
    # complexity vs rating
    ax = axes[1, 0]
    ax.scatter(independent["complexity"], independent["rating"], alpha=0.1, s=5, label="Independent")
    ax.scatter(conditional["complexity"], conditional["rating"], alpha=0.1, s=5, label="Conditional")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Rating")
    ax.set_title("Complexity vs Rating")

    # complexity vs users_rated
    ax = axes[1, 1]
    users_rated_ind = np.maximum(np.expm1(independent["users_rated"]), 25)
    users_rated_cond = np.maximum(np.expm1(conditional["users_rated"]), 25)
    ax.scatter(independent["complexity"], users_rated_ind, alpha=0.1, s=5, label="Independent")
    ax.scatter(conditional["complexity"], users_rated_cond, alpha=0.1, s=5, label="Conditional")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Users Rated")
    ax.set_title("Complexity vs Users Rated")
    ax.set_yscale("log")

    # rating vs geek_rating
    ax = axes[1, 2]
    ax.scatter(independent["rating"], independent["geek_rating"], alpha=0.1, s=5, label="Independent")
    ax.scatter(conditional["rating"], conditional["geek_rating"], alpha=0.1, s=5, label="Conditional")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Geek Rating")
    ax.set_title("Rating vs Geek Rating")

    # users_rated vs geek_rating
    ax = axes[1, 3]
    ax.scatter(users_rated_ind, independent["geek_rating"], alpha=0.1, s=5, label="Independent")
    ax.scatter(users_rated_cond, conditional["geek_rating"], alpha=0.1, s=5, label="Conditional")
    ax.set_xlabel("Users Rated")
    ax.set_ylabel("Geek Rating")
    ax.set_title("Users Rated vs Geek Rating")
    ax.set_xscale("log")

    plt.suptitle(f"Independent vs Conditional Simulation: {game_name[:50]}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {output_path}")


def main():
    print("=" * 70)
    print("Comparing Independent vs Conditional Simulation")
    print("=" * 70)

    # Load models
    print("\nLoading models...")
    complexity_pipeline = load_pipeline("complexity", "eval-complexity-2023")
    rating_pipeline = load_pipeline("rating", "eval-rating-2023")
    users_rated_pipeline = load_pipeline("users_rated", "eval-users_rated-2023")
    print("  Loaded complexity, rating, users_rated pipelines")

    # Pre-compute Cholesky
    cholesky_cache = precompute_cholesky(
        complexity_pipeline,
        rating_pipeline,
        users_rated_pipeline,
    )
    print("  Pre-computed Cholesky decompositions")

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

    # Filter to games with valid outcomes
    valid_mask = (
        ~df_pandas["rating"].isna()
        & ~df_pandas["users_rated"].isna()
        & (df_pandas["users_rated"] > 0)
        & ~df_pandas["complexity"].isna()
    )
    df_valid = df_pandas[valid_mask].reset_index(drop=True)
    print(f"  Valid games with all outcomes: {len(df_valid)}")

    # Pick a few example games
    n_samples = 2000
    random_state = 42

    # Select games with different characteristics
    # High complexity game
    high_complexity_idx = df_valid["complexity"].idxmax()
    # Popular game (high users_rated)
    popular_idx = df_valid["users_rated"].idxmax()
    # Random game
    np.random.seed(42)
    random_idx = np.random.choice(len(df_valid))

    example_indices = [
        ("High Complexity", high_complexity_idx),
        ("Most Popular", popular_idx),
        ("Random Game", random_idx),
    ]

    output_dir = Path("tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, idx in example_indices:
        game = df_valid.iloc[[idx]].reset_index(drop=True)
        game_name = game["name"].iloc[0]

        print(f"\n{'='*70}")
        print(f"{label}: {game_name}")
        print(f"{'='*70}")

        # Get actual values
        actuals = {
            "complexity": float(game["complexity"].iloc[0]),
            "rating": float(game["rating"].iloc[0]),
            "users_rated": float(game["users_rated"].iloc[0]),
        }
        actuals["geek_rating"] = (
            (actuals["rating"] * actuals["users_rated"]) + (5.5 * 2000)
        ) / (actuals["users_rated"] + 2000)

        print(f"\nActual values:")
        print(f"  Complexity: {actuals['complexity']:.2f}")
        print(f"  Rating: {actuals['rating']:.2f}")
        print(f"  Users Rated: {actuals['users_rated']:.0f}")
        print(f"  Geek Rating: {actuals['geek_rating']:.2f}")

        # Run both simulations
        print(f"\nRunning simulations ({n_samples} samples each)...")

        independent = simulate_independent(
            game,
            complexity_pipeline,
            rating_pipeline,
            users_rated_pipeline,
            n_samples=n_samples,
            random_state=random_state,
            cholesky_cache=cholesky_cache,
        )

        conditional = simulate_conditional(
            game,
            complexity_pipeline,
            rating_pipeline,
            users_rated_pipeline,
            n_samples=n_samples,
            random_state=random_state,
            cholesky_cache=cholesky_cache,
        )

        # Print comparison
        print_comparison("Complexity", independent["complexity"], conditional["complexity"])
        print_comparison("Rating", independent["rating"], conditional["rating"])

        # Convert users_rated to count scale for comparison
        users_rated_ind = np.maximum(np.expm1(independent["users_rated"]), 25)
        users_rated_cond = np.maximum(np.expm1(conditional["users_rated"]), 25)
        print_comparison("Users Rated (count)", users_rated_ind, users_rated_cond)
        print_comparison("Geek Rating", independent["geek_rating"], conditional["geek_rating"])

        # Correlation analysis
        print("\nCorrelation between complexity and rating:")
        print(f"  Independent: {np.corrcoef(independent['complexity'], independent['rating'])[0,1]:.3f}")
        print(f"  Conditional: {np.corrcoef(conditional['complexity'], conditional['rating'])[0,1]:.3f}")

        print("\nCorrelation between complexity and users_rated:")
        print(f"  Independent: {np.corrcoef(independent['complexity'], independent['users_rated'])[0,1]:.3f}")
        print(f"  Conditional: {np.corrcoef(conditional['complexity'], conditional['users_rated'])[0,1]:.3f}")

        # Create plot
        safe_name = "".join(c if c.isalnum() else "_" for c in game_name[:30])
        plot_path = output_dir / f"comparison_{safe_name}.png"
        plot_comparison(independent, conditional, game_name, actuals, plot_path)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
Key differences to look for:

1. STANDARD DEVIATION:
   - If conditional std > independent std: uncertainty is being propagated
   - If they're equal: complexity uncertainty isn't affecting downstream

2. CORRELATIONS:
   - Independent: complexity-rating correlation should be ~0 (no dependency)
   - Conditional: complexity-rating correlation reflects the true model relationship

3. DISTRIBUTION SHAPE:
   - Conditional distributions may be wider or have different shapes
   - This reflects the uncertainty propagation through the model chain
""")


if __name__ == "__main__":
    main()
