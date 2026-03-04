#!/usr/bin/env python
"""Explain a game's predicted complexity using LinearExplainer."""

import argparse

from src.models.explain import load_explainer


def main():
    parser = argparse.ArgumentParser(description="Explain a game's prediction")
    parser.add_argument("game_id", type=int, help="BGG game ID to explain")
    parser.add_argument(
        "--experiment", default="eval-complexity-2023", help="Experiment name"
    )
    parser.add_argument(
        "--model-type", default="complexity", help="Model type (complexity, rating, etc.)"
    )
    parser.add_argument("--top-n", type=int, default=15, help="Number of top features")
    args = parser.parse_args()

    # Load explainer
    explainer = load_explainer(args.experiment, args.model_type)

    # Explain the game
    result = explainer.explain_game(args.game_id, top_n=args.top_n)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"Game: {result['game_name']} (ID: {result['game_id']})")
    print(f"{'=' * 60}")

    if result["actual"] is not None:
        print(f"Actual:    {result['actual']:.3f}")
    print(f"Predicted: {result['prediction']:.3f}")
    print(f"Intercept: {result['intercept']:.3f}")

    print(f"\nTop {args.top_n} Feature Contributions:")
    print("-" * 60)

    for _, row in result["contributions"].iterrows():
        sign = "+" if row["contribution"] >= 0 else ""
        print(f"  {row['feature']:45} {sign}{row['contribution']:.4f}")

    print("-" * 60)


if __name__ == "__main__":
    main()
