"""Test collection integration with game features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.collection.collection_integration import CollectionIntegration
from src.utils.config import load_config


def test_integration(username: str = "phenrickson"):
    """Test collection integration end-to-end."""

    print("=" * 80)
    print(f"TESTING COLLECTION INTEGRATION FOR USER: {username}")
    print("=" * 80)

    # Initialize
    print("\n[1/5] Initializing integration layer...")
    config = load_config()
    bq_config = config.get_bigquery_config()
    integration = CollectionIntegration(bq_config)
    print("✓ Integration initialized")

    # Get collection with features
    print("\n[2/5] Getting collection with game features...")
    collection_features = integration.get_collection_with_features(username)

    if collection_features is None:
        print("✗ Failed to get collection with features")
        return False

    print(f"✓ Retrieved {len(collection_features)} games with features")
    print(f"  Columns: {len(collection_features.columns)}")
    print(f"  Sample columns: {collection_features.columns[:10]}")

    # Get training data
    print("\n[3/5] Getting collection training data...")
    training_data = integration.get_collection_training_data(
        username, end_train_year=2021, min_ratings=25
    )

    if training_data is None:
        print("✗ Failed to get training data")
        return False

    print(f"✓ Retrieved {len(training_data)} games for training")

    # Get missing games for prediction
    print("\n[4/5] Getting games not in collection...")
    missing_games = integration.get_missing_games_for_prediction(
        username, min_year=2020
    )

    if missing_games is None:
        print("✗ Failed to get missing games")
        return False

    print(f"✓ Found {len(missing_games)} games not in collection (2020+)")

    # Get summary statistics
    print("\n[5/5] Getting collection summary statistics...")
    summary = integration.get_collection_summary_stats(username)

    if summary is None:
        print("✗ Failed to get summary statistics")
        return False

    print("✓ Generated summary statistics:")
    print(f"  Total owned games: {summary['total_owned_games']}")
    print(f"  Avg complexity: {summary['avg_complexity']:.2f}" if summary['avg_complexity'] else "  Avg complexity: N/A")
    print(f"  Avg BGG rating: {summary['avg_bgg_rating']:.2f}" if summary['avg_bgg_rating'] else "  Avg BGG rating: N/A")
    print(f"  Avg year: {summary['avg_year_published']:.0f}" if summary['avg_year_published'] else "  Avg year: N/A")
    print(f"  Avg play time: {summary['avg_playing_time']:.0f} min" if summary['avg_playing_time'] else "  Avg play time: N/A")

    print("\n" + "=" * 80)
    print("✓ INTEGRATION TEST COMPLETE - All steps passed!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test collection integration")
    parser.add_argument(
        "--username", type=str, default="phenrickson", help="BGG username to test"
    )

    args = parser.parse_args()

    success = test_integration(username=args.username)
    sys.exit(0 if success else 1)
