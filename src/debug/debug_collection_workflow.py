"""Verify collection loader and processor work end-to-end."""

import sys
import os
from pathlib import Path

str(Path.cwd())

# Add src to path
try:
    sys.path.insert(0, str(Path(__file__).parent))
except Exception as e:
    sys.path.insert(0, str(Path.cwd()))

os.getcwd()

# load modules
from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_processor import CollectionProcessor
from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_processor import CollectionProcessor
from src.collection.collection_storage import CollectionStorage
from src.utils.config import load_config

# set username
username = "phenrickson"

# Step 1: Initialize loader
print("\n[1/5] Initializing collection loader...")
try:
    loader = BGGCollectionLoader(username=username)
    print(f"✓ Loader initialized for user '{username}'")
except Exception as e:
    print(f"✗ Failed to initialize loader: {e}")

# Step 2: Verify collection exists
print("\n[2/5] Verifying collection exists...")
try:
    exists = loader.verify_collection_exists()
    if exists:
        print(f"✓ Collection exists for user '{username}'")
    else:
        print(f"✗ Collection not found for user '{username}'")
except Exception as e:
    print(f"✗ Failed to verify collection: {e}")

# Step 3: Load collection
print("\n[3/5] Loading collection data...")
try:
    collection_df = loader.get_collection()
    if collection_df is None:
        print("✗ Failed to load collection (returned None)")
    print(f"✓ Loaded collection with {len(collection_df)} items")
    print(f"  Columns: {collection_df.columns}")
except Exception as e:
    print(f"✗ Failed to load collection: {e}")
    import traceback

    traceback.print_exc()

collection_df


# Step 4: Process collection
print("\n[4/5] Processing collection data...")
try:
    processor = CollectionProcessor(collection_df)

    # Get games only
    games = processor.get_games(owned_only=True)
    expansions = processor.get_expansions(owned_only=True)

    print(f"✓ Processor initialized successfully")
    print(f"  Owned games: {len(games)}")
    print(f"  Owned expansions: {len(expansions)}")

except Exception as e:
    print(f"✗ Failed to process collection: {e}")
# Step 5: Generate summary
print("\n[5/5] Generating collection summary...")
try:
    summary = processor.get_summary()
    print(f"✓ Summary generated successfully")
    print("\nCollection Summary:")
    print(f"  Total items: {summary['total_items']}")
    print(f"  Owned games: {summary['owned_games']}")
    print(f"  Unique games: {summary['unique_games']}")
    print(f"  Expansions: {summary['expansions']}")
    print(
        f"  Avg user rating: {summary['avg_user_rating']:.2f}"
        if summary["avg_user_rating"]
        else "  Avg user rating: N/A"
    )
    print(
        f"  Avg BGG rating: {summary['avg_bgg_rating']:.2f}"
        if summary["avg_bgg_rating"]
        else "  Avg BGG rating: N/A"
    )

    if summary["top_rated_game"]:
        print(
            f"  Top rated: {summary['top_rated_game']['name']} ({summary['top_rated_game']['rating']})"
        )

    # Show top rated games
    print("\nTop 5 Rated Games:")
    top_rated = processor.get_top_rated(n=5)
    print(top_rated)

    # Show ratings distribution
    print("\nRatings Distribution:")
    distribution = processor.get_ratings_distribution()
    print(distribution)

except Exception as e:
    print(f"✗ Failed to generate summary: {e}")
    import traceback

    traceback.print_exc()
    return False

print("\n" + "=" * 80)
print("✓ VERIFICATION COMPLETE - All steps passed!")
print("=" * 80)

print("Step 3: Storing collection in BigQuery...")
storage = CollectionStorage(environment="dev")
storage.save_collection(username, collection_df)
print("✓ Collection saved to BigQuery\n")

return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify BGG collection pipeline")
    parser.add_argument(
        "--username",
        type=str,
        default="phenrickson",
        help="BGG username to test (default: phenrickson)",
    )

    args = parser.parse_args()

    success = verify_collection_pipeline(username=args.username)
    sys.exit(0 if success else 1)
