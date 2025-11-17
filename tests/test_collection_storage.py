"""Test collection storage functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.collection.collection_loader import BGGCollectionLoader
from src.collection.collection_storage import CollectionStorage


def test_storage(username: str = "phenrickson"):
    """Test collection storage end-to-end."""

    print("=" * 80)
    print(f"TESTING COLLECTION STORAGE FOR USER: {username}")
    print("=" * 80)

    # Step 1: Load collection
    print("\n[1/5] Loading collection from BGG...")
    loader = BGGCollectionLoader(username=username)
    collection_df = loader.get_collection()

    if collection_df is None:
        print("✗ Failed to load collection")
        return False

    print(f"✓ Loaded {len(collection_df)} items")

    # Step 2: Initialize storage
    print("\n[2/5] Initializing collection storage...")
    storage = CollectionStorage(environment="dev")
    print("✓ Storage initialized")

    # Step 3: Save collection
    print("\n[3/5] Saving collection to BigQuery...")
    success = storage.save_collection(username, collection_df)

    if not success:
        print("✗ Failed to save collection")
        return False

    print("✓ Collection saved successfully")

    # Step 4: Retrieve collection
    print("\n[4/5] Retrieving collection from BigQuery...")
    retrieved_df = storage.get_latest_collection(username)

    if retrieved_df is None:
        print("✗ Failed to retrieve collection")
        return False

    print(f"✓ Retrieved {len(retrieved_df)} items")

    # Step 5: Get owned game IDs
    print("\n[5/5] Getting owned game IDs...")
    game_ids = storage.get_owned_game_ids(username)

    if game_ids is None:
        print("✗ Failed to get game IDs")
        return False

    print(f"✓ Found {len(game_ids)} owned games")
    print(f"  Sample game IDs: {game_ids[:5]}")

    # Show collection history
    print("\nCollection History:")
    history = storage.get_collection_history(username)
    if history is not None:
        print(history)

    print("\n" + "=" * 80)
    print("✓ STORAGE TEST COMPLETE - All steps passed!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test collection storage")
    parser.add_argument(
        "--username", type=str, default="phenrickson", help="BGG username to test"
    )

    args = parser.parse_args()

    success = test_storage(username=args.username)
    sys.exit(0 if success else 1)
