"""Debug collection data to see what's being loaded."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.collection.collection_loader import BGGCollectionLoader
import polars as pl


def debug_collection(username: str = "phenrickson"):
    """Debug collection data loading."""

    loader = BGGCollectionLoader(username=username)
    df = loader.get_collection()

    if df is None:
        print("Failed to load collection")
        return

    print(f"Loaded {len(df)} items\n")

    # Show first few rows
    print("First 3 items:")
    print(df.head(3))

    # Check for null values in key columns
    print("\nNull value counts:")
    for col in ["user_rating", "average_rating", "bayes_average", "bgg_rank"]:
        if col in df.columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            total = len(df)
            print(f"  {col}: {null_count}/{total} ({100*null_count/total:.1f}% null)")

    # Show a sample game with data
    print("\nSample of non-null ratings (if any):")
    rated = df.filter(pl.col("user_rating").is_not_null())
    if len(rated) > 0:
        print(rated.head(5).select(["game_name", "user_rating", "average_rating", "bgg_rank"]))
    else:
        print("  No user ratings found")

    # Show games with BGG stats
    print("\nSample of games with BGG stats (if any):")
    with_stats = df.filter(pl.col("average_rating").is_not_null())
    if len(with_stats) > 0:
        print(with_stats.head(5).select(["game_name", "average_rating", "bayes_average", "users_rated", "bgg_rank"]))
    else:
        print("  No BGG stats found")

    # Check what data we DO have
    print("\nColumns with non-null values:")
    for col in df.columns:
        null_count = df.select(pl.col(col).is_null().sum()).item()
        if null_count < len(df):
            non_null = len(df) - null_count
            print(f"  {col}: {non_null} non-null values")


if __name__ == "__main__":
    debug_collection()
