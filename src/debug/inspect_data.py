"""Inspect the hurdle.parquet file."""

import polars as pl
from pathlib import Path

data_path = Path("data/training/hurdle.parquet")
df = pl.read_parquet(data_path)

print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nNull counts:")
for col in df.columns:
    null_count = df[col].null_count()
    if null_count > 0:
        print(f"  {col}: {null_count}")

print(f"\nDescribe:")
print(df.describe())
