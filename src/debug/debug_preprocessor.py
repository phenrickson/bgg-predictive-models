"""Debug the preprocessor to find where NaNs are introduced."""

import polars as pl
import pandas as pd
from pathlib import Path

from src.models.outcomes.train import get_model_class
from src.models.outcomes.data import select_X_y
from src.features.preprocessor import create_bgg_preprocessor

data_path = Path("data/training/hurdle.parquet")
df = pl.read_parquet(data_path)
print(f"Loaded {len(df)} rows")

model_class = get_model_class("hurdle")
model = model_class()

X, y = select_X_y(df, model.target_column)
print(f"X shape: {X.shape}")
print(f"X dtypes:\n{X.dtypes.value_counts()}")
print(f"X NaN count: {X.isna().sum().sum()}")

# Create preprocessor
preprocessor = create_bgg_preprocessor(
    model_type="linear",
    preserve_columns=["year_published"],
    include_description_embeddings=False,
)

# Step through each transformer
print("\n--- Stepping through pipeline ---")
for name, transformer in preprocessor.named_steps.items():
    print(f"\nStep: {name}")
    print(f"  Transformer: {type(transformer).__name__}")

    # Fit and transform
    if name == list(preprocessor.named_steps.keys())[0]:
        X_out = transformer.fit_transform(X)
    else:
        X_out = transformer.fit_transform(X_out)

    # Check output
    if hasattr(X_out, 'shape'):
        print(f"  Output shape: {X_out.shape}")

    if isinstance(X_out, pd.DataFrame):
        nan_count = X_out.isna().sum().sum()
        print(f"  NaN count: {nan_count}")
        if nan_count > 0:
            nan_cols = X_out.isna().sum()
            nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)
            print(f"  Top NaN columns:\n{nan_cols.head(10)}")
    elif hasattr(X_out, '__array__'):
        import numpy as np
        nan_count = np.isnan(X_out).sum()
        print(f"  NaN count: {nan_count}")

    X = X_out
