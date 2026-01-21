"""Compare different 2D projection methods."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

MIN_RATINGS = 5

# Find the most recent experiment directory
experiments_base = Path(project_root) / "models/experiments/embeddings"
all_experiments = []
for model_dir in experiments_base.iterdir():
    if model_dir.is_dir():
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir() and (version_dir / "train_embeddings.parquet").exists():
                all_experiments.append(version_dir)

if not all_experiments:
    raise FileNotFoundError(f"No experiment directories found in {experiments_base}")

# Sort by modification time, most recent first
exp_dir = max(all_experiments, key=lambda p: p.stat().st_mtime)
logger.info(f"Using most recent experiment: {exp_dir}")

# Load data (has users_rated) and embeddings separately
train_data = pd.read_parquet(exp_dir / "train_data.parquet")
tune_data = pd.read_parquet(exp_dir / "tune_data.parquet")
test_data = pd.read_parquet(exp_dir / "test_data.parquet")

train_emb = pd.read_parquet(exp_dir / "train_embeddings.parquet")
tune_emb = pd.read_parquet(exp_dir / "tune_embeddings.parquet")
test_emb = pd.read_parquet(exp_dir / "test_embeddings.parquet")

# Merge data with embeddings
train = train_data.merge(train_emb[["game_id", "embedding"]], on="game_id")
tune = tune_data.merge(tune_emb[["game_id", "embedding"]], on="game_id")
test = test_data.merge(test_emb[["game_id", "embedding"]], on="game_id")

# Fit data = train + tune filtered to min_ratings
fit_df = pd.concat([train, tune], ignore_index=True)
fit_df = fit_df[fit_df["users_rated"] >= MIN_RATINGS]

# All data (for transform)
all_df = pd.concat([train, tune, test], ignore_index=True)

logger.info(f"Fit data: {len(fit_df)} games (train + tune with users_rated >= {MIN_RATINGS})")
logger.info(f"All data: {len(all_df)} games")

fit_embeddings = np.array(fit_df["embedding"].tolist())
all_embeddings = np.array(all_df["embedding"].tolist())

output_dir = Path(project_root) / "data"

# 1. PCA (baseline - linear projection)
logger.info("Running PCA...")
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
pca.fit(fit_embeddings)
pca_coords = pca.transform(all_embeddings)
logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# 2. UMAP grid search
from umap import UMAP

n_neighbors_values = [100]
min_dist_values = [0.1, 0.5, 0.8]
metric_values = ['euclidean', 'cosine']

umap_results = {}
total_configs = len(n_neighbors_values) * len(min_dist_values) * len(metric_values)
config_num = 0

for metric in metric_values:
    for n_neighbors in n_neighbors_values:
        for min_dist in min_dist_values:
            config_num += 1
            metric_short = 'euc' if metric == 'euclidean' else 'cos'
            dist_short = str(min_dist).replace('.', '')
            name = f"umap_{metric_short}_n{n_neighbors}_d{dist_short}"
            logger.info(f"[{config_num}/{total_configs}] Running UMAP (n={n_neighbors}, d={min_dist}, {metric})...")

            umap_model = UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=42
            )
            umap_model.fit(fit_embeddings)
            coords = umap_model.transform(all_embeddings)
            umap_results[name] = coords
            logger.info(f"  {name} complete")

# Save all to a single parquet for comparison
logger.info("Saving results...")
results = all_df[["game_id", "name"]].copy()
results["pca_1"] = pca_coords[:, 0]
results["pca_2"] = pca_coords[:, 1]

for name, coords in umap_results.items():
    results[f"{name}_1"] = coords[:, 0]
    results[f"{name}_2"] = coords[:, 1]

results.to_parquet(output_dir / "projection_comparison.parquet", index=False)
logger.info(f"Saved to {output_dir / 'projection_comparison.parquet'}")

# Quick stats
logger.info("Coordinate Ranges:")
logger.info(f"  {'PCA':30s}: x=[{results['pca_1'].min():.1f}, {results['pca_1'].max():.1f}], y=[{results['pca_2'].min():.1f}, {results['pca_2'].max():.1f}]")
for name in umap_results.keys():
    c1, c2 = f"{name}_1", f"{name}_2"
    logger.info(f"  {name:30s}: x=[{results[c1].min():.1f}, {results[c1].max():.1f}], y=[{results[c2].min():.1f}, {results[c2].max():.1f}]")
