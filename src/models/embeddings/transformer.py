"""Embedding-specific transformer for feature preprocessing.

This module provides a transformer optimized for embedding model training,
with different default settings than the base transformer used for
predictive models.
"""

from typing import List, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.features.transformers import (
    BaseBGGTransformer,
    LogTransformer,
    MinCountSelector,
    TwoSDScaler,
    YearTransformer,
)


class PrefixColumnDropper(BaseEstimator, TransformerMixin):
    """Drop DataFrame columns whose names start with any of the given prefixes.

    Used as a post-preprocessor step to exclude columns (e.g., year_published*)
    that should be carried through preprocessing for context but excluded from
    the feature matrix passed to the embedding algorithm.
    """

    def __init__(self, prefixes: Optional[List[str]] = None):
        self.prefixes = prefixes or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return X
        drop_cols = [
            c for c in X.columns
            if any(c.startswith(p) for p in self.prefixes)
        ]
        if drop_cols:
            return X.drop(columns=drop_cols)
        return X


# Default family patterns for embeddings - focus on game characteristic types
DEFAULT_EMBEDDING_FAMILY_PATTERNS = [
    "^Players:",
    "^Category",
    "^Sports",
    "^Traditional",
    "^Card",
    "^Collectible",
]


class EmbeddingTransformer(BaseBGGTransformer):
    """Transformer optimized for embedding model training.

    This transformer extends BaseBGGTransformer with defaults suited for
    learning game embeddings:
    - Excludes designer, artist, and publisher features to focus on
      game characteristics rather than creator metadata
    - Restricts family features to game characteristic types only

    The transformer is kept separate from BaseBGGTransformer to avoid
    breaking pickle compatibility with deployed predictive models.
    """

    def __init__(
        self,
        # Embedding-specific defaults (different from base transformer)
        create_designer_features: bool = False,
        create_artist_features: bool = False,
        create_publisher_features: bool = False,
        include_count_features: bool = False,
        family_allow_patterns: Optional[List[str]] = None,
        max_family_features: int = 150,
        # Player count enters as continuous min_players / max_players (via
        # preserve_columns), not a 10-column thermometer one-hot: the one-hot
        # block is correlated + common, so unscaled it dominates a PCA
        # component and quantises the space into bands.
        create_player_dummies: bool = False,
        # Inherit other defaults from base
        **kwargs,
    ):
        """Initialize embedding transformer with embedding-optimized defaults.

        Args:
            create_designer_features: Whether to create designer features.
                Default False for embeddings.
            create_artist_features: Whether to create artist features.
                Default False for embeddings.
            create_publisher_features: Whether to create publisher features.
                Default False for embeddings.
            family_allow_patterns: Regex patterns to filter family features.
                Defaults to game characteristic patterns.
            max_family_features: Maximum number of family features.
                Default 150 for embeddings.
            **kwargs: Additional arguments passed to BaseBGGTransformer.
        """
        if family_allow_patterns is None:
            family_allow_patterns = DEFAULT_EMBEDDING_FAMILY_PATTERNS

        super().__init__(
            create_designer_features=create_designer_features,
            create_artist_features=create_artist_features,
            create_publisher_features=create_publisher_features,
            include_count_features=include_count_features,
            family_allow_patterns=family_allow_patterns,
            max_family_features=max_family_features,
            create_player_dummies=create_player_dummies,
            **kwargs,
        )


def create_embedding_preprocessor(
    model_type: str = "linear",
    reference_year: int = 2000,
    normalization_factor: int = 25,
    log_columns: Optional[List[str]] = None,
    preserve_columns: Optional[List[str]] = None,
    include_description_embeddings: bool = False,
    min_feature_count: int = 10,
    **kwargs,
) -> Pipeline:
    """Create a preprocessing pipeline optimized for embedding training.

    This factory function creates a Pipeline with EmbeddingTransformer,
    using defaults suited for learning game embeddings. Mirrors the structure
    of create_bgg_preprocessor but with embedding-specific defaults.

    Args:
        model_type: Type of model ('linear' or 'tree'). Affects scaling.
        reference_year: Reference year for year normalization.
        normalization_factor: Factor for year normalization.
        log_columns: Columns to apply log transformation.
        preserve_columns: Columns to preserve through transformation.
        min_feature_count: Drop binary indicator features (mechanic/category/
            family dummies) carried by fewer than this many games before the
            decomposition. Linear model_type only.
        **kwargs: Additional arguments passed to EmbeddingTransformer.

    Returns:
        sklearn Pipeline with EmbeddingTransformer and preprocessing steps.
    """
    if model_type not in ["linear", "tree"]:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Choose 'linear' or 'tree'."
        )

    if log_columns is None:
        log_columns = [
            "min_age",
            "min_playtime",
            "max_playtime",
            "time_per_player",
            "description_word_count",
        ]

    if preserve_columns is None:
        # min/max_players carried through as continuous features (see
        # EmbeddingTransformer — no player_count_* one-hots).
        preserve_columns = [
            "year_published",
            "predicted_complexity",
            "min_players",
            "max_players",
        ]

    # Create embedding transformer with preserved columns
    transformer = EmbeddingTransformer(
        preserve_columns=preserve_columns,
        include_description_embeddings=include_description_embeddings,
        **kwargs,
    )

    # Build pipeline steps (same structure as create_bgg_preprocessor)
    pipeline_steps = [
        ("bgg_preprocessor", transformer),
        (
            # No add_indicator: a "this value was missing" flag encodes data
            # completeness, not how a game plays, and (being common) it lands
            # a PCA component of its own.
            "impute",
            SimpleImputer(
                strategy="median", add_indicator=False, keep_empty_features=False
            ),
        ),
    ]

    # Add additional steps for linear models
    if model_type == "linear":
        pipeline_steps.extend(
            [
                ("log", LogTransformer(columns=log_columns)),
                (
                    "year",
                    YearTransformer(
                        reference_year=reference_year,
                        normalization_factor=normalization_factor,
                    ),
                ),
                ("variance_selector", VarianceThreshold(threshold=0)),
                ("min_count", MinCountSelector(min_count=min_feature_count)),
                # Gelman-style: continuous / 2*SD, dummies left at 0/1 so a rare
                # feature keeps its natural variance p(1-p) and PCA never spends
                # a component on it. Pairs with algorithm="pca" (centres) —
                # a blanket StandardScaler was what forced every dummy to
                # variance 1.
                ("scaler", TwoSDScaler()),
            ]
        )
    elif model_type == "tree":
        pipeline_steps.extend([("variance_selector", VarianceThreshold(threshold=0))])

    pipeline = Pipeline(pipeline_steps)
    pipeline.set_output(transform="pandas")

    return pipeline
