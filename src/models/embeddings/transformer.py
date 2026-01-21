"""Embedding-specific transformer for feature preprocessing.

This module provides a transformer optimized for embedding model training,
with different default settings than the base transformer used for
predictive models.
"""

from typing import List, Optional

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.transformers import (
    BaseBGGTransformer,
    LogTransformer,
    YearTransformer,
)


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
        family_allow_patterns: Optional[List[str]] = None,
        max_family_features: int = 150,
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
            family_allow_patterns=family_allow_patterns,
            max_family_features=max_family_features,
            **kwargs,
        )


def create_embedding_preprocessor(
    model_type: str = "linear",
    reference_year: int = 2000,
    normalization_factor: int = 25,
    log_columns: Optional[List[str]] = None,
    preserve_columns: Optional[List[str]] = None,
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
        preserve_columns = ["year_published", "predicted_complexity"]

    # Create embedding transformer with preserved columns
    transformer = EmbeddingTransformer(
        preserve_columns=preserve_columns,
        **kwargs,
    )

    # Build pipeline steps (same structure as create_bgg_preprocessor)
    pipeline_steps = [
        ("bgg_preprocessor", transformer),
        (
            "impute",
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=False
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
                ("scaler", StandardScaler()),
            ]
        )
    elif model_type == "tree":
        pipeline_steps.extend([("variance_selector", VarianceThreshold(threshold=0))])

    pipeline = Pipeline(pipeline_steps)
    pipeline.set_output(transform="pandas")

    return pipeline
