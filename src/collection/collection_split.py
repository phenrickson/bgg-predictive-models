"""Train/val/test splitting for user collection models.

The splitter takes a single labeled feature frame — produced by joining
the game universe (features) with the user's collection (status) and
applying an outcome to derive the ``label`` column — and divides it into
train/val/test.

Dispatches on :class:`OutcomeDefinition.task`:

- classification (`stratified_random` or `time_based`): same frame,
  same logic for both classes; the label distribution is preserved by
  splitting positives and negatives independently in `stratified_random`.
- regression: shuffle-and-slice on whatever rows the caller passed in
  (typically already filtered to qualifying rows by the outcome's
  ``require``).

Downsampling the negative class for training-time class balance is NOT
this module's concern. The split preserves the real class distribution;
class balancing is a training-time concern (see :func:`downsample_negatives`).
"""

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, Literal, Optional, Tuple

import polars as pl

from src.collection.outcomes import OutcomeDefinition
from src.models.splitting import time_based_split

logger = logging.getLogger(__name__)


@dataclass
class ClassificationSplitConfig:
    """Config for classification splitting.

    ``split_mode`` selects how the labeled frame is bucketed:

    - ``stratified_random``: random split per label class (preserves the
      positive:negative ratio in each of train/val/test).
    - ``time_based``: year cutoff via :func:`src.models.splitting.time_based_split`.
    """

    split_mode: Literal["stratified_random", "time_based"] = "stratified_random"

    # stratified_random params
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # time_based params (train_through required when split_mode == "time_based")
    train_through: Optional[int] = None
    """Last year (inclusive) to include in training data. Required for time_based."""
    prediction_window: int = 2
    """Validation window in years."""
    test_window: Optional[int] = None
    """Test window in years; None means train/val only (2 splits)."""
    time_column: str = "year_published"

    random_seed: int = 42


@dataclass
class RegressionSplitConfig:
    """Config for regression splitting."""

    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


def _coerce(cls, data: Optional[Dict[str, Any]]):
    """Build ``cls`` from a YAML-loaded dict. Unknown keys are ignored so
    the config can carry stratified-random fallback ratios on a time-based
    config (and vice versa) without erroring."""
    if not data:
        return cls()
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_split_configs(
    config: Dict[str, Any],
) -> Tuple[ClassificationSplitConfig, RegressionSplitConfig]:
    """Read ``collections.split`` from a YAML config and return a
    ``(ClassificationSplitConfig, RegressionSplitConfig)`` pair.

    Falls back to dataclass defaults when sections are missing.
    """
    section = (config.get("collections") or {}).get("split") or {}
    return (
        _coerce(ClassificationSplitConfig, section.get("classification")),
        _coerce(RegressionSplitConfig, section.get("regression")),
    )


SplitTriple = Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]


def _shuffle_split(df: pl.DataFrame, val_ratio: float, test_ratio: float, seed: int) -> SplitTriple:
    """Shuffle ``df`` and slice into (train, val, test)."""
    if df.height == 0:
        empty = df.head(0)
        return empty, empty, empty
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n = shuffled.height
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_df = shuffled[:n_test]
    val_df = shuffled[n_test : n_test + n_val]
    train_df = shuffled[n_test + n_val :]
    return train_df, val_df, test_df


class _ClassificationSplitter:
    """Split a labeled classification frame.

    Input: a single ``labeled_df`` with ``game_id``, ``label`` (bool),
    ``year_published`` (for time_based), and feature columns.
    """

    def __init__(self, config: ClassificationSplitConfig):
        self.config = config

    def split(self, labeled_df: pl.DataFrame) -> SplitTriple:
        if labeled_df.height == 0:
            raise ValueError("No rows in classification split input")

        if self.config.split_mode == "stratified_random":
            return self._stratified_random_split(labeled_df)
        if self.config.split_mode == "time_based":
            return self._time_based_split(labeled_df)
        raise ValueError(f"Unknown split_mode: {self.config.split_mode!r}")

    def _stratified_random_split(self, df: pl.DataFrame) -> SplitTriple:
        """Random split per class so the positive:negative ratio is preserved
        in each of train/val/test."""
        cfg = self.config
        positives = df.filter(pl.col("label").cast(pl.Boolean) == True)
        negatives = df.filter(pl.col("label").cast(pl.Boolean) == False)

        train_pos, val_pos, test_pos = _shuffle_split(
            positives, cfg.validation_ratio, cfg.test_ratio, cfg.random_seed
        )
        train_neg, val_neg, test_neg = _shuffle_split(
            negatives, cfg.validation_ratio, cfg.test_ratio, cfg.random_seed
        )
        return (
            pl.concat([train_pos, train_neg], how="vertical_relaxed"),
            pl.concat([val_pos, val_neg], how="vertical_relaxed"),
            pl.concat([test_pos, test_neg], how="vertical_relaxed"),
        )

    def _time_based_split(self, df: pl.DataFrame) -> SplitTriple:
        cfg = self.config
        if cfg.train_through is None:
            raise ValueError("time_based split requires config.train_through")
        if cfg.time_column not in df.columns:
            raise ValueError(
                f"time_based split requires column {cfg.time_column!r}"
            )
        result = time_based_split(
            df,
            train_through=cfg.train_through,
            prediction_window=cfg.prediction_window,
            test_window=cfg.test_window,
            time_col=cfg.time_column,
            return_dict=False,
        )
        if cfg.test_window is None:
            train_df, val_df = result
            test_df = df.head(0)  # empty, same schema
        else:
            train_df, val_df, test_df = result
        return train_df, val_df, test_df


class _RegressionSplitter:
    """Split a labeled regression frame.

    Input: a single ``labeled_df`` with ``label`` (numeric) and feature
    columns. Caller is responsible for filtering to qualifying rows
    (e.g. via the outcome's ``require``).
    """

    def __init__(self, config: RegressionSplitConfig):
        self.config = config

    def split(self, labeled_df: pl.DataFrame) -> SplitTriple:
        if labeled_df.height == 0:
            raise ValueError("No rows in regression split input")
        return _shuffle_split(
            labeled_df,
            self.config.validation_ratio,
            self.config.test_ratio,
            self.config.random_seed,
        )


class CollectionSplitter:
    """Public dispatcher: picks classification or regression strategy from outcome.task.

    Takes a single ``labeled_df`` (universe-joined-to-collection, with the
    outcome's label applied) and returns a (train, val, test) triple.
    """

    def __init__(
        self,
        classification_config: Optional[ClassificationSplitConfig] = None,
        regression_config: Optional[RegressionSplitConfig] = None,
    ):
        self._classification = _ClassificationSplitter(
            classification_config or ClassificationSplitConfig()
        )
        self._regression = _RegressionSplitter(
            regression_config or RegressionSplitConfig()
        )

    def split(self, labeled_df: pl.DataFrame, outcome: OutcomeDefinition) -> SplitTriple:
        if outcome.task == "classification":
            return self._classification.split(labeled_df)
        if outcome.task == "regression":
            return self._regression.split(labeled_df)
        raise ValueError(f"Unsupported outcome task: {outcome.task!r}")


def downsample_negatives(
    df: pl.DataFrame,
    ratio: float,
    protect_min_ratings: int = 25,
    rating_column: str = "users_rated",
    random_seed: int = 1999,
) -> pl.DataFrame:
    """Downsample the negatives (label=False rows) to a target ratio of
    negatives-per-positive. Positives are preserved.

    Negatives with ``rating_column >= protect_min_ratings`` are always kept;
    the low-rating tail (``< protect_min_ratings``) is sampled to fill the
    remaining budget up to ``n_positives * ratio`` total negatives. If the
    protected pool alone already exceeds the target, no low-rating negatives
    are added.

    Set ``protect_min_ratings=0`` to recover the old behavior (uniform
    sampling over all negatives).

    Use this at training time on the training DataFrame only — never on
    validation or test. The split itself should preserve the real class
    distribution; downsampling is a training-time class-balance concern.

    Args:
        df: DataFrame with a boolean ``label`` column.
        ratio: Target negatives-per-positive (e.g., 5.0 → 5 negatives per positive).
        protect_min_ratings: Keep every negative with ``rating_column`` at or
            above this value; only sample the rows below it. Default 25.
        rating_column: Column used for the protection threshold. Default
            ``users_rated``.
        random_seed: Seed for the negative sampling RNG.

    Returns:
        DataFrame with all positives, all protected high-rating negatives,
        and a sampled subset of low-rating negatives.
    """
    if "label" not in df.columns:
        raise ValueError("downsample_negatives requires a 'label' column")
    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio!r}")
    if protect_min_ratings < 0:
        raise ValueError(
            f"protect_min_ratings must be >= 0, got {protect_min_ratings!r}"
        )
    if protect_min_ratings > 0 and rating_column not in df.columns:
        raise ValueError(
            f"rating_column {rating_column!r} missing from df; "
            f"available columns start with: {df.columns[:10]}"
        )

    positives = df.filter(pl.col("label").cast(pl.Boolean) == True)
    negatives = df.filter(pl.col("label").cast(pl.Boolean) == False)

    target = int(positives.height * ratio)
    if target >= negatives.height:
        return df

    if protect_min_ratings == 0:
        sampled = negatives.sample(n=target, shuffle=True, seed=random_seed)
        return pl.concat([positives, sampled], how="vertical_relaxed")

    protected = negatives.filter(pl.col(rating_column) >= protect_min_ratings)
    tail = negatives.filter(pl.col(rating_column) < protect_min_ratings)

    remaining = target - protected.height
    if remaining <= 0:
        return pl.concat([positives, protected], how="vertical_relaxed")
    if remaining >= tail.height:
        return pl.concat([positives, protected, tail], how="vertical_relaxed")

    sampled_tail = tail.sample(n=remaining, shuffle=True, seed=random_seed)
    return pl.concat(
        [positives, protected, sampled_tail], how="vertical_relaxed"
    )
