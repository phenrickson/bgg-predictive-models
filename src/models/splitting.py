"""Time-based data splitting utilities for model training and evaluation."""

from typing import Dict, Tuple, Optional, Union

import polars as pl


def time_based_split(
    df: pl.DataFrame,
    train_through: int,
    prediction_window: int = 2,
    test_window: Optional[int] = None,
    time_col: str = "year_published",
    return_dict: bool = False,
) -> Union[
    Dict[str, pl.DataFrame],
    Tuple[pl.DataFrame, pl.DataFrame],
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
]:
    """
    Split a dataset into training, validation, and optionally test sets based on time.

    Args:
        df: Full dataset as Polars DataFrame
        train_through: Last year to include in training data (inclusive)
        prediction_window: Number of years for validation window
        test_window: Optional number of years for test window after validation
        time_col: Column name containing the time information
        return_dict: If True, returns a dictionary; if False, returns individual DataFrames

    Returns:
        If return_dict is True:
            Dictionary containing train, validation, and optionally test DataFrames
        If return_dict is False and test_window is None:
            (train_df, val_df)
        If return_dict is False and test_window is provided:
            (train_df, val_df, test_df)
    """
    # Define training period (all data through train_through, inclusive)
    train_df = df.filter(pl.col(time_col) <= train_through)

    # Define validation period (starts after train_through)
    val_start = train_through + 1
    val_df = df.filter(
        (pl.col(time_col) >= val_start)
        & (pl.col(time_col) < val_start + prediction_window)
    )

    # Add test set if requested
    if test_window:
        test_start_year = val_start + prediction_window
        test_df = df.filter(
            (pl.col(time_col) >= test_start_year)
            & (pl.col(time_col) < test_start_year + test_window)
        )

        if return_dict:
            return {"train": train_df, "validation": val_df, "test": test_df}
        else:
            return train_df, val_df, test_df

    if return_dict:
        return {"train": train_df, "validation": val_df}
    else:
        return train_df, val_df


def time_based_cross_validation_splits(
    df: pl.DataFrame,
    start_year: int,
    end_year: int,
    prediction_window: int = 2,
    time_col: str = "year_published",
    return_dict: bool = True,
) -> Union[
    Dict[int, Dict[str, pl.DataFrame]], Dict[int, Tuple[pl.DataFrame, pl.DataFrame]]
]:
    """
    Generate multiple time-based train/validation splits for cross-validation.

    Args:
        df: Full dataset as Polars DataFrame
        start_year: First year to use as train_through
        end_year: Last year to predict
        prediction_window: Number of years for validation window
        time_col: Column name containing the time information
        return_dict: If True, returns nested dictionaries; if False, returns dictionary of tuples

    Returns:
        If return_dict is True:
            Dictionary mapping each train_through year to its corresponding data splits as dictionaries
        If return_dict is False:
            Dictionary mapping each train_through year to (train_df, val_df) tuples
    """
    splits = {}

    # Iterate through time periods
    for train_through in range(start_year, end_year):
        splits[train_through] = time_based_split(
            df=df,
            train_through=train_through,
            prediction_window=prediction_window,
            time_col=time_col,
            return_dict=return_dict,
        )

    return splits
