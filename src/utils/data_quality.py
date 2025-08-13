"""Utilities for data quality assessment."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import polars as pl


def view_missingness(
    df: pl.DataFrame,
    include_list_columns: bool = True,
    list_empty_as_missing: bool = True,
    sort_by: str = "missing_percent",
    threshold: Optional[float] = None,
) -> pl.DataFrame:
    """
    Analyze missingness in a Polars DataFrame.

    Args:
        df: Polars DataFrame to analyze
        include_list_columns: Whether to check for empty lists in list columns
        list_empty_as_missing: Whether to count empty lists as missing values
        sort_by: Column to sort results by ("column", "missing_count", or "missing_percent")
        threshold: Optional threshold to filter results (e.g., 0.05 for columns with >5% missing)

    Returns:
        DataFrame with missingness statistics for each column
    """
    # Get total row count
    total_rows = df.height

    # Initialize results
    results = []

    # Analyze each column
    for col_name in df.columns:
        col = df[col_name]
        col_dtype = str(col.dtype)

        # Check if it's a list column
        is_list_column = "List" in col_dtype

        # Skip list columns if specified
        if is_list_column and not include_list_columns:
            continue

        # Count missing values
        if is_list_column and list_empty_as_missing:
            # For list columns, count nulls and empty lists
            missing_mask = col.is_null() | (col.list.len() == 0)
            missing_count = missing_mask.sum()
        else:
            # For regular columns, just count nulls
            missing_count = col.null_count()

        # Calculate percentage
        missing_percent = missing_count / total_rows if total_rows > 0 else 0

        # Add to results
        results.append(
            {
                "column": col_name,
                "dtype": col_dtype,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "is_list_column": is_list_column,
            }
        )

    # Convert to DataFrame
    result_df = pl.DataFrame(results)

    # Sort results
    if sort_by in ["column", "missing_count", "missing_percent"]:
        if sort_by == "column":
            result_df = result_df.sort("column")
        else:
            # Sort in descending order for counts and percentages
            result_df = result_df.sort(sort_by, descending=True)

    # Filter by threshold if specified
    if threshold is not None:
        result_df = result_df.filter(pl.col("missing_percent") >= threshold)

    # Format percentage for display
    result_df = result_df.with_columns(
        (pl.col("missing_percent") * 100).round(2).alias("missing_percent")
    )

    return result_df


def plot_missingness(
    df: pl.DataFrame,
    include_list_columns: bool = True,
    list_empty_as_missing: bool = True,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plot missingness in a Polars DataFrame.

    Args:
        df: Polars DataFrame to analyze
        include_list_columns: Whether to check for empty lists in list columns
        list_empty_as_missing: Whether to count empty lists as missing values
        threshold: Optional threshold to filter results (e.g., 0.05 for columns with >5% missing)
        figsize: Figure size for the plot
    """
    # Get missingness data
    miss_df = view_missingness(
        df,
        include_list_columns=include_list_columns,
        list_empty_as_missing=list_empty_as_missing,
        sort_by="missing_percent",
        threshold=threshold,
    )

    # Convert to pandas for plotting
    miss_pd = miss_df.to_pandas()

    # Create figure
    plt.figure(figsize=figsize)

    # Create horizontal bar chart
    bars = plt.barh(
        miss_pd["column"],
        miss_pd["missing_percent"],
        color="skyblue",
    )

    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1
        plt.text(
            label_x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
        )

    # Add labels and title
    plt.xlabel("Missing Values (%)")
    plt.ylabel("Column")
    plt.title("Percentage of Missing Values by Column")
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def summarize_missingness(
    df: pl.DataFrame,
    include_list_columns: bool = True,
    list_empty_as_missing: bool = True,
) -> Dict[str, Union[int, float, List[str]]]:
    """
    Provide a summary of missingness in a DataFrame.

    Args:
        df: Polars DataFrame to analyze
        include_list_columns: Whether to check for empty lists in list columns
        list_empty_as_missing: Whether to count empty lists as missing values

    Returns:
        Dictionary with summary statistics
    """
    # Get missingness data
    miss_df = view_missingness(
        df,
        include_list_columns=include_list_columns,
        list_empty_as_missing=list_empty_as_missing,
    )

    # Calculate summary statistics
    total_columns = miss_df.height
    columns_with_missing = miss_df.filter(pl.col("missing_count") > 0).height

    # Get columns with high missingness (>50%)
    high_missing_cols = miss_df.filter(pl.col("missing_percent") > 50)[
        "column"
    ].to_list()

    # Get columns with moderate missingness (10-50%)
    moderate_missing_cols = miss_df.filter(
        (pl.col("missing_percent") > 10) & (pl.col("missing_percent") <= 50)
    )["column"].to_list()

    # Calculate average missingness percentage
    avg_missing_percent = miss_df["missing_percent"].mean()

    # Return summary
    return {
        "total_columns": total_columns,
        "columns_with_missing": columns_with_missing,
        "percent_columns_with_missing": (
            round(columns_with_missing / total_columns * 100, 2)
            if total_columns > 0
            else 0
        ),
        "average_missing_percent": round(avg_missing_percent, 2),
        "high_missing_columns": high_missing_cols,
        "moderate_missing_columns": moderate_missing_cols,
    }
