"""Tests for MinCountSelector — drop rare binary indicator columns."""

import numpy as np
import pandas as pd

from src.features.transformers import MinCountSelector


def _frame():
    return pd.DataFrame(
        {
            "cont": np.arange(100.0),
            "common_dummy": ([1] * 40) + ([0] * 60),
            "rare_dummy": ([1] * 3) + ([0] * 97),
        }
    )


def test_drops_rare_binary_columns():
    out = MinCountSelector(min_count=10).fit_transform(_frame())
    assert "rare_dummy" not in out.columns
    assert "common_dummy" in out.columns


def test_keeps_continuous_regardless_of_sum():
    df = pd.DataFrame({"x": [0.01] * 100})  # tiny sum, but continuous
    out = MinCountSelector(min_count=10).fit_transform(df)
    assert "x" in out.columns


def test_transform_applies_fit_column_set():
    sel = MinCountSelector(min_count=10).fit(_frame())
    out = sel.transform(_frame().iloc[:5])  # rare_dummy all-1 in this slice
    assert "rare_dummy" not in out.columns
    assert list(out.columns) == list(sel.get_feature_names_out())


def test_get_feature_names_out_reflects_kept_columns():
    sel = MinCountSelector(min_count=10).fit(_frame())
    assert set(sel.get_feature_names_out()) == {"cont", "common_dummy"}


def test_min_count_boundary_is_inclusive_keep():
    # a feature on exactly min_count games is kept
    df = pd.DataFrame({"d": ([1] * 10) + ([0] * 90)})
    out = MinCountSelector(min_count=10).fit_transform(df)
    assert "d" in out.columns


def test_nan_in_binary_column_counts_as_absent():
    df = pd.DataFrame({"d": [1, 1, np.nan, 0, 0], "n": [1.0, 2.0, 3.0, 4.0, 5.0]})
    sel = MinCountSelector(min_count=3).fit(df)
    assert "d" in sel.columns_to_drop_  # only 2 present
