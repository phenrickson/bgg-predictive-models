"""Tests for TwoSDScaler — Gelman-style scaling (continuous / 2*SD, dummies at 0/1)."""

import numpy as np
import pandas as pd
import pytest

from src.features.transformers import TwoSDScaler


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "playtime": rng.normal(60, 30, 500),  # continuous
            "complexity": rng.uniform(1, 5, 500),  # continuous
            "mechanic_dice": rng.integers(0, 2, 500),  # binary dummy
            "category_war": rng.integers(0, 2, 500),  # binary dummy
        }
    )


def test_continuous_columns_get_mean_zero_sd_half(frame):
    out = TwoSDScaler().fit_transform(frame)
    for col in ["playtime", "complexity"]:
        assert out[col].mean() == pytest.approx(0.0, abs=1e-9)
        # divided by 2 SD -> resulting population SD is ~0.5
        assert out[col].std(ddof=0) == pytest.approx(0.5, rel=1e-6)


def test_binary_columns_pass_through_untouched(frame):
    out = TwoSDScaler().fit_transform(frame)
    for col in ["mechanic_dice", "category_war"]:
        pd.testing.assert_series_equal(out[col], frame[col], check_dtype=False)


def test_explicit_continuous_columns_override(frame):
    # force 'complexity' to be treated as pass-through
    out = TwoSDScaler(continuous_columns=["playtime"]).fit_transform(frame)
    pd.testing.assert_series_equal(
        out["complexity"], frame["complexity"], check_dtype=False
    )
    assert out["playtime"].std(ddof=0) == pytest.approx(0.5, rel=1e-6)


def test_transform_applies_fit_time_statistics(frame):
    """transform must use fit-time mean/scale, not re-fit on the new frame."""
    scaler = TwoSDScaler(continuous_columns=["playtime"]).fit(frame)
    shifted = frame.copy()
    shifted["playtime"] = shifted["playtime"] + 100.0
    base = scaler.transform(frame)
    moved = scaler.transform(shifted)
    scale = 2 * frame["playtime"].std(ddof=0)
    # a +100 raw shift becomes a constant +100/scale shift in the output
    assert (moved["playtime"] - base["playtime"]).to_numpy() == pytest.approx(
        100.0 / scale
    )


def test_unknown_columns_pass_through(frame):
    extra = frame.assign(game_id=range(len(frame)))
    out = TwoSDScaler(continuous_columns=["playtime"]).fit_transform(extra)
    pd.testing.assert_series_equal(out["game_id"], extra["game_id"], check_dtype=False)


def test_get_feature_names_out_is_identity(frame):
    scaler = TwoSDScaler().fit(frame)
    assert list(scaler.get_feature_names_out()) == list(frame.columns)


def test_zero_sd_column_is_safe():
    df = pd.DataFrame({"const": [3.0] * 10, "x": np.arange(10.0)})
    out = TwoSDScaler(continuous_columns=["const", "x"]).fit_transform(df)
    assert np.isfinite(out["const"]).all()  # no divide-by-zero blow-up


def test_nan_values_do_not_break_binary_detection():
    df = pd.DataFrame(
        {
            "dummy": [0, 1, 1, np.nan, 0, 1],  # binary apart from a missing value
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    scaler = TwoSDScaler().fit(df)
    assert "dummy" not in scaler.continuous_columns_
    assert "num" in scaler.continuous_columns_
