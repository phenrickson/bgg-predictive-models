"""Tests for PlayerCountSanitizer (embedding-only player-count cleanup)."""

import numpy as np
import pandas as pd

from src.models.embeddings.transformer import PlayerCountSanitizer


def _df(rows):
    return pd.DataFrame(rows, columns=["game_id", "min_players", "max_players"])


def test_caps_max_players_at_cap():
    out = PlayerCountSanitizer(cap=12).fit_transform(
        _df([[1, 2, 99], [2, 1, 2000], [3, 2, 4]])
    )
    assert list(out["max_players"]) == [12, 12, 4]


def test_min_clipped_to_max_and_repairs_inversion():
    # game-year fat-fingered into min_players: 2024 / 2  ->  min collapses to max
    out = PlayerCountSanitizer(cap=12).fit_transform(_df([[1, 2024, 2], [2, 5, 4]]))
    assert list(out["min_players"]) == [2, 4]


def test_supports_solo_flag():
    out = PlayerCountSanitizer(cap=12).fit_transform(
        _df([[1, 1, 4], [2, 2, 5], [3, 1, 1]])
    )
    assert list(out["supports_solo"]) == [1.0, 0.0, 1.0]


def test_values_below_one_become_missing():
    out = PlayerCountSanitizer(cap=12).fit_transform(_df([[1, 0, 4], [2, 2, 0]]))
    assert np.isnan(out.loc[0, "min_players"])
    assert np.isnan(out.loc[1, "max_players"])
    # missing min is not solo
    assert list(out["supports_solo"]) == [0.0, 0.0]


def test_missing_max_caps_min_at_cap():
    out = PlayerCountSanitizer(cap=12).fit_transform(_df([[1, 50, 0]]))
    assert out.loc[0, "min_players"] == 12
    assert np.isnan(out.loc[0, "max_players"])


def test_passthrough_when_columns_absent():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = PlayerCountSanitizer().fit_transform(df)
    pd.testing.assert_frame_equal(out, df)


def test_get_feature_names_out_adds_solo():
    s = PlayerCountSanitizer().fit(_df([[1, 2, 4]]))
    assert "supports_solo" in list(s.get_feature_names_out())
