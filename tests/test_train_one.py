"""Tests for train_one — the pure model-training function.

Hermetic: builds frames in-memory, calls train_one, asserts on returned artifacts.
"""

from pathlib import Path

import polars as pl
import pytest

from src.models.outcomes.train import train_one
from src.models.outcomes.hurdle import HurdleModel


def _synthetic_hurdle_frames():
    """Build minimal frames the hurdle pipeline can train on.

    Hurdle target: binary int column ``hurdle`` (pre-computed in feature table).
    List-typed columns (categories, mechanics, designers, artists, publishers,
    families) are required by the BGG preprocessor.

    Numeric columns are varied so that VarianceThreshold(0) does not drop all
    features.
    """
    import random

    rng = random.Random(42)

    n_train, n_tune, n_test = 60, 20, 20

    def _make_rows(n, year, id_offset=0):
        hurdle_vals = [1] * (n // 2) + [0] * (n - n // 2)
        # Vary numeric columns enough to survive VarianceThreshold(0)
        users_rated = [rng.randint(5, 200) for _ in range(n)]
        complexity = [round(rng.uniform(1.0, 5.0), 2) for _ in range(n)]
        rating = [round(rng.uniform(4.0, 9.0), 2) for _ in range(n)]
        min_players = [rng.choice([1, 2, 3]) for _ in range(n)]
        max_players = [rng.choice([2, 4, 6, 8]) for _ in range(n)]
        min_playtime = [rng.choice([15, 30, 45, 60]) for _ in range(n)]
        max_playtime = [rng.choice([60, 90, 120, 180]) for _ in range(n)]
        min_age = [rng.choice([6, 8, 10, 12, 14]) for _ in range(n)]
        num_weights = [rng.randint(1, 30) for _ in range(n)]
        # Vary categories/mechanics/designers to produce real OHE columns
        cat_pool = [["Strategy"], ["Party Game"], ["Abstract"], ["Wargame"], ["Economic"]]
        mech_pool = [["Area Control"], ["Deck Building"], ["Worker Placement"],
                     ["Auction"], ["Cooperative Game"]]
        designer_pool = [["Designer A"], ["Designer B"], ["Designer C"],
                         ["Designer D"], ["Designer E"]]
        categories = [rng.choice(cat_pool) for _ in range(n)]
        mechanics = [rng.choice(mech_pool) for _ in range(n)]
        designers = [rng.choice(designer_pool) for _ in range(n)]
        return pl.DataFrame({
            "game_id": list(range(id_offset + 1, id_offset + n + 1)),
            "year_published": [year] * n,
            "users_rated": users_rated,
            "hurdle": hurdle_vals,
            "num_weights": num_weights,
            "complexity": complexity,
            "rating": rating,
            "min_players": min_players,
            "max_players": max_players,
            "min_playtime": min_playtime,
            "max_playtime": max_playtime,
            "min_age": min_age,
            "name": [f"game_{id_offset + i}" for i in range(n)],
            "categories": categories,
            "mechanics": mechanics,
            "designers": designers,
            "artists": [["Artist A"]] * n,
            "publishers": [["Publisher A"]] * n,
            "families": [["Family A"]] * n,
        })

    train = _make_rows(n_train, year=2018, id_offset=0)
    tune = _make_rows(n_tune, year=2019, id_offset=n_train)
    test = _make_rows(n_test, year=2020, id_offset=n_train + n_tune)
    return train, tune, test


def test_train_one_returns_expected_artifacts(monkeypatch):
    train_df, tune_df, test_df = _synthetic_hurdle_frames()

    candidate_config = {
        "name": "logistic-hurdle",
        "algorithm": "logistic",
        "use_embeddings": False,
        "use_sample_weights": False,
    }

    out = train_one(
        model_type="hurdle",
        candidate_config=candidate_config,
        train_df=train_df,
        tune_df=tune_df,
        test_df=test_df,
    )

    assert "pipeline" in out
    assert "metrics" in out
    assert "parameters" in out
    assert "tune_predictions" in out
    assert "test_predictions" in out
    assert set(out["metrics"].keys()) >= {"train", "tune", "test"}
