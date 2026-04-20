"""Tests for src.collection.outcomes."""

import pytest
from src.collection.outcomes import (
    OutcomeDefinition,
    DirectColumnRule,
    AnyOfRule,
    PredicateRule,
    load_outcomes,
)


SAMPLE_CONFIG = {
    "collections": {
        "outcomes": {
            "own": {"task": "classification", "label_from": "owned"},
            "ever_owned": {
                "task": "classification",
                "label_from": {"any_of": ["owned", "prev_owned"]},
            },
            "rated": {
                "task": "classification",
                "label_from": {"column": "user_rating", "predicate": "> 0"},
            },
            "rating": {
                "task": "regression",
                "label_from": "user_rating",
                "require": "user_rating > 0",
            },
            "love": {
                "task": "classification",
                "label_from": {"column": "user_rating", "predicate": ">= 8"},
                "require": "user_rating > 0",
            },
        }
    }
}


def test_load_outcomes_returns_five_definitions():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    assert set(outcomes.keys()) == {"own", "ever_owned", "rated", "rating", "love"}


def test_own_is_direct_column_rule():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    own = outcomes["own"]
    assert own.task == "classification"
    assert isinstance(own.label_rule, DirectColumnRule)
    assert own.label_rule.column == "owned"
    assert own.require is None


def test_ever_owned_is_any_of_rule():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    ever = outcomes["ever_owned"]
    assert isinstance(ever.label_rule, AnyOfRule)
    assert ever.label_rule.columns == ["owned", "prev_owned"]


def test_rated_is_predicate_rule():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    rated = outcomes["rated"]
    assert isinstance(rated.label_rule, PredicateRule)
    assert rated.label_rule.column == "user_rating"
    assert rated.label_rule.operator == ">"
    assert rated.label_rule.value == 0


def test_rating_is_regression_with_require():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    r = outcomes["rating"]
    assert r.task == "regression"
    assert isinstance(r.label_rule, DirectColumnRule)
    assert r.require == "user_rating > 0"


def test_love_has_ge_predicate():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    love = outcomes["love"]
    assert isinstance(love.label_rule, PredicateRule)
    assert love.label_rule.operator == ">="
    assert love.label_rule.value == 8


def test_invalid_task_raises():
    bad_config = {"collections": {"outcomes": {"x": {"task": "clustering", "label_from": "y"}}}}
    with pytest.raises(ValueError, match="task"):
        load_outcomes(bad_config)


def test_invalid_predicate_operator_raises():
    bad_config = {
        "collections": {
            "outcomes": {
                "x": {
                    "task": "classification",
                    "label_from": {"column": "y", "predicate": "~= 5"},
                }
            }
        }
    }
    with pytest.raises(ValueError, match="operator"):
        load_outcomes(bad_config)


import polars as pl
from src.collection.outcomes import apply_outcome


def _fixture_df() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5],
        "owned": [True, False, False, True, False],
        "prev_owned": [False, True, False, False, False],
        "user_rating": [8.0, 0.0, 9.5, 7.0, 0.0],
    })


def test_apply_outcome_own_binary():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["own"])
    assert out["label"].to_list() == [True, False, False, True, False]
    assert len(out) == 5  # no require filter


def test_apply_outcome_ever_owned_any_of():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["ever_owned"])
    assert out["label"].to_list() == [True, True, False, True, False]


def test_apply_outcome_rated_predicate():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["rated"])
    assert out["label"].to_list() == [True, False, True, True, False]


def test_apply_outcome_rating_regression_with_require():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["rating"])
    # require drops user_rating <= 0 (rows 2 and 5)
    assert out["game_id"].to_list() == [1, 3, 4]
    assert out["label"].to_list() == [8.0, 9.5, 7.0]


def test_apply_outcome_love_predicate_with_require():
    outcomes = load_outcomes(SAMPLE_CONFIG)
    out = apply_outcome(_fixture_df(), outcomes["love"])
    # require drops rows 2 and 5; love is user_rating >= 8
    assert out["game_id"].to_list() == [1, 3, 4]
    assert out["label"].to_list() == [True, True, False]
