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
