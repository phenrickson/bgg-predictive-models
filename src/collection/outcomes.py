"""Outcome definitions for user collection models.

An OutcomeDefinition describes how to construct a training label from a
user's collection dataframe. Defined declaratively in config.yaml under
collections.outcomes; parsed into dataclasses here.

Consumers:
- CollectionProcessor / apply_outcome: constructs the label column
- CollectionSplitter: dispatches to classification vs regression strategy
- CollectionModel: dispatches to classification vs regression training path
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Union

import polars as pl


VALID_OPERATORS = {">", ">=", "<", "<=", "==", "!="}
VALID_TASKS = {"classification", "regression"}

_OPS = {
    ">":  lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
    "<":  lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
    "==": lambda c, v: c == v,
    "!=": lambda c, v: c != v,
}


@dataclass(frozen=True)
class DirectColumnRule:
    """Label is the value of a single column."""
    column: str


@dataclass(frozen=True)
class AnyOfRule:
    """Label is the boolean OR of multiple columns."""
    columns: List[str]


@dataclass(frozen=True)
class PredicateRule:
    """Label is the result of applying a comparison operator to a column."""
    column: str
    operator: str
    value: float


LabelRule = Union[DirectColumnRule, AnyOfRule, PredicateRule]


@dataclass(frozen=True)
class OutcomeDefinition:
    """Declarative description of one outcome model."""
    name: str
    task: Literal["classification", "regression"]
    label_rule: LabelRule
    require: str | None = None


def _parse_predicate(predicate_str: str) -> tuple[str, float]:
    """Parse 'operator value' string (e.g. '>= 8') into (operator, value)."""
    match = re.match(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$", predicate_str)
    if not match:
        raise ValueError(
            f"Invalid predicate {predicate_str!r}. "
            f"Expected format '<operator> <number>' with operator in {sorted(VALID_OPERATORS)}"
        )
    operator, value_str = match.groups()
    return operator, float(value_str)


def _parse_label_rule(spec: Any) -> LabelRule:
    """Parse one of the four primitives into a LabelRule dataclass."""
    if isinstance(spec, str):
        return DirectColumnRule(column=spec)
    if isinstance(spec, dict):
        if "any_of" in spec:
            cols = spec["any_of"]
            if not isinstance(cols, list) or len(cols) == 0 or not all(isinstance(c, str) for c in cols):
                raise ValueError(f"any_of must be a non-empty list of column names, got {cols!r}")
            return AnyOfRule(columns=list(cols))
        if "column" in spec and "predicate" in spec:
            operator, value = _parse_predicate(spec["predicate"])
            return PredicateRule(column=spec["column"], operator=operator, value=value)
    raise ValueError(f"Unrecognized label_from shape: {spec!r}")


def _parse_outcome(name: str, entry: Dict[str, Any]) -> OutcomeDefinition:
    task = entry.get("task")
    if task not in VALID_TASKS:
        raise ValueError(f"Outcome {name!r} has invalid task {task!r}. Valid: {sorted(VALID_TASKS)}")
    label_from = entry.get("label_from")
    if label_from is None:
        raise ValueError(f"Outcome {name!r} missing required 'label_from'")
    label_rule = _parse_label_rule(label_from)
    require = entry.get("require")
    if require is not None and not isinstance(require, str):
        raise ValueError(f"Outcome {name!r} 'require' must be a string, got {require!r}")
    return OutcomeDefinition(name=name, task=task, label_rule=label_rule, require=require)


def load_outcomes(config: Dict[str, Any]) -> Dict[str, OutcomeDefinition]:
    """Parse config.collections.outcomes into a registry of OutcomeDefinitions."""
    outcomes_section = config.get("collections", {}).get("outcomes", {})
    if not outcomes_section:
        raise ValueError("config.collections.outcomes is empty or missing")
    return {name: _parse_outcome(name, entry) for name, entry in outcomes_section.items()}


def _apply_require(df: pl.DataFrame, require: str) -> pl.DataFrame:
    """Filter rows by a 'col op value' expression."""
    match = re.match(r"^\s*(\w+)\s+(>=|<=|==|!=|>|<)\s+(-?\d+(?:\.\d+)?)\s*$", require)
    if not match:
        raise ValueError(f"Invalid require expression {require!r}")
    column, operator, value_str = match.groups()
    value = float(value_str)
    return df.filter(_OPS[operator](pl.col(column), value))


def _apply_label_rule(df: pl.DataFrame, rule: LabelRule) -> pl.DataFrame:
    """Add a 'label' column from the given rule."""
    if isinstance(rule, DirectColumnRule):
        return df.with_columns(pl.col(rule.column).alias("label"))
    if isinstance(rule, AnyOfRule):
        expr = pl.col(rule.columns[0]).cast(pl.Boolean)
        for c in rule.columns[1:]:
            expr = expr | pl.col(c).cast(pl.Boolean)
        return df.with_columns(expr.alias("label"))
    if isinstance(rule, PredicateRule):
        return df.with_columns(_OPS[rule.operator](pl.col(rule.column), rule.value).alias("label"))
    raise TypeError(f"Unknown rule type: {type(rule)!r}")


def apply_outcome(df: pl.DataFrame, outcome: OutcomeDefinition) -> pl.DataFrame:
    """Apply the outcome's require filter (if any), then add a label column.

    Nulls in source columns propagate into the label column (polars three-valued
    logic). Callers training on the result should drop null-label rows first.
    """
    filtered = df if outcome.require is None else _apply_require(df, outcome.require)
    return _apply_label_rule(filtered, outcome.label_rule)
