"""Candidate model specs for collection-model experiments.

A :class:`CollectionCandidate` describes one combination of (model type,
hyperparameter tuning strategy, optional feature subset, optional fixed
params, training-set downsampling) that we want to train and evaluate as a
distinct experiment. Candidates are persisted alongside their fitted
pipeline + metrics so multiple candidates can be compared after the fact
without retraining.

Comparison is **not** part of this module — see
:mod:`src.collection.candidate_comparison`. Training is **not** part of this
module — see :mod:`src.collection.candidate_runner`.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Literal, Optional

from src.collection.collection_model import (
    ClassificationModelConfig,
    RegressionModelConfig,
)


_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass
class CollectionCandidate:
    """One named experiment configuration.

    A candidate carries everything needed to reproduce a run *except* the
    splits — those live separately in storage so multiple candidates can
    share the same val/test for honest comparison.

    Args:
        name: Filesystem-safe candidate identifier. Used in storage paths
            (e.g. ``{outcome}/{name}/v{N}/``). Letters, digits, ``_``, ``-``;
            must start with a letter. Cannot start with ``v`` (reserved for
            version directories) or ``_`` (reserved for ``_splits``).
        classification_config: Required when the outcome is classification.
        regression_config: Required when the outcome is regression.
        feature_columns: Optional explicit feature subset. ``None`` means
            "use whatever columns are in the training frame". When set, the
            runner slices the train/val/test frames to ``feature_columns``
            plus any protected columns (``game_id``, ``label``, etc.).
        tuning: ``"holdout"`` (val-set tuning), ``"cv"`` (k-fold on train),
            or ``"none"`` (skip tuning, fit ``fixed_params``).
        cv_folds: Folds for ``tuning="cv"``. Ignored otherwise.
        fixed_params: Required when ``tuning="none"``; otherwise unused.
            Keys use the same ``model__<param>`` shape that the tuners
            return as ``best_params``.
        downsample_negatives_ratio: If set, applied to the *training frame
            only* before tuning. ``None`` disables. The protected-min-ratings
            floor for downsampling is ``downsample_protect_min_ratings``.
        downsample_protect_min_ratings: Negatives with
            ``users_rated >= this`` are always kept; only the low-rating
            tail gets sampled out.
        notes: Free-form description (recorded in the registration metadata).
    """

    name: str
    classification_config: Optional[ClassificationModelConfig] = None
    regression_config: Optional[RegressionModelConfig] = None
    feature_columns: Optional[List[str]] = None
    tuning: Literal["holdout", "cv", "none"] = "cv"
    cv_folds: int = 5
    fixed_params: Optional[Dict[str, Any]] = None
    downsample_negatives_ratio: Optional[float] = None
    downsample_protect_min_ratings: int = 25
    notes: str = ""

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(
                f"Invalid candidate name {self.name!r}: must match {_NAME_RE.pattern}"
            )
        if self.name.startswith("v") or self.name.startswith("_"):
            raise ValueError(
                f"Candidate name {self.name!r} cannot start with 'v' or '_' "
                "(reserved for version and internal directories)."
            )
        if self.tuning not in ("holdout", "cv", "none"):
            raise ValueError(f"Invalid tuning strategy: {self.tuning!r}")
        if self.tuning == "cv" and self.cv_folds < 2:
            raise ValueError(
                f"cv_folds must be >= 2 when tuning='cv', got {self.cv_folds}"
            )
        if self.tuning == "none" and self.fixed_params is None:
            raise ValueError(
                "fixed_params is required when tuning='none' "
                "(set fixed_params={} to fit with model defaults)"
            )
        if self.downsample_negatives_ratio is not None and self.downsample_negatives_ratio <= 0:
            raise ValueError(
                f"downsample_negatives_ratio must be > 0 (got {self.downsample_negatives_ratio})"
            )
        if self.downsample_protect_min_ratings < 0:
            raise ValueError(
                f"downsample_protect_min_ratings must be >= 0 "
                f"(got {self.downsample_protect_min_ratings})"
            )

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly representation suitable for ``registration.json``."""
        d = asdict(self)
        # asdict already turns the inner dataclass configs into dicts.
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollectionCandidate":
        """Inverse of :meth:`to_dict`. Reconstructs nested config dataclasses
        from their dict form. Unknown fields are ignored to allow the spec
        to evolve without breaking older saved runs.
        """
        known = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {k: v for k, v in data.items() if k in known}

        if isinstance(kwargs.get("classification_config"), dict):
            kwargs["classification_config"] = ClassificationModelConfig(
                **kwargs["classification_config"]
            )
        if isinstance(kwargs.get("regression_config"), dict):
            kwargs["regression_config"] = RegressionModelConfig(
                **kwargs["regression_config"]
            )
        return cls(**kwargs)
