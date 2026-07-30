"""Label predictions as fitted values or forecasts.

The training cutoff is a property of the model, not the game, so it is read from
the registration of the model actually loaded. Reading it from config.yaml would
let the label drift from the deployed artifact the next time a model is refit,
which is the whole thing this is meant to prevent.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resolve_training_cutoff_year(registrations: Dict[str, Dict[str, Any]]) -> int:
    """Return the year the loaded models were fitted through.

    Models are refit through their test year, so `test_through` in the
    registration is the training cutoff.

    Args:
        registrations: Registration dicts keyed by target (hurdle, rating, ...)

    Returns:
        The training cutoff year. If the models disagree, the minimum, so that
        in_sample means "seen by every model".

    Raises:
        ValueError: If any registration has no test_through to read.
    """
    cutoffs = {}
    for target, registration in registrations.items():
        metadata = registration.get("original_experiment", {}).get("metadata", {})
        cutoff = metadata.get("test_through")
        if cutoff is None:
            raise ValueError(
                f"Registration for '{target}' has no test_through in "
                f"original_experiment.metadata -- cannot determine the training "
                f"cutoff, and guessing it would make sample_status meaningless"
            )
        cutoffs[target] = int(cutoff)

    if len(set(cutoffs.values())) > 1:
        logger.warning(
            f"Models disagree on training cutoff: {cutoffs}. "
            f"Using {min(cutoffs.values())} so in_sample means seen by every model."
        )
    return min(cutoffs.values())


def compute_sample_status(
    year_published: pd.Series, training_cutoff_year: int
) -> pd.Series:
    """Label each prediction as a fitted value or a forecast.

    Binary: games without a year_published are never scored, so there is no
    third case to represent.
    """
    return pd.Series(
        np.where(year_published <= training_cutoff_year, "in_sample", "out_of_sample"),
        index=year_published.index,
        name="sample_status",
    )
