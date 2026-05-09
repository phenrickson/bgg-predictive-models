"""K-fold out-of-fold prediction utility.

Given an unfitted pipeline and a (X, y) frame, produce predictions for
every row of X using only models that did not see that row. Used to
generate honest training-time features for downstream cascaded models
(complexity → rating, etc).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def kfold_oof_predict(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 5,
    seed: int = 42,
    predict_proba: bool = False,
) -> np.ndarray:
    """Return out-of-fold predictions for every row of X.

    Args:
        pipeline: Unfitted sklearn pipeline. Cloned per fold.
        X: Feature frame.
        y: Target series.
        k: Number of folds.
        seed: Random seed for fold assignment.
        predict_proba: If True, return probability of the positive class
            (binary classification). Default False.
    """
    n = len(X)
    out = np.zeros(n, dtype=float)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_idx, val_idx in kf.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)
        if predict_proba:
            proba = fold_pipeline.predict_proba(X_val)
            out[val_idx] = proba[:, 1]
        else:
            out[val_idx] = fold_pipeline.predict(X_val)
    return out
