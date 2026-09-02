"""diagnose_components summarises each PCA component by loading x feature prevalence."""

import numpy as np
import pandas as pd

from src.models.embeddings.diagnose_components import (
    _feature_prevalence,
    summarize_components,
)


def test_feature_prevalence_aligns_by_name_not_position():
    # transformed frame carries an extra column (year_published_transformed) that
    # the trainer drops from feature_names — positional alignment would shift.
    frame = pd.DataFrame(
        {
            "year_published_transformed": [0.1, -0.3, 0.5, 0.2],
            "dummy_a": [1, 0, 1, 1],
            "cont_x": [3.0, 1.5, 9.0, 4.0],
            "dummy_b": [0, 0, 1, 0],
        }
    )
    feature_names = ["dummy_a", "cont_x", "dummy_b"]

    prev = _feature_prevalence(frame, feature_names)

    assert prev[0] == 0.75  # dummy_a, aligned by name despite the leading extra col
    assert np.isnan(prev[1])  # cont_x is not binary
    assert prev[2] == 0.25  # dummy_b


def test_feature_prevalence_nan_for_names_absent_from_frame():
    frame = pd.DataFrame({"dummy_a": [1, 0, 1]})
    prev = _feature_prevalence(frame, ["dummy_a", "missing_feature"])
    assert prev[0] == 2 / 3
    assert np.isnan(prev[1])


def test_summarize_flags_rare_feature_dominated_component():
    # component 0 loads almost entirely on feature index 2 (a rare feature)
    components = np.array(
        [
            [0.02, 0.03, 0.99, 0.01],
            [0.5, 0.5, 0.0, 0.5],
        ]
    )
    feature_names = ["common_a", "common_b", "rare_x", "common_c"]
    prevalence = np.array([0.40, 0.35, 0.0006, 0.30])

    rows = summarize_components(components, feature_names, prevalence, top_k=3)

    comp0 = rows[0]
    assert comp0["component"] == 0
    assert comp0["top_features"][0]["feature"] == "rare_x"
    assert comp0["top_features"][0]["prevalence"] == 0.0006
    # concentration = max(loading^2) / sum(loading^2), ~1.0 here
    assert comp0["concentration"] > 0.9
    assert comp0["min_prevalence_in_top"] == 0.0006

    comp1 = rows[1]
    assert comp1["concentration"] < 0.5  # spread across three features
    assert comp1["min_prevalence_in_top"] == 0.30


def test_explained_variance_ratio_is_carried_through_when_given():
    components = np.array([[1.0, 0.0], [0.0, 1.0]])
    rows = summarize_components(
        components,
        ["a", "b"],
        np.array([0.5, 0.5]),
        top_k=1,
        explained_variance_ratio=np.array([0.7, 0.3]),
    )
    assert rows[0]["explained_variance_ratio"] == 0.7
    assert rows[1]["explained_variance_ratio"] == 0.3


def test_nan_prevalence_for_continuous_features_is_ignored_in_min():
    components = np.array([[0.9, 0.1, 0.4]])
    rows = summarize_components(
        components,
        ["cont_feature", "b", "c"],
        np.array([np.nan, 0.2, 0.15]),
        top_k=3,
    )
    # min over the top features ignores the NaN (continuous) entry
    assert rows[0]["min_prevalence_in_top"] == 0.15
