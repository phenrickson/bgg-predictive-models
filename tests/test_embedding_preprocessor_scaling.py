"""The embedding preprocessor uses Gelman scaling + a min-count floor,
not a blanket StandardScaler."""

from sklearn.preprocessing import StandardScaler

from src.features.transformers import MinCountSelector, TwoSDScaler
from src.models.embeddings.transformer import create_embedding_preprocessor


def _step_types(pipe):
    return [type(est) for _, est in pipe.steps]


def test_linear_pipeline_uses_two_sd_scaler_not_standardscaler():
    pipe = create_embedding_preprocessor(model_type="linear")
    types = _step_types(pipe)
    assert TwoSDScaler in types
    assert StandardScaler not in types


def test_linear_pipeline_has_min_count_step():
    pipe = create_embedding_preprocessor(model_type="linear")
    assert any(isinstance(est, MinCountSelector) for _, est in pipe.steps)


def test_min_count_is_configurable():
    pipe = create_embedding_preprocessor(model_type="linear", min_feature_count=25)
    sel = next(est for _, est in pipe.steps if isinstance(est, MinCountSelector))
    assert sel.min_count == 25


def test_min_count_defaults_to_ten():
    pipe = create_embedding_preprocessor(model_type="linear")
    sel = next(est for _, est in pipe.steps if isinstance(est, MinCountSelector))
    assert sel.min_count == 10


def test_scaler_runs_after_min_count():
    pipe = create_embedding_preprocessor(model_type="linear")
    names = [name for name, _ in pipe.steps]
    assert names.index("min_count") < names.index("scaler")


def test_tree_pipeline_unchanged():
    pipe = create_embedding_preprocessor(model_type="tree")
    types = _step_types(pipe)
    assert TwoSDScaler not in types
    assert MinCountSelector not in types
