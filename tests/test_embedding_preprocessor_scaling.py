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


def test_imputer_adds_no_missing_indicators():
    from sklearn.impute import SimpleImputer

    pipe = create_embedding_preprocessor(model_type="linear")
    imp = next(est for _, est in pipe.steps if isinstance(est, SimpleImputer))
    assert imp.add_indicator is False


def test_player_count_is_continuous_not_one_hot():
    import pandas as pd

    df = pd.read_parquet("tests/fixtures/sample_games.parquet")
    pipe = create_embedding_preprocessor(model_type="linear", min_feature_count=1)
    out = pipe.fit_transform(df)
    cols = list(out.columns)
    assert not any(c.startswith("player_count_") and c[-1].isdigit() for c in cols)
    for c in ("min_players", "max_players", "supports_solo"):
        assert c in cols, c
    assert "player_count_range" not in cols
    assert not any(c.startswith("missingindicator") for c in cols)
