"""Embedding training config resolution: whiten toggle + min_feature_count."""

import argparse

from src.models.embeddings.train import get_algorithm_params, parse_arguments
from src.utils.config import load_config


def test_whiten_defaults_to_none_not_true():
    """The CLI default must be None so config.yaml's whiten setting wins."""
    args = parse_arguments([])
    assert args.whiten is None


def test_no_whiten_flag_sets_false():
    assert parse_arguments(["--no-whiten"]).whiten is False


def test_whiten_flag_sets_true():
    assert parse_arguments(["--whiten"]).whiten is True


def test_min_feature_count_arg_defaults_none():
    assert parse_arguments([]).min_feature_count is None
    assert parse_arguments(["--min-feature-count", "25"]).min_feature_count == 25


def _ns(**kw) -> argparse.Namespace:
    base = dict(
        algorithm="pca",
        whiten=None,
        n_iter=5,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        epochs=None,
        batch_size=None,
        learning_rate=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def test_whiten_none_is_dropped_so_config_can_win():
    assert "whiten" not in get_algorithm_params(_ns(algorithm="pca", whiten=None))


def test_whiten_false_is_passed_through():
    assert get_algorithm_params(_ns(algorithm="pca", whiten=False)) == {"whiten": False}


def test_whiten_true_is_passed_through():
    assert get_algorithm_params(_ns(algorithm="pca", whiten=True)) == {"whiten": True}


def test_config_exposes_min_feature_count_default_ten():
    cfg = load_config()
    assert cfg.embeddings is not None
    assert cfg.embeddings.min_feature_count == 10
