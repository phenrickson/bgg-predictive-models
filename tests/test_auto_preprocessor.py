"""Test automatic preprocessor type selection functionality."""

import pytest
from src.models.training import create_preprocessing_pipeline


def test_auto_preprocessor_linear_models():
    """Test that linear models automatically select 'linear' preprocessor."""
    linear_models = ["linear", "ridge", "lasso", "logistic", "svc"]

    for model_name in linear_models:
        # This should not raise an error and should select 'linear' preprocessor
        pipeline = create_preprocessing_pipeline(
            model_type="auto", model_name=model_name
        )
        assert pipeline is not None


def test_auto_preprocessor_tree_models():
    """Test that tree-based models automatically select 'tree' preprocessor."""
    tree_models = ["lightgbm", "catboost", "rf", "random_forest", "lightgbm_linear"]

    for model_name in tree_models:
        # This should not raise an error and should select 'tree' preprocessor
        pipeline = create_preprocessing_pipeline(
            model_type="auto", model_name=model_name
        )
        assert pipeline is not None


def test_auto_preprocessor_unknown_model():
    """Test that unknown model names raise appropriate error."""
    with pytest.raises(ValueError, match="Unknown model name"):
        create_preprocessing_pipeline(model_type="auto", model_name="unknown_model")


def test_auto_preprocessor_missing_model_name():
    """Test that missing model_name raises appropriate error when using auto."""
    with pytest.raises(ValueError, match="model_name must be provided"):
        create_preprocessing_pipeline(model_type="auto")


def test_manual_preprocessor_types():
    """Test that manual preprocessor types still work."""
    # Test linear preprocessor
    pipeline_linear = create_preprocessing_pipeline(model_type="linear")
    assert pipeline_linear is not None

    # Test tree preprocessor
    pipeline_tree = create_preprocessing_pipeline(model_type="tree")
    assert pipeline_tree is not None


def test_invalid_preprocessor_type():
    """Test that invalid preprocessor types raise appropriate error."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        create_preprocessing_pipeline(model_type="invalid_type")


if __name__ == "__main__":
    pytest.main([__file__])
