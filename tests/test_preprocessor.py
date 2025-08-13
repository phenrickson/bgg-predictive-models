import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.features.transformers import BaseBGGTransformer
from src.features.preprocessor import create_bgg_preprocessor


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing transformer features."""
    return pd.read_parquet("data/raw/game_features.parquet").sample(
        n=10000, random_state=42
    )


def test_create_bgg_preprocessor_model_types(sample_dataframe):
    """Test the create_bgg_preprocessor function with different model types."""
    # Test linear model preprocessing
    linear_preprocessor = create_bgg_preprocessor(model_type="linear")
    assert isinstance(linear_preprocessor, Pipeline)

    # Verify linear model pipeline steps
    linear_steps = dict(linear_preprocessor.steps)
    assert "bgg_preprocessor" in linear_steps
    assert "impute" in linear_steps
    assert "log" in linear_steps
    assert "year" in linear_steps
    assert "variance_selector" in linear_steps
    assert "scaler" in linear_steps

    # Transform sample data with linear preprocessor
    linear_transformed = linear_preprocessor.fit_transform(sample_dataframe)
    assert isinstance(linear_transformed, pd.DataFrame)

    # Test tree model preprocessing
    tree_preprocessor = create_bgg_preprocessor(model_type="tree")
    assert isinstance(tree_preprocessor, Pipeline)

    # Verify tree model pipeline steps
    tree_steps = dict(tree_preprocessor.steps)
    assert "bgg_preprocessor" in tree_steps
    assert "impute" in tree_steps
    assert "variance_selector" in tree_steps
    assert "log" not in tree_steps
    assert "year" not in tree_steps
    assert "scaler" not in tree_steps

    # Transform sample data with tree preprocessor
    tree_transformed = tree_preprocessor.fit_transform(sample_dataframe)
    assert isinstance(tree_transformed, pd.DataFrame)

    # Verify the presence of specific transformations
    linear_columns = set(linear_transformed.columns)
    tree_columns = set(tree_transformed.columns)

    # Check for transformations specific to linear models
    assert "year_published_transformed" in linear_columns
    assert "year_published_transformed" not in tree_columns

    # Verify that the core feature engineering remains consistent
    core_features = {
        "mechanics_count",
        "categories_count",
        "time_per_player",
        "min_age",
        "min_playtime",
        "max_playtime",
    }

    # Verify core features are present
    assert core_features.issubset(linear_columns), (
        "Core features missing in linear model"
    )
    assert core_features.issubset(tree_columns), "Core features missing in tree model"

    # Test invalid model type raises ValueError
    with pytest.raises(ValueError, match="Unsupported model_type"):
        create_bgg_preprocessor(model_type="invalid_type")


def test_bgg_preprocessor_feature_scaling(sample_dataframe):
    """Test feature scaling and transformation for linear models."""
    # Create linear preprocessor
    linear_preprocessor = create_bgg_preprocessor(model_type="linear")

    # Fit and transform the data
    linear_transformed = linear_preprocessor.fit_transform(sample_dataframe)

    # Check scaling for specific features
    scaled_features = ["time_per_player", "min_playtime", "max_playtime"]

    for feature in scaled_features:
        # Verify the feature is scaled (mean close to 0, std close to 1)
        feature_data = linear_transformed[feature]
        assert np.isclose(feature_data.mean(), 0, atol=1e-2), (
            f"{feature} mean should be close to 0"
        )
        assert np.isclose(feature_data.std(), 1, atol=1e-2), (
            f"{feature} std should be close to 1"
        )


def test_bgg_preprocessor_log_transformation(sample_dataframe):
    """Test logarithmic transformations for linear models."""

    # Define log columns to test
    log_columns = [
        "min_age",
        "min_playtime",
        "max_playtime",
        "time_per_player",
        "description_word_count",
    ]

    # Create linear preprocessor with specified log columns
    linear_preprocessor = create_bgg_preprocessor(
        model_type="linear", log_columns=log_columns
    )

    # get original, processed
    processed_dataframe = linear_preprocessor.named_steps[
        "bgg_preprocessor"
    ].fit_transform(sample_dataframe)

    # Fit and transform the data
    linear_transformed = linear_preprocessor.fit_transform(sample_dataframe)

    # Check log-transformed features
    for feature in log_columns:
        # Verify log-transformed feature exists
        assert f"{feature}" in linear_transformed.columns, (
            f"Log transformation missing for {feature}"
        )

        # Verify log transformation reduces skewness
        original_skew = processed_dataframe[feature].skew()
        log_transformed_skew = linear_transformed[f"{feature}"].skew()

        # Log transformation should reduce skewness
        assert abs(log_transformed_skew) < abs(original_skew), (
            f"Log transformation did not reduce skewness for {feature}"
        )


def test_bgg_preprocessor_variance_selection():
    """Test variance-based feature selection and feature differences between model types."""
    # Use the sample dataframe to test variance selection
    sample_dataframe = pd.read_parquet("data/raw/game_features.parquet").sample(
        n=10000, random_state=42
    )

    # Test both linear and tree preprocessors
    linear_preprocessor = create_bgg_preprocessor(model_type="linear")
    tree_preprocessor = create_bgg_preprocessor(model_type="tree")

    # Transform data
    linear_transformed = linear_preprocessor.fit_transform(sample_dataframe)
    tree_transformed = tree_preprocessor.fit_transform(sample_dataframe)

    # Verify variance selector is applied
    linear_variance_selector = linear_preprocessor.named_steps["variance_selector"]
    tree_variance_selector = tree_preprocessor.named_steps["variance_selector"]

    # Verify different feature sets for linear and tree models
    assert set(linear_transformed.columns) != set(tree_transformed.columns), (
        "Linear and tree preprocessors should produce different feature sets"
    )

    # Verify core features are consistent across model types
    core_features = {
        "mechanics_count",
        "categories_count",
        "time_per_player",
        "min_age",
        "min_playtime",
        "max_playtime",
    }

    # Verify core features are present in both transformed datasets
    assert core_features.issubset(linear_transformed.columns), (
        "Core features missing in linear model"
    )
    assert core_features.issubset(tree_transformed.columns), (
        "Core features missing in tree model"
    )
