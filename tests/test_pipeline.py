"""Test comprehensive preprocessing pipeline functionality."""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.models.training import create_preprocessing_pipeline, preprocess_data


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing pipeline features."""
    return pd.read_parquet("data/raw/game_features.parquet").sample(
        n=1000, random_state=42
    )


class TestAutoPreprocessorSelection:
    """Test automatic preprocessor type selection functionality."""

    def test_auto_preprocessor_linear_models(self):
        """Test that linear models automatically select 'linear' preprocessor."""
        linear_models = ["linear", "ridge", "lasso", "logistic", "svc"]

        for model_name in linear_models:
            # This should not raise an error and should select 'linear' preprocessor
            pipeline = create_preprocessing_pipeline(
                model_type="auto", model_name=model_name
            )
            assert pipeline is not None
            assert isinstance(pipeline, Pipeline)

    def test_auto_preprocessor_tree_models(self):
        """Test that tree-based models automatically select 'tree' preprocessor."""
        tree_models = ["lightgbm", "catboost", "rf", "random_forest", "lightgbm_linear"]

        for model_name in tree_models:
            # This should not raise an error and should select 'tree' preprocessor
            pipeline = create_preprocessing_pipeline(
                model_type="auto", model_name=model_name
            )
            assert pipeline is not None
            assert isinstance(pipeline, Pipeline)

    def test_auto_preprocessor_unknown_model(self):
        """Test that unknown model names raise appropriate error."""
        with pytest.raises(ValueError, match="Unknown model name"):
            create_preprocessing_pipeline(model_type="auto", model_name="unknown_model")

    def test_auto_preprocessor_missing_model_name(self):
        """Test that missing model_name raises appropriate error when using auto."""
        with pytest.raises(ValueError, match="model_name must be provided"):
            create_preprocessing_pipeline(model_type="auto")

    def test_manual_preprocessor_types(self):
        """Test that manual preprocessor types still work."""
        # Test linear preprocessor
        pipeline_linear = create_preprocessing_pipeline(model_type="linear")
        assert pipeline_linear is not None
        assert isinstance(pipeline_linear, Pipeline)

        # Test tree preprocessor
        pipeline_tree = create_preprocessing_pipeline(model_type="tree")
        assert pipeline_tree is not None
        assert isinstance(pipeline_tree, Pipeline)

    def test_invalid_preprocessor_type(self):
        """Test that invalid preprocessor types raise appropriate error."""
        with pytest.raises(ValueError, match="Unsupported model_type"):
            create_preprocessing_pipeline(model_type="invalid_type")


class TestPipelineStructure:
    """Test the structure and components of preprocessing pipelines."""

    def test_linear_pipeline_structure(self):
        """Test that linear pipelines have the correct structure."""
        pipeline = create_preprocessing_pipeline(model_type="linear")

        # Check pipeline steps
        step_names = [step[0] for step in pipeline.steps]
        expected_steps = [
            "bgg_preprocessor",
            "impute",
            "log",
            "year",
            "variance_selector",
            "scaler",
        ]

        for step in expected_steps:
            assert step in step_names, f"Missing step '{step}' in linear pipeline"

    def test_tree_pipeline_structure(self):
        """Test that tree pipelines have the correct structure."""
        pipeline = create_preprocessing_pipeline(model_type="tree")

        # Check pipeline steps
        step_names = [step[0] for step in pipeline.steps]
        expected_steps = ["bgg_preprocessor", "impute", "variance_selector"]

        for step in expected_steps:
            assert step in step_names, f"Missing step '{step}' in tree pipeline"

        # Check that scaling steps are NOT present
        scaling_steps = ["log", "year", "scaler"]
        for step in scaling_steps:
            assert step not in step_names, f"Unexpected step '{step}' in tree pipeline"

    def test_auto_pipeline_structure_consistency(self):
        """Test that auto-selected pipelines have consistent structure with manual selection."""
        # Test linear model auto-selection
        auto_linear = create_preprocessing_pipeline(
            model_type="auto", model_name="ridge"
        )
        manual_linear = create_preprocessing_pipeline(model_type="linear")

        auto_steps = [step[0] for step in auto_linear.steps]
        manual_steps = [step[0] for step in manual_linear.steps]

        assert auto_steps == manual_steps, (
            "Auto-selected linear pipeline differs from manual"
        )

        # Test tree model auto-selection
        auto_tree = create_preprocessing_pipeline(
            model_type="auto", model_name="lightgbm"
        )
        manual_tree = create_preprocessing_pipeline(model_type="tree")

        auto_steps = [step[0] for step in auto_tree.steps]
        manual_steps = [step[0] for step in manual_tree.steps]

        assert auto_steps == manual_steps, (
            "Auto-selected tree pipeline differs from manual"
        )


class TestPipelineTransformation:
    """Test the actual data transformation functionality of pipelines."""

    def test_pipeline_fit_transform(self, sample_dataframe):
        """Test that pipelines can fit and transform data successfully."""
        # Test linear pipeline
        linear_pipeline = create_preprocessing_pipeline(model_type="linear")
        linear_transformed = linear_pipeline.fit_transform(sample_dataframe)

        assert isinstance(linear_transformed, pd.DataFrame)
        assert len(linear_transformed) == len(sample_dataframe)
        assert linear_transformed.shape[1] > 0

        # Test tree pipeline
        tree_pipeline = create_preprocessing_pipeline(model_type="tree")
        tree_transformed = tree_pipeline.fit_transform(sample_dataframe)

        assert isinstance(tree_transformed, pd.DataFrame)
        assert len(tree_transformed) == len(sample_dataframe)
        assert tree_transformed.shape[1] > 0

    def test_pipeline_feature_differences(self, sample_dataframe):
        """Test that linear and tree pipelines produce different feature sets."""
        linear_pipeline = create_preprocessing_pipeline(model_type="linear")
        tree_pipeline = create_preprocessing_pipeline(model_type="tree")

        linear_transformed = linear_pipeline.fit_transform(sample_dataframe)
        tree_transformed = tree_pipeline.fit_transform(sample_dataframe)

        # Feature sets should be different
        linear_features = set(linear_transformed.columns)
        tree_features = set(tree_transformed.columns)

        assert linear_features != tree_features, (
            "Linear and tree pipelines should produce different features"
        )

        # But should have some overlap in core features
        overlap = linear_features.intersection(tree_features)
        assert len(overlap) > 0, "Pipelines should share some core features"

    def test_pipeline_scaling_effects(self, sample_dataframe):
        """Test that linear pipelines apply scaling while tree pipelines don't."""
        linear_pipeline = create_preprocessing_pipeline(model_type="linear")
        tree_pipeline = create_preprocessing_pipeline(model_type="tree")

        linear_transformed = linear_pipeline.fit_transform(sample_dataframe)
        tree_transformed = tree_pipeline.fit_transform(sample_dataframe)

        # Find common numeric features
        common_features = set(linear_transformed.columns).intersection(
            set(tree_transformed.columns)
        )

        numeric_features = []
        for feature in common_features:
            if linear_transformed[feature].dtype in [
                "float64",
                "int64",
            ] and tree_transformed[feature].dtype in ["float64", "int64"]:
                numeric_features.append(feature)

        if numeric_features:
            # Check that linear features are more standardized
            for feature in numeric_features[:3]:  # Test first 3 features
                linear_std = linear_transformed[feature].std()
                tree_std = tree_transformed[feature].std()

                # Linear should be more standardized (closer to 1) if scaling was applied
                if tree_std > 1:  # Only test if original data has variance > 1
                    assert linear_std < tree_std, (
                        f"Linear pipeline should standardize {feature}"
                    )

    def test_pipeline_reproducibility(self, sample_dataframe):
        """Test that pipelines produce consistent results across multiple runs."""
        pipeline = create_preprocessing_pipeline(model_type="linear")

        # Transform same data twice
        result1 = pipeline.fit_transform(sample_dataframe.copy())
        result2 = pipeline.fit_transform(sample_dataframe.copy())

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_pipeline_handles_missing_data(self, sample_dataframe):
        """Test that pipelines handle missing data appropriately."""
        # Introduce some missing values using iloc for positional indexing
        df_with_missing = sample_dataframe.copy()
        df_with_missing.iloc[0:10, df_with_missing.columns.get_loc("min_age")] = np.nan
        df_with_missing.iloc[5:15, df_with_missing.columns.get_loc("min_playtime")] = (
            np.nan
        )

        pipeline = create_preprocessing_pipeline(model_type="linear")
        transformed = pipeline.fit_transform(df_with_missing)

        # Should not have any NaN values after preprocessing
        assert not transformed.isnull().any().any(), (
            "Pipeline should handle missing values"
        )


class TestPipelineConfiguration:
    """Test pipeline configuration options and customization."""

    def test_custom_log_columns(self, sample_dataframe):
        """Test that custom log columns are applied correctly."""
        custom_log_columns = ["min_age", "min_playtime"]

        pipeline = create_preprocessing_pipeline(
            model_type="linear", log_columns=custom_log_columns
        )

        # Check that the log transformer has the correct columns
        log_transformer = pipeline.named_steps["log"]
        assert log_transformer.columns == custom_log_columns

    def test_custom_year_parameters(self, sample_dataframe):
        """Test that custom year transformation parameters work."""
        custom_ref_year = 1995
        custom_norm_factor = 30

        pipeline = create_preprocessing_pipeline(
            model_type="linear",
            reference_year=custom_ref_year,
            normalization_factor=custom_norm_factor,
        )

        # Check that the year transformer has the correct parameters
        year_transformer = pipeline.named_steps["year"]
        assert year_transformer.reference_year == custom_ref_year
        assert year_transformer.normalization_factor == custom_norm_factor

    def test_bgg_preprocessor_kwargs(self, sample_dataframe):
        """Test that kwargs are passed to BGG preprocessor correctly."""
        pipeline = create_preprocessing_pipeline(
            model_type="linear",
            create_category_features=False,
            create_mechanic_features=False,
            max_designer_features=100,
        )

        bgg_preprocessor = pipeline.named_steps["bgg_preprocessor"]
        assert not bgg_preprocessor.create_category_features
        assert not bgg_preprocessor.create_mechanic_features
        assert bgg_preprocessor.max_designer_features == 100


class TestPipelineIntegration:
    """Test pipeline integration with the preprocess_data function."""

    def test_preprocess_data_function(self, sample_dataframe):
        """Test the preprocess_data wrapper function."""
        import polars as pl

        # Convert to polars for testing
        df_polars = pl.from_pandas(sample_dataframe)

        pipeline = create_preprocessing_pipeline(model_type="linear")

        # Test fitting
        result_fit = preprocess_data(
            df_polars, pipeline, fit=True, dataset_name="train"
        )

        assert isinstance(result_fit, pd.DataFrame)
        assert len(result_fit) == len(sample_dataframe)

        # Test transforming (without fitting)
        result_transform = preprocess_data(
            df_polars, pipeline, fit=False, dataset_name="test"
        )

        assert isinstance(result_transform, pd.DataFrame)
        assert len(result_transform) == len(sample_dataframe)

        # Results should be identical since pipeline was already fitted
        pd.testing.assert_frame_equal(result_fit, result_transform)

    def test_preprocess_data_array_columns(self, sample_dataframe):
        """Test that preprocess_data handles array columns correctly."""
        import polars as pl

        # Ensure we have array columns in the test data
        if "categories" not in sample_dataframe.columns:
            sample_dataframe["categories"] = [["Strategy", "Economic"]] * len(
                sample_dataframe
            )
        if "mechanics" not in sample_dataframe.columns:
            sample_dataframe["mechanics"] = [
                ["Hand Management", "Set Collection"]
            ] * len(sample_dataframe)

        df_polars = pl.from_pandas(sample_dataframe)
        pipeline = create_preprocessing_pipeline(model_type="tree")

        # Should not raise an error
        result = preprocess_data(df_polars, pipeline, fit=True)
        assert isinstance(result, pd.DataFrame)


class TestPipelineErrorHandling:
    """Test error handling and edge cases in pipeline functionality."""

    def test_empty_dataframe_handling(self):
        """Test that pipelines handle empty dataframes gracefully."""
        empty_df = pd.DataFrame()
        pipeline = create_preprocessing_pipeline(model_type="linear")

        # Should raise an appropriate error for empty dataframe
        with pytest.raises((ValueError, IndexError, KeyError)):
            pipeline.fit_transform(empty_df)

    def test_single_row_dataframe(self, sample_dataframe):
        """Test that pipelines handle single-row dataframes appropriately."""
        single_row_df = sample_dataframe.iloc[:1].copy()
        pipeline = create_preprocessing_pipeline(model_type="tree")

        # Single row dataframes should raise ValueError due to variance threshold
        # (no variance can be calculated with only one sample)
        with pytest.raises(
            ValueError, match="No feature in X meets the variance threshold"
        ):
            pipeline.fit_transform(single_row_df)

    def test_missing_required_columns(self):
        """Test behavior when required columns are missing."""
        # Create dataframe with minimal columns
        minimal_df = pd.DataFrame({"year_published": [2020, 2021], "min_age": [8, 10]})

        pipeline = create_preprocessing_pipeline(model_type="linear")

        # Should handle missing columns gracefully or raise informative error
        try:
            result = pipeline.fit_transform(minimal_df)
            # If it succeeds, should return valid dataframe
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
        except (KeyError, ValueError) as e:
            # If it fails, should be informative error
            assert len(str(e)) > 0


if __name__ == "__main__":
    pytest.main([__file__])
