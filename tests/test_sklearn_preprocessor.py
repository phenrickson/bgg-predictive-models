import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.features.sklearn_preprocessor import BGGSklearnPreprocessor, CorrelationFilter
from src.features.transformers import LogTransformer

@pytest.fixture(scope="module")
def raw_game_features():
    """Load raw game features data for testing."""
    return pd.read_parquet("data/raw/game_features.parquet")

def test_preprocessor_basic_functionality(raw_game_features):
    """Test basic functionality of the BGGSklearnPreprocessor."""
    # Create preprocessor with default settings
    preprocessor = BGGSklearnPreprocessor()
    
    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(raw_game_features)
    
    # Check basic properties of transformed data
    assert isinstance(X_transformed, pd.DataFrame), "Transformed data should be a DataFrame"
    assert len(X_transformed) == len(raw_game_features), "Transformed data should have same number of rows"
    
    # Check feature names
    feature_names = preprocessor.get_feature_names()
    assert len(feature_names) > 0, "Feature names should be generated"
    assert len(X_transformed.columns) == len(feature_names), "Transformed data columns should match feature names"
    
    # Check for no NaN values
    assert not X_transformed.isna().any().any(), "Transformed data should not contain NaN values"

def test_preprocessor_feature_generation_options(raw_game_features):
    """Test different feature generation options."""
    # Test with various feature generation flags
    test_configs = [
        {
            "name": "No features",
            "config": {
                "create_category_features": False,
                "create_mechanic_features": False,
                "create_designer_features": False,
                "create_artist_features": False,
                "create_publisher_features": False,
                "create_family_features": False,
                "create_player_dummies": False,
                "include_base_numeric": False,
            }
        },
        {
            "name": "Minimal features",
            "config": {
                "create_category_features": True,
                "create_mechanic_features": True,
                "create_player_dummies": True,
                "include_base_numeric": True,
            }
        }
    ]
    
    for test_case in test_configs:
        preprocessor = BGGSklearnPreprocessor(**test_case["config"])
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(raw_game_features)
        
        # Check basic properties
        assert isinstance(X_transformed, pd.DataFrame), f"{test_case['name']}: Transformed data should be a DataFrame"
        assert len(X_transformed) == len(raw_game_features), f"{test_case['name']}: Transformed data should have same number of rows"
        
        # Verify feature names match configuration
        feature_names = preprocessor.get_feature_names()
        assert len(feature_names) == len(X_transformed.columns), f"{test_case['name']}: Feature names should match transformed columns"

def test_preprocessor_pipeline_integration(raw_game_features):
    """Test preprocessor integration with sklearn Pipeline including CorrelationFilter."""
    # Create preprocessor
    preprocessor = BGGSklearnPreprocessor()
    
    # Create a pipeline with preprocessor, correlation filter, and standard scaler
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('correlation_filter', CorrelationFilter(threshold=0.95)),
        ('scaler', StandardScaler())
    ])
    
    # Fit and transform
    X_transformed = pipeline.fit_transform(raw_game_features)
    
    # Check basic properties
    assert isinstance(X_transformed, np.ndarray), "Pipeline transform should return numpy array"
    assert X_transformed.shape[0] == len(raw_game_features), "Transformed data should have same number of rows"
    
    # Check scaling (standard scaler should result in mean close to 0 and std close to 1 for non-constant features)
    means = X_transformed.mean(axis=0)
    stds = X_transformed.std(axis=0)
    
    # Check means are close to 0
    assert np.allclose(means, 0, atol=1e-1), "Scaled data should have mean close to 0"
    
    # Check stds are close to 1 for non-constant features
    non_constant_features = stds > 0
    if non_constant_features.any():
        assert np.allclose(stds[non_constant_features], 1, atol=1e-1), "Scaled data should have std close to 1 for non-constant features"
    
    # Check that the correlation filter reduces features
    preprocessor_features = preprocessor.get_feature_names()
    assert X_transformed.shape[1] < len(preprocessor_features), "Correlation filter should reduce number of features"

def test_preprocessor_year_transformation(raw_game_features):
    """Test year transformation features."""
    # Create preprocessor with year transformation
    preprocessor = BGGSklearnPreprocessor(
        transform_year=True, 
        reference_year=2000, 
        normalization_factor=25
    )
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(raw_game_features)
    
    # Check year transformation features
    year_transform_cols = [
        "year_published_centered", 
        "year_published_normalized", 
        "year_published_transformed"
    ]
    
    for col in year_transform_cols:
        assert col in X_transformed.columns, f"Year transformation column {col} should be present"
        
        # Check basic statistical properties
        col_data = X_transformed[col]
        assert not col_data.isna().any(), f"No NaN values should be in {col}"
        assert col_data.dtype in [np.float64, np.float32], f"{col} should be a float type"

def test_preprocessor_zero_variance_features(raw_game_features):
    """Test for presence of zero variance features."""
    # Create preprocessor with default settings
    preprocessor = BGGSklearnPreprocessor()
    
    # build pipeline with
    variance_pipeline = Pipeline([
        'preprocessor', preprocessor,
        'variance', VarianceThreshold(threshold = 0.1)
    ])
    
    # Fit and transform with regular
    X_transformed = preprocessor.fit_transform(raw_game_features)
    
    # fit and transform with variance pipeline
    X_filtered = variance_pipeline.fit_transform(raw_game_features)
    
    # Calculate variances for each feature
    variances = X_transformed.var()
    zero_var_features = variances[variances == 0].index.tolist()
    
    # Log zero variance features and their percentage
    total_features = len(X_transformed.columns)
    zero_var_count = len(zero_var_features)
    zero_var_percentage = (zero_var_count / total_features) * 100
    
    # Use sys.stderr to ensure output is displayed
    import sys
    sys.stderr.write(f"\nFeature variance analysis:\n")
    sys.stderr.write(f"Total features: {total_features}\n")
    sys.stderr.write(f"Zero variance features: {zero_var_count} ({zero_var_percentage:.1f}%)\n")
    
    if zero_var_features:
        sys.stderr.write("\nZero variance features found:\n")
        for feature in zero_var_features:
            sys.stderr.write(f"  - {feature}\n")
    
    # We allow zero variance features but want to be aware of them
    assert len(zero_var_features) < len(X_transformed.columns), "All features have zero variance"

def test_preprocessor_missing_values(raw_game_features):
    """Test handling of missing values."""
    # Create preprocessor with missing value handling
    preprocessor = BGGSklearnPreprocessor(
        handle_missing_values=True
    )
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(raw_game_features)
    
    # Check no NaN values
    assert not X_transformed.isna().any().any(), "Transformed data should not contain NaN values"
    
    # Check specific numeric columns
    numeric_cols = ["min_age", "min_playtime", "max_playtime"]
    for col in numeric_cols:
        if col in X_transformed.columns:
            assert not X_transformed[col].isna().any(), f"No NaN values should be in {col}"

def test_preprocessor_feature_selection_techniques(raw_game_features):
    """Test various scikit-learn feature selection techniques."""
    import numpy as np
    
    # Create preprocessor with more restrictive feature generation
    preprocessor = BGGSklearnPreprocessor()
    
    # Get preprocessor features
    X_transformed = preprocessor.fit_transform(raw_game_features)
    preprocessor_features = preprocessor.get_feature_names()
    
    # Compute variances of original transformed features
    original_variances = X_transformed.var()
    
    # Print low variance features
    low_variance_features = original_variances[original_variances <= 0.01]
    
    print("\nLow Variance Features (variance <= 0.1):")
    for feature, variance in low_variance_features.items():
        print(f"  {feature}: variance = {variance:.4f}")
    
    print("\nChecking Low Variance Features in Preprocessor Features:")
    for feature in low_variance_features.index:
        assert feature in preprocessor_features, f"Low variance feature {feature} not found in preprocessor features"
    
    # Demonstrate variance thresholding after scaling
    variance_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_selector', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler())
    ])
    
    # Fit and transform
    X_variance_selected = variance_pipeline.fit_transform(raw_game_features)
    
    print("\nFeature Selection Comparison:")
    print(f"Preprocessor features: {len(preprocessor_features)}")
    print(f"Features after variance selection: {X_variance_selected.shape[1]}")
    
    # Assertions
    assert X_variance_selected.shape[1] > 0, "Variance threshold should not remove all features"
    assert X_variance_selected.shape[1] < len(preprocessor_features), "Some features should be removed"
    assert len(low_variance_features) > 0, "There should be some low variance features"

def test_preprocessor_feature_correlations(raw_game_features):
    """Test feature correlations in the transformed dataset."""
    # Create preprocessor with default settings
    preprocessor = BGGSklearnPreprocessor()
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(raw_game_features)
    
    # Compute correlation matrix
    corr_matrix = X_transformed.corr()
    
    # Print correlation matrix for debugging
    print("\nFeature Correlation Matrix:")
    print(corr_matrix)
    
    # Basic correlation checks
    # 1. Check for extreme correlations (absolute value > 0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_value
                ))
    
    # Print high correlation pairs
    if high_corr_pairs:
        print("\nHigh Correlation Pairs:")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.2f}")
    
    # Assertions
    # 1. Not too many highly correlated features
    assert len(high_corr_pairs) < len(X_transformed.columns) * 0.1, \
        f"Too many highly correlated feature pairs: {high_corr_pairs}"
    
    # 2. Check for problematic perfect correlations
    perfect_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value == 1.0:
                perfect_correlations.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_value
                ))
    
    # Print and handle perfect correlations
    if perfect_correlations:
        print("\nPerfect Correlation Pairs (Potential Multicollinearity):")
        for pair in perfect_correlations:
            print(f"  {pair[0]} - {pair[1]}: {pair[2]:.2f}")
            
            # Special handling for year transformation features
            if set(pair[:2]) == {"year_published_centered", "year_published_normalized"}:
                print("  WARNING: These year transformation features are perfectly correlated.")
                print("  Recommendation: Consider using only one of these features in your model.")
                print("  Potential solutions:")
                print("  1. Remove one of the features")
                print("  2. Create a single, more informative year feature")
                print("  3. Use feature selection techniques to choose the most relevant feature")
    
    # Assert that we don't have too many perfect correlations
    assert len(perfect_correlations) <= 2, \
        f"Too many perfectly correlated feature pairs: {perfect_correlations}"
    
    # 3. Mechanics count should have some correlation with other features
    mechanics_count_correlations = corr_matrix["mechanics_count"].drop("mechanics_count")
    significant_correlations = mechanics_count_correlations[
        abs(mechanics_count_correlations) > 0.2
    ]
    
    print("\nSignificant Mechanics Count Correlations:")
    print(significant_correlations)
    
    # Mechanics count should correlate with at least a few features
    assert len(significant_correlations) > 0, \
        "Mechanics count should have meaningful correlations with other features"

def test_preprocessor_imputation_pipeline(raw_game_features):
    """Test preprocessor with imputation and log transformation pipeline."""
    # Create the preprocessor without missing value handling
    preprocessor = BGGSklearnPreprocessor(handle_missing_values=False)
    
    # Specify numeric columns to impute and transform
    numeric_columns = ['min_age', 'min_playtime', 'max_playtime']
    
    # Create column transformer for imputation
    imputer = ColumnTransformer([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True), numeric_columns)
    ], remainder='passthrough')
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('imputer', imputer),
        ('log_transform', LogTransformer(columns=numeric_columns)),
        ('variance_selector', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler())
    ])
    
    # Fit and transform the data
    X_transformed = pipeline.fit_transform(raw_game_features)
    
    # Check basic properties
    assert isinstance(X_transformed, np.ndarray), "Pipeline transform should return numpy array"
    assert X_transformed.shape[0] == len(raw_game_features), "Transformed data should have same number of rows"
    
    # Check that the variance selector and imputation reduce features
    preprocessor_features = preprocessor.get_feature_names()
    assert X_transformed.shape[1] < len(preprocessor_features), "Pipeline steps should reduce number of features"
    
    # Look at the column names from the fit pipeline
    print(pipeline.named_steps['imputer'].get_feature_names_out())
    
    # Check scaling (standard scaler should result in mean close to 0 and std close to 1 for non-constant features)
    means = X_transformed.mean(axis=0)
    stds = X_transformed.std(axis=0)
    
    # Check means are close to 0
    assert np.allclose(means, 0, atol=1e-1), "Scaled data should have mean close to 0"
    
    # Check stds are close to 1 for non-constant features
    non_constant_features = stds > 0
    if non_constant_features.any():
        assert np.allclose(stds[non_constant_features], 1, atol=1e-1), "Scaled data should have std close to 1 for non-constant features"
