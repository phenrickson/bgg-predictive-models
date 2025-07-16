import pytest
import pandas as pd
import numpy as np
import polars as pl
from src.features.transformers import BaseBGGTransformer

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing transformer features."""
    return pd.read_parquet('data/raw/game_features.parquet').sample(n=10000, random_state=42)

def test_mechanics_count(sample_dataframe):
    """Test mechanics count feature."""
    transformer = BaseBGGTransformer(include_base_numeric=True)
    transformed = transformer.fit_transform(sample_dataframe)
    
    assert 'mechanics_count' in transformed.columns
    assert transformed['mechanics_count'].notna().all()

def test_categories_count(sample_dataframe):
    """Test categories count feature."""
    transformer = BaseBGGTransformer(include_base_numeric=True)
    transformed = transformer.fit_transform(sample_dataframe)
    
    assert 'categories_count' in transformed.columns
    assert transformed['categories_count'].notna().all()

def test_time_per_player(sample_dataframe):
    """Test time per player feature."""
    transformer = BaseBGGTransformer(include_base_numeric=True)
    transformed = transformer.fit_transform(sample_dataframe)
    
    assert 'time_per_player' in transformed.columns
    
    # Verify time per player calculation
    for idx, row in sample_dataframe.iterrows():
        # Expect NaN if max_playtime is 0 or max_players is 0
        expected = np.nan if row['max_playtime'] == 0 or row['max_players'] == 0 else row['max_playtime'] / row['max_players']
        actual = transformed.loc[idx, 'time_per_player']
        
        # Use more flexible comparison
        if pd.isna(expected):
            assert pd.isna(actual), f"Expected NaN for index {idx}"
        else:
            assert np.isclose(actual, expected, equal_nan=True), f"Mismatch for index {idx}: expected {expected}, got {actual}"

def test_description_word_count(sample_dataframe):
    """Test description word count feature."""
    transformer = BaseBGGTransformer(include_base_numeric=True)
    transformed = transformer.fit_transform(sample_dataframe)
    
    assert 'description_word_count' in transformed.columns
    
    # Verify word count calculation
    for idx, row in sample_dataframe.iterrows():
        expected_count = len(str(row['description']).split()) if pd.notna(row['description']) else 0
        assert transformed.loc[idx, 'description_word_count'] == expected_count

def test_feature_names(sample_dataframe):
    """Test that all new features are in feature names."""
    transformer = BaseBGGTransformer(include_base_numeric=True)
    transformer.fit(sample_dataframe)
    
    feature_names = transformer.get_feature_names_out()
    
    expected_features = [
        'mechanics_count', 
        'categories_count', 
        'time_per_player', 
        'description_word_count',
        'min_age', 
        'min_playtime', 
        'max_playtime'
    ]
    
    for feature in expected_features:
        assert feature in feature_names
        
def test_feature_names_generation(sample_dataframe):
    """Test that feature names are generated correctly based on flags."""
    # Test with all features enabled
    transformer_all = BaseBGGTransformer()
    transformer_all.fit(sample_dataframe)
    all_feature_names = transformer_all.get_feature_names_out()
    
    # Test with some features disabled
    transformer_partial = BaseBGGTransformer(
        create_category_features=False,
        create_family_features=False
    )
    transformer_partial.fit(sample_dataframe)
    partial_feature_names = transformer_partial.get_feature_names_out()
    
    # Verify that disabled feature types are not in feature names
    assert not any(name.startswith('category_') for name in partial_feature_names)
    assert not any(name.startswith('family_') for name in partial_feature_names)
    
def test_feature_generation_flags(sample_dataframe):
    """Test that feature generation flags work correctly."""
    # Test disabling all array features and player dummies
    transformer = BaseBGGTransformer(
        create_category_features=False,
        create_mechanic_features=False,
        create_designer_features=False,
        create_artist_features=False,
        create_publisher_features=False,
        create_family_features=False,
        create_player_dummies=False,
        include_base_numeric=False
    )
    
    # Fit and transform
    transformed_data = transformer.fit_transform(sample_dataframe)
    
    # Check that only preserved columns are present
    expected_columns = [
        'year_published'
    ]
    assert set(transformed_data.columns) == set(expected_columns)

def test_feature_flag_with_min_max_features(sample_dataframe):
    """Test that feature generation flags take precedence over min/max feature settings."""
    # Test scenarios where create_*_features is False, but min/max features are set
    test_cases = [
        {
            'name': 'families_disabled_with_min_max',
            'flags': {
                'create_family_features': False,
                'family_min_freq': 10,
                'max_family_features': 100
            },
            'excluded_prefix': 'family_'
        },
        {
            'name': 'categories_disabled_with_min_max',
            'flags': {
                'create_category_features': False,
                'category_min_freq': 5,
                'max_category_features': 50
            },
            'excluded_prefix': 'category_'
        },
        {
            'name': 'mechanics_disabled_with_min_max',
            'flags': {
                'create_mechanic_features': False,
                'mechanic_min_freq': 8,
                'max_mechanic_features': 75
            },
            'excluded_prefix': 'mechanic_'
        }
    ]
    
    for case in test_cases:
        # Create transformer with feature disabled but min/max set
        transformer = BaseBGGTransformer(**case['flags'])
        
        # Fit and transform
        transformed_data = transformer.fit_transform(sample_dataframe)
        
        # Check that no columns start with the excluded prefix
        excluded_columns = [
            col for col in transformed_data.columns 
            if col.startswith(case['excluded_prefix'])
        ]
        assert len(excluded_columns) == 0, f"Failed for {case['name']}"
        
        # Verify that the feature names do not include the excluded prefix
        transformer.fit(sample_dataframe)
        feature_names = transformer.get_feature_names_out()
        assert not any(name.startswith(case['excluded_prefix']) for name in feature_names), \
            f"Feature names incorrectly generated for {case['name']}"

def test_handle_missing_values(sample_dataframe):
    """Test handling of missing or zero values."""
    # Create a copy of the sample DataFrame to modify
    df_with_missing = sample_dataframe.copy()
    
    # Simulate some missing/zero values
    df_with_missing.loc[df_with_missing.index[0], 'max_playtime'] = 0
    df_with_missing.loc[df_with_missing.index[1], 'description'] = np.nan
    df_with_missing.loc[df_with_missing.index[2], 'min_age'] = 0
    
    transformer = BaseBGGTransformer(include_base_numeric=True)
    transformed = transformer.fit_transform(df_with_missing)
    
    # Check that NaN/zero values are handled gracefully
    assert pd.isna(transformed.loc[df_with_missing.index[0], 'time_per_player'])
    assert transformed.loc[df_with_missing.index[1], 'description_word_count'] == 0
    
    # Specifically check that zeros in min_age are converted to NaN
    assert pd.isna(transformed.loc[df_with_missing.index[2], 'min_age']), \
        "Zero values in min_age should be converted to NaN"

def test_categorical_feature_generation(sample_dataframe):
    """Test generation of categorical features under different configurations."""
    # Test with all categorical features enabled
    transformer_all = BaseBGGTransformer(
        create_designer_features=True,
        create_artist_features=True,
        create_publisher_features=True
    )
    transformed_all = transformer_all.fit_transform(sample_dataframe)
    
    # Check that designer features are generated
    designer_columns = [col for col in transformed_all.columns if col.startswith('designer_')]
    assert len(designer_columns) > 0, "No designer features generated when create_designer_features=True"
    
    # Check that artist features are generated
    artist_columns = [col for col in transformed_all.columns if col.startswith('artist_')]
    assert len(artist_columns) > 0, "No artist features generated when create_artist_features=True"
    
    # Check that publisher features are generated
    publisher_columns = [col for col in transformed_all.columns if col.startswith('publisher_')]
    assert len(publisher_columns) > 0, "No publisher features generated when create_publisher_features=True"
    
    # Test with categorical features disabled
    transformer_none = BaseBGGTransformer(
        create_designer_features=False,
        create_artist_features=False,
        create_publisher_features=False
    )
    transformed_none = transformer_none.fit_transform(sample_dataframe)
    
    # Check that no categorical feature columns are present
    designer_columns = [col for col in transformed_none.columns if col.startswith('designer_')]
    artist_columns = [col for col in transformed_none.columns if col.startswith('artist_')]
    publisher_columns = [col for col in transformed_none.columns if col.startswith('publisher_')]
    
    assert len(designer_columns) == 0, "Designer features present when create_designer_features=False"
    assert len(artist_columns) == 0, "Artist features present when create_artist_features=False"
    assert len(publisher_columns) == 0, "Publisher features present when create_publisher_features=False"
    
    # Test feature names generation
    transformer_all.fit(sample_dataframe)
    feature_names = transformer_all.get_feature_names_out()
    
    # Verify that feature names include categorical features when enabled
    assert any(name.startswith('designer_') for name in feature_names), "No designer feature names generated"
    assert any(name.startswith('artist_') for name in feature_names), "No artist feature names generated"
    assert any(name.startswith('publisher_') for name in feature_names), "No publisher feature names generated"
