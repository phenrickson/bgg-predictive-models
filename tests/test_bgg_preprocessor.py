import pytest
import pandas as pd
import numpy as np
from src.features.transformers import BaseBGGTransformer

@pytest.fixture
def sample_bgg_data():
    """Create a sample DataFrame for testing the BGG preprocessor."""
    return pd.DataFrame({
        'min_age': [10, 12, 15],
        'min_playtime': [30, 45, 60],
        'max_playtime': [60, 90, 120],
        'min_players': [1, 2, 3],
        'max_players': [4, 5, 6],
        'year_published': [2010, 2015, 2020],
        'average_weight': [2.5, 3.0, 3.5],
        'categories': [
            ['Strategy', 'Economic', 'Sci-Fi'], 
            ['War', 'Historical'], 
            ['Party', 'Card Game']
        ],
        'mechanics': [
            ['Worker Placement', 'Area Control'], 
            ['Dice Rolling', 'Deck Building'], 
            ['Set Collection', 'Hand Management']
        ],
        'designers': [
            ['Designer A', 'Designer B'], 
            ['Designer C'], 
            ['Designer D', 'Designer E']
        ],
        'artists': [
            ['Artist 1'], 
            ['Artist 2', 'Artist 3'], 
            ['Artist 4']
        ],
        'publishers': [
            ['Hasbro', 'Days of Wonder'], 
            ['Asmodee'], 
            ['Fantasy Flight Games']
        ],
        'families': [
            ['Series: 18xx', 'Country: USA'], 
            ['Players: Two-Player Only Games'], 
            ['Mythology: Greek']
        ]
    })

def test_feature_generation_flags(sample_bgg_data):
    """Test that feature generation flags work correctly."""
    # Test disabling all array features and player dummies
    transformer = BaseBGGTransformer(
        create_category_features=False,
        create_mechanic_features=False,
        create_designer_features=False,
        create_artist_features=False,
        create_publisher_features=False,
        create_family_features=False,
        create_player_dummies=False
    )
    
    # Fit and transform
    transformed_data = transformer.fit_transform(sample_bgg_data)
    
    # Check that only base features and preserved columns are present
    expected_columns = [
        'year_published', 
        'mechanics_count', 
        'min_age', 
        'min_playtime', 
        'max_playtime'
    ]
    assert set(transformed_data.columns) == set(expected_columns)

def test_individual_feature_flags(sample_bgg_data):
    """Test disabling individual feature types."""
    # Test disabling specific feature types
    test_cases = [
        {
            'name': 'categories_disabled',
            'flags': {'create_category_features': False},
            'excluded_prefix': 'category_'
        },
        {
            'name': 'mechanics_disabled',
            'flags': {'create_mechanic_features': False},
            'excluded_prefix': 'mechanic_'
        },
        {
            'name': 'designers_disabled',
            'flags': {'create_designer_features': False},
            'excluded_prefix': 'designer_'
        },
        {
            'name': 'artists_disabled',
            'flags': {'create_artist_features': False},
            'excluded_prefix': 'artist_'
        },
        {
            'name': 'publishers_disabled',
            'flags': {'create_publisher_features': False},
            'excluded_prefix': 'publisher_'
        },
        {
            'name': 'families_disabled',
            'flags': {'create_family_features': False},
            'excluded_prefix': 'family_'
        }
    ]
    
    for case in test_cases:
        # Create transformer with specific feature disabled
        transformer = BaseBGGTransformer(**case['flags'])
        
        # Fit and transform
        transformed_data = transformer.fit_transform(sample_bgg_data)
        
        # Check that no columns start with the excluded prefix
        excluded_columns = [
            col for col in transformed_data.columns 
            if col.startswith(case['excluded_prefix'])
        ]
        assert len(excluded_columns) == 0, f"Failed for {case['name']}"

def test_feature_names_generation(sample_bgg_data):
    """Test that feature names are generated correctly based on flags."""
    # Test with all features enabled
    transformer_all = BaseBGGTransformer()
    transformer_all.fit(sample_bgg_data)
    all_feature_names = transformer_all.get_feature_names_out()
    
    # Test with some features disabled
    transformer_partial = BaseBGGTransformer(
        create_category_features=False,
        create_family_features=False
    )
    transformer_partial.fit(sample_bgg_data)
    partial_feature_names = transformer_partial.get_feature_names_out()
    
    # Verify that disabled feature types are not in feature names
    assert not any(name.startswith('category_') for name in partial_feature_names)
    assert not any(name.startswith('family_') for name in partial_feature_names)

def test_feature_flag_with_min_max_features(sample_bgg_data):
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
        transformed_data = transformer.fit_transform(sample_bgg_data)
        
        # Check that no columns start with the excluded prefix
        excluded_columns = [
            col for col in transformed_data.columns 
            if col.startswith(case['excluded_prefix'])
        ]
        assert len(excluded_columns) == 0, f"Failed for {case['name']}"
        
        # Verify that the feature names do not include the excluded prefix
        transformer.fit(sample_bgg_data)
        feature_names = transformer.get_feature_names_out()
        assert not any(name.startswith(case['excluded_prefix']) for name in feature_names), \
            f"Feature names incorrectly generated for {case['name']}"
