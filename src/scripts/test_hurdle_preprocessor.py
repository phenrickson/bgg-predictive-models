"""
Script to test BGGSklearnPreprocessor for Hurdle Model
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Project imports
from src.data.loader import BGGDataLoader
from src.data.config import load_config
from src.features.sklearn_preprocessor import BGGSklearnPreprocessor

def main():
    # Load configuration and data
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Load training data
    full_data = loader.load_training_data(end_train_year=2022, min_ratings=0)
    
    # Convert to pandas
    df_pandas = full_data.to_pandas()
    
    # Demonstrate different year transformation types
    year_transform_types = ['centered', 'normalized', 'transformed']
    
    for transform_type in year_transform_types:
        print(f"\n--- Year Transform Type: {transform_type} ---")
        
        # Configure preprocessor with specific year transformation
        preprocessor = BGGSklearnPreprocessor(
            # Feature generation parameters
            max_player_count=10,
            
            # Logging
            verbose=True,
            
            # Year transformation
            year_transform_type=transform_type,
            
            # Array feature parameters
            category_min_freq=5,
            mechanic_min_freq=5,
            designer_min_freq=10,
            artist_min_freq=10,
            publisher_min_freq=5,
            family_min_freq=5,
            
            max_category_features=500,
            max_mechanic_features=500,
            max_designer_features=250,
            max_artist_features=250,
            max_publisher_features=250,
            max_family_features=250,
            
            # Feature generation flags
            handle_missing_values=True,
            transform_year=True,
            create_player_dummies=True,
            create_category_features=True,
            create_mechanic_features=True,
            create_designer_features=True,
            create_artist_features=True,
            create_publisher_features=True,
            create_family_features=True,
            
            # Missingness features
            create_missingness_features=True,
            
            # Base numeric features
            include_base_numeric=True
        )
        
        # Fit the preprocessor
        preprocessor.fit(df_pandas)
        
        # Transform the data
        X_transformed = preprocessor.transform(df_pandas)
        
        # Detailed feature exploration
        print(f"\nPreprocessor Feature Overview for {transform_type} transformation:")
        print(f"Total Features: {len(X_transformed.columns)}")
        
        # Print year transformation column details
        year_columns = [col for col in X_transformed.columns if 'year_published' in col]
        print("\nYear Transformation Column:")
        for col in year_columns:
            print(f"  {col}:")
            print(f"    Mean: {X_transformed[col].values.mean():.4f}")
            print(f"    Std Dev: {X_transformed[col].values.std():.4f}")
    
    # Numeric and Missingness Features
    numeric_features = [col for col in X_transformed.columns if col in ['min_age', 'min_playtime', 'max_playtime', 'year_published_centered', 'year_published_normalized', 'year_published_transformed']]
    missingness_features = [col for col in X_transformed.columns if 'missing' in col]
    
    print("\nNumeric Features:")
    for feature in numeric_features:
        print(f"  {feature}:")
        print(f"    Mean: {X_transformed[feature].mean():.4f}")
        print(f"    Std Dev: {X_transformed[feature].std():.4f}")
    
    print("\nMissingness Features:")
    unique_missingness_features = sorted(set(missingness_features))
    for feature in unique_missingness_features:
        # Skip zero-variance features
        if X_transformed[feature].values.std() > 0:
            print(f"  {feature}:")
            print(f"    Proportion Missing: {X_transformed[feature].values.mean():.4f}")
            print(f"    Total Missing: {(X_transformed[feature].values == 1).sum()}")
            print(f"    Total Records: {len(X_transformed)}")
    
    # Player Count Dummy Features
    player_features = [col for col in X_transformed.columns if col.startswith('player_count_')]
    print("\nPlayer Count Features:")
    for feature in player_features:
        print(f"  {feature}:")
        print(f"    Proportion: {X_transformed[feature].mean():.4f}")
    
    # Array Features (first 10 of each type)
    array_feature_types = {
        'category': 'category_',
        'mechanic': 'mechanic_',
        'designer': 'designer_',
        'artist': 'artist_',
        'publisher': 'publisher_',
        'family': 'family_'
    }
    
    print("\nArray Features (first 10 of each type):")
    for feature_type, prefix in array_feature_types.items():
        array_features = [col for col in X_transformed.columns if col.startswith(prefix)][:10]
        print(f"  {feature_type.capitalize()} Features:")
        for feature in array_features:
            print(f"    {feature}: {X_transformed[feature].mean():.4f}")
    
    # Optional: Save transformed data for further inspection
    X_transformed.to_csv('preprocessed_data.csv', index=False)
    print("\nFull preprocessed data saved to 'preprocessed_data.csv'")

if __name__ == "__main__":
    main()
