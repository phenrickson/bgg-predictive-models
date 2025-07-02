"""Example of using BGGPreprocessor with imputation in a scikit-learn pipeline."""
import polars as pl
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data.loader import BGGDataLoader
from src.features.sklearn_preprocessor import BGGSklearnPreprocessor
from src.features.imputation import ColumnSpecificImputer
from src.models.train_hurdle_model import load_config


def create_pipeline():
    """Create a pipeline with imputation and preprocessing."""
    # Define numeric columns that need imputation
    numeric_features = [
        'min_age',
        'min_playtime',
        'max_playtime',
        'average_weight',
        'mechanics_count'
    ]
    
    # Create column-specific imputation strategies
    imputation_strategies = {
        'min_age': {'strategy': 'median'},
        'min_playtime': {'strategy': 'median'},
        'max_playtime': {'strategy': 'median'},
        'average_weight': {'strategy': 'median'},
        'mechanics_count': {'strategy': 'constant', 'fill_value': 0},
    }
    
    # Create the preprocessor
    preprocessor = BGGSklearnPreprocessor(
        # Include specific features
        include_average_weight=True,
        create_category_mechanic_features=True,
        
        # Exclude patterns
        exclude_feature_patterns=[
            "^game_id$",
            "^name$", 
            "^year_published$",
            "^year_published_normalized$",
            "^description",
            "^image",
            "^thumbnail",
            "^complexity"
            "^users_rated",
            "^average_rating"
        ],
    )
    
    # Create the pipeline
    pipeline = Pipeline([
        # Step 1: Impute missing values using column-specific strategies
        ('imputer', ColumnSpecificImputer(imputation_strategies)),
        
        # Step 2: Apply BGG preprocessing
        ('preprocessor', preprocessor),
        
        # Step 3: Scale numeric features
        ('scaler', StandardScaler()),
        
        # Step 4: Train a classifier
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline


def create_advanced_pipeline():
    """Create a more advanced pipeline with separate preprocessing steps."""
    # Define numeric columns that need imputation
    numeric_features = [
        'min_age',
        'min_playtime',
        'max_playtime',
        'average_weight',
        'mechanics_count'
    ]
    
    # Create the preprocessor
    preprocessor = BGGSklearnPreprocessor(
        # Include specific features
        include_average_weight=True,
        create_category_mechanic_features=True,
        
        # Exclude patterns
        exclude_feature_patterns=[
            "^game_id$",
            "^name$", 
            "^year_published$",
            "^year_published_normalized$",
            "^description",
            "^image",
            "^thumbnail"
        ],
    )
    
    # Create preprocessing steps for different types of features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create the ColumnTransformer
    preprocessor_with_imputation = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bgg', preprocessor, 'passthrough')
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Create the full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor_with_imputation),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline


def main():
    """Run the example."""
    # Load data
    config = load_config()
    loader = BGGDataLoader(config)
    full_data = loader.load_training_data(end_train_year=2022, min_ratings=0)
    
    # Convert targets to pandas Series
    y = full_data.select(pl.col("users_rated") >= 25).to_series().to_pandas()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        full_data, y, test_size=0.2, random_state=42
    )
    
    print("Example 1: Simple Pipeline with Column-Specific Imputation")
    print("-------------------------------------------------------")
    # Create and fit the simple pipeline
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    print("\nExample 2: Advanced Pipeline with ColumnTransformer")
    print("------------------------------------------------")
    # Create and fit the advanced pipeline
    advanced_pipeline = create_advanced_pipeline()
    advanced_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred_advanced = advanced_pipeline.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred_advanced))


if __name__ == "__main__":
    main()
