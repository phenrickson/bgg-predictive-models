"""Example of using BGGPreprocessor with scikit-learn pipeline and imputation."""
import polars as pl
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.data.loader import BGGDataLoader
from src.features.preprocessor import BGGPreprocessor
from src.features.imputation import BGGImputer, ColumnSpecificImputer
from src.models.train_hurdle_model import load_config


class PolarsToDataFrame:
    """Convert a Polars DataFrame to a pandas DataFrame.
    
    This is a simple transformer that converts a Polars DataFrame to a pandas DataFrame,
    which is required for scikit-learn pipelines.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pl.DataFrame):
            return X.to_pandas()
        return X
    
    def fit_transform(self, X, y=None):
        return self.transform(X)


def main():
    """Run the example pipeline."""
    # Load data
    config = load_config()
    loader = BGGDataLoader(config)
    
    # Set end year
    end_train_year = 2022
    
    # Load training data
    print("Loading data...")
    full_data = loader.load_training_data(end_train_year=end_train_year, min_ratings=0)
    
    # Create preprocessor
    print("Creating preprocessor...")
    preprocessor = BGGPreprocessor(
        # Include average_weight as a feature
        include_average_weight=True,
        
        # Basic configuration
        transform_year=True,
        reference_year=2000,
        normalization_factor=25,
        max_player_count=10,
        
        # Feature selection
        include_base_numeric=True,
        include_player_count=True,
        include_categories=True,
        include_mechanics=True,
        
        # Feature generation
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
    
    # Process the data
    print("Processing data...")
    features, targets = preprocessor.fit_transform(full_data)
    
    # Convert to pandas DataFrame for scikit-learn
    print("Converting to pandas DataFrame...")
    features_pd = features.to_pandas()
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_pd, 
        targets["hurdle"].to_pandas(), 
        test_size=0.2, 
        random_state=42
    )
    
    # Create column-specific imputation strategies
    print("Creating imputation strategies...")
    imputation_strategies = {
        'min_age': {'strategy': 'median'},
        'min_playtime': {'strategy': 'median'},
        'max_playtime': {'strategy': 'median'},
        'average_weight': {'strategy': 'median'},
        'mechanics_count': {'strategy': 'median'},
    }
    
    # Create the pipeline
    print("Creating pipeline...")
    pipeline = Pipeline([
        # Step 1: Impute missing values
        ('imputer', ColumnSpecificImputer(imputation_strategies)),
        
        # Step 2: Scale numeric features
        ('scaler', StandardScaler()),
        
        # Step 3: Train a classifier
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    print("Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate the pipeline
    print("Evaluating pipeline...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("Feature importance:")
    feature_importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = X_train.columns
    
    # Sort feature importances in descending order
    indices = np.argsort(feature_importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    for i, idx in enumerate(indices[:20]):  # Print top 20 features
        print(f"{i+1}. {feature_names[idx]} ({feature_importances[idx]:.4f})")
    
    print("Pipeline example completed successfully!")


if __name__ == "__main__":
    main()
