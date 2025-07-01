"""Scikit-learn compatible wrapper for BGG preprocessor."""
from typing import List, Optional, Dict, Any
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.preprocessor import BGGPreprocessor


class BGGSklearnPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible wrapper for BGGPreprocessor.
    
    This class wraps the BGGPreprocessor to make it compatible with scikit-learn's
    transformer interface, allowing it to be used in scikit-learn pipelines.
    
    Parameters
    ----------
    max_player_count : int, default=10
        Maximum player count for player count dummy variables.
    
    reference_year : int, default=2000
        Reference year for year transformations.
    
    normalization_factor : int, default=25
        Normalization factor for year transformations.
    
    category_min_freq : int, default=10
        Minimum frequency for category features.
    
    mechanic_min_freq : int, default=10
        Minimum frequency for mechanic features.
    
    designer_min_freq : int, default=20
        Minimum frequency for designer features.
    
    artist_min_freq : int, default=20
        Minimum frequency for artist features.
    
    publisher_min_freq : int, default=15
        Minimum frequency for publisher features.
    
    family_min_freq : int, default=10
        Minimum frequency for family features.
    
    max_category_features : int, default=500
        Maximum number of category features.
    
    max_mechanic_features : int, default=500
        Maximum number of mechanic features.
    
    max_designer_features : int, default=250
        Maximum number of designer features.
    
    max_artist_features : int, default=250
        Maximum number of artist features.
    
    max_publisher_features : int, default=250
        Maximum number of publisher features.
    
    max_family_features : int, default=250
        Maximum number of family features.
    
    handle_missing_values : bool, default=True
        Whether to handle missing values.
    
    transform_year : bool, default=True
        Whether to transform year features.
    
    create_player_dummies : bool, default=True
        Whether to create player count dummy variables.
    
    create_category_mechanic_features : bool, default=True
        Whether to create category and mechanic features.
    
    create_designer_artist_features : bool, default=True
        Whether to create designer and artist features.
    
    create_publisher_features : bool, default=True
        Whether to create publisher features.
    
    create_family_features : bool, default=True
        Whether to create family features.
    
    include_base_numeric : bool, default=True
        Whether to include base numeric features.
    
    include_average_weight : bool, default=False
        Whether to include average_weight as a feature.
    
    include_player_count : bool, default=True
        Whether to include player count features.
    
    include_categories : bool, default=True
        Whether to include category features.
    
    include_mechanics : bool, default=True
        Whether to include mechanic features.
    
    include_designers : bool, default=True
        Whether to include designer features.
    
    include_artists : bool, default=True
        Whether to include artist features.
    
    include_publishers : bool, default=True
        Whether to include publisher features.
    
    include_families : bool, default=True
        Whether to include family features.
    
    custom_feature_patterns : List[str], optional
        List of custom feature patterns to include.
    
    exclude_feature_patterns : List[str], optional
        List of feature patterns to exclude.
    
    always_include_columns : List[str], optional
        List of columns to always include.
    
    always_exclude_columns : List[str], optional
        List of columns to always exclude.
    """
    
    def __init__(
        self,
        # Feature generation parameters
        max_player_count: int = 10,
        
        # Year transformation parameters
        reference_year: int = 2000,
        normalization_factor: int = 25,
        
        # Array feature parameters
        category_min_freq: int = 10,
        mechanic_min_freq: int = 10,
        designer_min_freq: int = 20,
        artist_min_freq: int = 20,
        publisher_min_freq: int = 15,
        family_min_freq: int = 10,
        max_category_features: int = 500,
        max_mechanic_features: int = 500,
        max_designer_features: int = 250,
        max_artist_features: int = 250,
        max_publisher_features: int = 250,
        max_family_features: int = 250,
        
        # Feature generation flags
        handle_missing_values: bool = True,
        transform_year: bool = True,
        create_player_dummies: bool = True,
        create_category_mechanic_features: bool = True,
        create_designer_artist_features: bool = True,
        create_publisher_features: bool = True,
        create_family_features: bool = True,
        
        # Feature selection parameters
        include_base_numeric: bool = True,
        include_average_weight: bool = False,
        include_player_count: bool = True,
        include_categories: bool = True,
        include_mechanics: bool = True,
        include_designers: bool = True,
        include_artists: bool = True,
        include_publishers: bool = True,
        include_families: bool = True,
        custom_feature_patterns: Optional[List[str]] = None,
        exclude_feature_patterns: Optional[List[str]] = [
            "^game_id$",
            "^name$", 
            "^year_published$",
            "^year_published_transformed",
            "^year_published_normalized$",
            "^description",
            "^image",
            "^thumbnail"
        ],
        always_include_columns: Optional[List[str]] = None,
        always_exclude_columns: Optional[List[str]] = ["game_id", "name", "year_published"],
    ):
        """Initialize the preprocessor."""
        self.max_player_count = max_player_count
        self.reference_year = reference_year
        self.normalization_factor = normalization_factor
        self.category_min_freq = category_min_freq
        self.mechanic_min_freq = mechanic_min_freq
        self.designer_min_freq = designer_min_freq
        self.artist_min_freq = artist_min_freq
        self.publisher_min_freq = publisher_min_freq
        self.family_min_freq = family_min_freq
        self.max_category_features = max_category_features
        self.max_mechanic_features = max_mechanic_features
        self.max_designer_features = max_designer_features
        self.max_artist_features = max_artist_features
        self.max_publisher_features = max_publisher_features
        self.max_family_features = max_family_features
        self.handle_missing_values = handle_missing_values
        self.transform_year = transform_year
        self.create_player_dummies = create_player_dummies
        self.create_category_mechanic_features = create_category_mechanic_features
        self.create_designer_artist_features = create_designer_artist_features
        self.create_publisher_features = create_publisher_features
        self.create_family_features = create_family_features
        self.include_base_numeric = include_base_numeric
        self.include_average_weight = include_average_weight
        self.include_player_count = include_player_count
        self.include_categories = include_categories
        self.include_mechanics = include_mechanics
        self.include_designers = include_designers
        self.include_artists = include_artists
        self.include_publishers = include_publishers
        self.include_families = include_families
        self.custom_feature_patterns = custom_feature_patterns
        self.exclude_feature_patterns = exclude_feature_patterns
        self.always_include_columns = always_include_columns
        self.always_exclude_columns = always_exclude_columns
        
        # Initialize the underlying preprocessor
        self.preprocessor = BGGPreprocessor(
            max_player_count=self.max_player_count,
            reference_year=self.reference_year,
            normalization_factor=self.normalization_factor,
            category_min_freq=self.category_min_freq,
            mechanic_min_freq=self.mechanic_min_freq,
            designer_min_freq=self.designer_min_freq,
            artist_min_freq=self.artist_min_freq,
            publisher_min_freq=self.publisher_min_freq,
            family_min_freq=self.family_min_freq,
            max_category_features=self.max_category_features,
            max_mechanic_features=self.max_mechanic_features,
            max_designer_features=self.max_designer_features,
            max_artist_features=self.max_artist_features,
            max_publisher_features=self.max_publisher_features,
            max_family_features=self.max_family_features,
            handle_missing_values=self.handle_missing_values,
            transform_year=self.transform_year,
            create_player_dummies=self.create_player_dummies,
            create_category_mechanic_features=self.create_category_mechanic_features,
            create_designer_artist_features=self.create_designer_artist_features,
            create_publisher_features=self.create_publisher_features,
            create_family_features=self.create_family_features,
            include_base_numeric=self.include_base_numeric,
            include_average_weight=self.include_average_weight,
            include_player_count=self.include_player_count,
            include_categories=self.include_categories,
            include_mechanics=self.include_mechanics,
            include_designers=self.include_designers,
            include_artists=self.include_artists,
            include_publishers=self.include_publishers,
            include_families=self.include_families,
            custom_feature_patterns=self.custom_feature_patterns,
            exclude_feature_patterns=self.exclude_feature_patterns,
            always_include_columns=self.always_include_columns,
            always_exclude_columns=self.always_exclude_columns,
        )
    
    def _convert_to_polars(self, X: Any) -> pl.DataFrame:
        """Convert input to a polars DataFrame."""
        if isinstance(X, pl.DataFrame):
            return X
        elif isinstance(X, pd.DataFrame):
            return pl.from_pandas(X)
        else:
            raise ValueError(f"Input type {type(X)} not supported")
    
    def fit(self, X: Any, y=None) -> 'BGGSklearnPreprocessor':
        """Fit the preprocessor.
        
        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            The input data to fit the preprocessor on.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : BGGSklearnPreprocessor
            The fitted preprocessor.
        """
        # Convert input to polars DataFrame
        X_pl = self._convert_to_polars(X)
        
        # Fit the preprocessor
        self.preprocessor.fit(X_pl)
        
        return self
    
    def transform(self, X: Any) -> pd.DataFrame:
        """Transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            The input data to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        # Convert input to polars DataFrame
        X_pl = self._convert_to_polars(X)
        
        # Transform the data
        features, _ = self.preprocessor.transform(X_pl)
        
        # Convert to pandas DataFrame
        return features.to_pandas()
    
    def fit_transform(self, X: Any, y=None) -> pd.DataFrame:
        """Fit the preprocessor and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame or pl.DataFrame
            The input data to fit the preprocessor on and transform.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature names.
        
        Returns
        -------
        List[str]
            The list of feature names.
        """
        return self.preprocessor.get_selected_features()


# Example usage:
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from src.data.loader import BGGDataLoader
    from src.models.train_hurdle_model import load_config
    
    # Load data
    config = load_config()
    loader = BGGDataLoader(config)
    full_data = loader.load_training_data(end_train_year=2022, min_ratings=0)
    
    # Create and configure the preprocessor with default settings
    preprocessor = BGGSklearnPreprocessor()
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Get pre-computed hurdle target from SQL
    y = full_data.select("hurdle").to_series().to_pandas()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        full_data, y, test_size=0.2, random_state=42
    )
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
