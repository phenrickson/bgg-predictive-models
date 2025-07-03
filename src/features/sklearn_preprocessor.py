"""Standalone scikit-learn compatible BGG preprocessor."""
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
import re
import logging
from sklearn.base import BaseEstimator, TransformerMixin


class BGGSklearnPreprocessor(BaseEstimator, TransformerMixin):
    """Standalone scikit-learn compatible BGG preprocessor.
    
    This class implements BGG-specific feature engineering directly without
    dependencies on other preprocessor classes.
    
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
    
    create_category_features : bool, default=True
        Whether to create category features.
    
    create_mechanic_features : bool, default=True
        Whether to create mechanic features.
    
    create_designer_features : bool, default=True
        Whether to create designer features.
    
    create_artist_features : bool, default=True
        Whether to create artist features.
    
    create_publisher_features : bool, default=True
        Whether to create publisher features.
    
    create_family_features : bool, default=True
        Whether to create family features.
    
    include_base_numeric : bool, default=True
        Whether to include base numeric features.
    
    include_average_weight : bool, default=False
        Whether to include average_weight as a feature.
    """
    
    def __init__(
        self,
        # Feature generation parameters
        max_player_count: int = 10,
        
        # Year transformation parameters
        reference_year: int = 2000,
        normalization_factor: int = 25,
        year_transform_type: str = 'normalized',  # New parameter
        
        # Logging parameters
        verbose: bool = False,
        
        # Array feature parameters
        category_min_freq: int = 0,      # Lower default threshold
        mechanic_min_freq: int = 0,      # Lower default threshold
        designer_min_freq: int = 10,     # Lower default threshold
        artist_min_freq: int = 10,       # Lower default threshold
        publisher_min_freq: int = 5,     # Lower default threshold
        family_min_freq: int = 10,        # Lower default threshold
        max_category_features: int = 100, # More reasonable default
        max_mechanic_features: int = 100, # More reasonable default
        max_designer_features: int = 50,  # More reasonable default
        max_artist_features: int = 50,    # More reasonable default
        max_publisher_features: int = 25, # More reasonable default
        max_family_features: int = 50,    # More reasonable default
        
        # Feature generation flags
        handle_missing_values: bool = True,
        transform_year: bool = True,
        create_player_dummies: bool = True,
        create_category_features: bool = True,
        create_mechanic_features: bool = True,
        create_designer_features: bool = True,
        create_artist_features: bool = True,
        create_publisher_features: bool = True,
        create_family_features: bool = True,
        create_missingness_features: bool = False,  # New parameter
        
        # Feature selection parameters
        include_base_numeric: bool = True,
        include_average_weight: bool = False,
        variance_threshold: float = 0.0,  # New parameter to drop zero-variance features
    ):
        """Initialize the preprocessor."""
        self.max_player_count = max_player_count
        self.reference_year = reference_year
        self.normalization_factor = normalization_factor
        self.year_transform_type = year_transform_type  # Add this line
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
        self.create_category_features = create_category_features
        self.create_mechanic_features = create_mechanic_features
        self.create_designer_features = create_designer_features
        self.create_artist_features = create_artist_features
        self.create_publisher_features = create_publisher_features
        self.create_family_features = create_family_features
        self.create_missingness_features = create_missingness_features  # New attribute
        self.include_base_numeric = include_base_numeric
        self.include_average_weight = include_average_weight
        self.verbose = verbose
        
        # Fitted attributes
        self.feature_names_ = None
        self.frequent_categories_ = None
        self.frequent_mechanics_ = None
        self.frequent_designers_ = None
        self.frequent_artists_ = None
        self.frequent_publishers_ = None
        self.frequent_families_ = None
        
        # Publisher allow list
        self.ALLOWED_PUBLISHER_NAMES = {
            "Hasbro", "Mayfair Games", "Decision Games", "Multi-Man Publishing",
            "Alderac Entertainment Group", "Days of Wonder", "Pandasaurus Games",
            "(web published)", "(Self-Published)", "Splotter Spellen", "Asmodee",
            "Ravensburger", "Parker Brothers", "Pegasus Spiele", "KOSMOS",
            "Milton Bradley", "Rio Grande Games", "Z-Man Games", "GMT Games",
            "Fantasy Flight Games", "Avalon Hill", "(Unknown)", "Eagle-Gryphon Games",
            "Matagot", "Games Workshop Ltd", "Queen Games", "Stronghold Games",
            "Steve Jackson Games", "Wizards of the Coast", "Cryptozoic Entertainment",
            "Plaid Hat Games", "CMON Global Limited", "Gamewright", "WizKids",
            "(Public Domain)", "Mattel, Inc", "Space Cowboys", "Stonemaier Games",
            "Plan B Games", "Capstone Games", "Chip Theory Games", "Ares Games",
            "Greater Than Games", "Renegade Games", "Restoration Games", "Osprey Games",
            "Roxley", "Czech Games Edition", "Awaken Realms", "Compass Games",
            "Button Shy", "The Game Crafter", "Cheapass Games", "alea",
            "NorthStar Game Studio", "Bézier Games", "Red Raven Games",
            "3W (World Wide Wargames)"
        }
        
        # Family patterns
        self.FAMILY_REMOVE_PATTERNS = [
            "^Admin:", "^Misc:", "^Promotional:", "^Digital Implementations:",
            "^Crowdfunding: Spieleschmiede", "^Crowdfunding: Verkami", 
            "^Crowdfunding: Indiegogo", "^Contests:", "^Game:",
            "^Players: Expansions", "^Players: Games with expansions"
        ]
        
        self.FAMILY_ALLOW_PATTERNS = [
            "^Series: Monopoly-Like", "^Series: 18xx", "^Series: Cards Against Humanity-Like",
            "^Series: Exit: The Game", "^Players: Games with Solitaire Rules",
            "^Players: Wargames with Solitaire Rules", "^Players: One versus Many",
            "^Players: Solitaire Only Games", "^Players: Solitaire Only Wargames",
            "^Players: Two-Player Only Games", "^Players: Three Players Only Games",
            "^Players: Wargames with Rules Supporting Only Two Players",
            "^Players: Solitaire Only Card Games", "^Country:", "^Animals", "^History",
            "^Sports", "^Category", "^Cities", "^Traditional", "^Creatures", "^TV",
            "^Region", "^Card", "^Comic", "^Ancient", "^Brands", "^Versions & Editions",
            "^Food", "^Movies", "^Setting", "^Card Games", "^Collectible", "^Containers",
            "^Crowdfunding: Kickstarter", "^Crowdfunding: Gamefound", "^Authors",
            "^Characters", "^Religious", "^Holidays", "^Space", "^Folk", "^Word",
            "^Mythology", "^Occupation", "^Celebrities", "^Toys"
        ]
    
    def _safe_column_name(self, name: str) -> str:
        """Create a safe column name from a string."""
        return str(name).lower().replace(' ', '_').replace('-', '_').replace(':', '').replace('/', '_')
    
    def _handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Replace zeros with NaN, create missingness indicators, and apply log transformation to numeric features."""
        if not self.handle_missing_values:
            return df, pd.DataFrame(index=df.index)
        
        logger = logging.getLogger(__name__)
        logger.info("Processing missing values:")
        
        df = df.copy()
        missingness_df = pd.DataFrame(index=df.index)
        
        # Dynamically identify numeric columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Replace zeros with NaN for numeric columns
        for col in numeric_columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                logger.info(f"  {col}: replacing {zero_count} zeros with NaN ({(zero_count/len(df))*100:.2f}% of values)")
                df[col] = df[col].replace(0, np.nan)
            
            # Create missingness indicator if feature is enabled
            if self.create_missingness_features:
                missingness_df[f"{col}_missing"] = df[col].isna().astype(int)
        
        # Handle numeric features
        for col in numeric_columns:
            # Compute median from non-zero values (SimpleImputer-like strategy)
            non_zero_vals = df[df[col] > 0][col]
            if len(non_zero_vals) > 0:
                median_val = non_zero_vals.median()
            else:
                # If no non-zero values, use a small default value
                median_val = 1.0
                logger.warning(f"  {col}: no non-zero values found, using default value {median_val}")
            
            # Impute NaN values with median
            df[col] = df[col].fillna(median_val)
            logger.info(f"  {col}: imputed NaN values with median {median_val:.1f}")
            
            # Apply log transformation for specific columns
            log_transform_columns = ['min_age', 'min_playtime', 'max_playtime', 'users_rated']
            if col in log_transform_columns:
                df[col] = np.log1p(df[col])
                logger.info(f"  {col}: applied log1p transformation")
        
        return df, missingness_df
    
    def _transform_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform year_published variable."""
        if not self.transform_year or "year_published" not in df.columns:
            return df
        
        logger = logging.getLogger(__name__)
        logger.info("Transforming year features")
        
        df = df.copy()
        
        # Remove rows with missing year_published
        df = df.dropna(subset=["year_published"])
        
        # Always create all year features
        df["year_published_centered"] = df["year_published"] - self.reference_year
        df["year_published_normalized"] = (df["year_published"] - self.reference_year) / self.normalization_factor
        
        # Log-transformed normalized distance (with safe handling for edge cases)
        df["year_published_transformed"] = np.where(
            df["year_published"] <= self.reference_year,
            np.log(np.maximum(self.reference_year - df["year_published"] + 1, 1e-8)),
            np.log(np.maximum(df["year_published"] - self.reference_year + 1, 1e-8))
        )
        
        # Select the appropriate year transformation
        year_transform_column = {
            'centered': 'year_published_centered',
            'normalized': 'year_published_normalized',
            'transformed': 'year_published_transformed'
        }.get(self.year_transform_type, 'year_published_normalized')
        
        # Always return all year features
        logger.info(f"Selected year transform column: {year_transform_column}")
        logger.info("Returning all year transformation features")
        
        return df[["year_published_centered", "year_published_normalized", "year_published_transformed"]]
    
    def _create_mechanics_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a feature counting the number of mechanics a game has."""
        if "mechanics" not in df.columns:
            return df
        
        df = df.copy()
        df["mechanics_count"] = df["mechanics"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        return df
    
    def _create_player_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for player counts."""
        if not self.create_player_dummies:
            return df
        
        df = df.copy()
        
        for count in range(1, self.max_player_count + 1):
            col_name = f"player_count_{count}"
            df[col_name] = (
                (df["min_players"] <= count) & 
                (df["max_players"] >= count)
            ).astype(int)
        
        return df
    
    def _fit_array_features(self, df: pd.DataFrame, column: str, min_freq: int, max_features: int) -> List[str]:
        """Fit array features and return frequent values."""
        logger = logging.getLogger(__name__)
        
        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return []
        
        # Detailed logging of column contents
        logger.info(f"Analyzing column: {column}")
        logger.info(f"Column dtype: {df[column].dtype}")
        
        # Explode the array column and count frequencies
        all_values = []
        for idx, values in enumerate(df[column]):
            # Extensive type checking and logging
            if values is None:
                logger.debug(f"  Row {idx}: None value")
                continue
            
            # Handle various list-like types
            if isinstance(values, list):
                logger.debug(f"  Row {idx}: Python list, length {len(values)}")
                all_values.extend(values)
            elif hasattr(values, '__array__'):  # Handle numpy arrays
                logger.debug(f"  Row {idx}: Numpy-like array")
                # Convert to list and extend
                values_list = values.tolist() if hasattr(values, 'tolist') else list(values)
                # Filter out empty strings and None values
                values_list = [str(v).strip() for v in values_list if v and str(v).strip() != '']
                all_values.extend(values_list)
            else:
                # Attempt to convert to string and handle
                try:
                    str_val = str(values).strip()
                    if str_val:
                        logger.debug(f"  Row {idx}: Converted to string: {str_val}")
                        all_values.append(str_val)
                    else:
                        logger.debug(f"  Row {idx}: Empty string after conversion")
                except Exception as e:
                    logger.warning(f"Unexpected value type in {column} at row {idx}: {values} (type: {type(values)})")
        
        # Count frequencies
        value_counts = pd.Series(all_values).value_counts()
        
        # Filter by frequency and limit
        frequent_values = value_counts[value_counts >= min_freq].head(max_features).index.tolist()
        
        # Detailed logging
        logger.info(f"Processing {column}:")
        logger.info(f"  Total input rows: {len(df)}")
        logger.info(f"  Total extracted values: {len(all_values)}")
        logger.info(f"  Unique values: {len(value_counts)}")
        logger.info(f"  Selected {len(frequent_values)} frequent values")
        
        # Log the top frequent values
        if frequent_values:
            logger.info("  Top frequent values:")
            for val, count in value_counts[frequent_values].items():
                logger.info(f"    {val}: {count} occurrences")
        
        return frequent_values
    
    def _create_array_features(self, df: pd.DataFrame, column: str, frequent_values: List[str], prefix: str) -> pd.DataFrame:
        """Create one-hot encoded features for array column."""
        logger = logging.getLogger(__name__)
        
        if not frequent_values or column not in df.columns:
            logger.warning(f"No frequent values or column {column} not found")
            return df
        
        # Detailed logging of column contents
        logger.info(f"Creating array features for column: {column}")
        logger.info(f"Column dtype: {df[column].dtype}")
        logger.info(f"Frequent values: {frequent_values}")
        
        # Create a dictionary to store new columns
        new_columns = {}
        
        # Initialize all columns with zeros first
        for value in frequent_values:
            if value is not None:
                col_name = f"{prefix}_{self._safe_column_name(value)}"
                new_columns[col_name] = pd.Series(0, index=df.index)
        
        # Detailed tracking of feature generation
        feature_generation_stats = {val: 0 for val in frequent_values}
        
        # Then set 1s only for values that were in the frequent_values list
        for idx, values in df[column].items():
            # Extensive type checking and logging
            if values is None:
                logger.debug(f"  Row {idx}: None value")
                continue
            
            # Handle various list-like types
            if isinstance(values, list):
                logger.debug(f"  Row {idx}: Python list, length {len(values)}")
                values_list = values
            elif hasattr(values, '__array__'):  # Handle numpy arrays
                logger.debug(f"  Row {idx}: Numpy-like array")
                values_list = values.tolist() if hasattr(values, 'tolist') else list(values)
            else:
                # Attempt to convert to list
                try:
                    values_list = list(values)
                    logger.debug(f"  Row {idx}: Converted to list, length {len(values_list)}")
                except Exception as e:
                    logger.warning(f"Unexpected value type in {column} at row {idx}: {values} (type: {type(values)})")
                    continue
            
            # Filter and set features
            for value in values_list:
                if value in frequent_values:
                    col_name = f"{prefix}_{self._safe_column_name(value)}"
                    new_columns[col_name][idx] = 1
                    feature_generation_stats[value] += 1
        
        # Create a new DataFrame with only the columns from frequent_values
        new_df = pd.DataFrame(new_columns, index=df.index)
        
        # Log feature generation statistics
        logger.info("Feature generation statistics:")
        for val, count in feature_generation_stats.items():
            logger.info(f"  {val}: {count} occurrences")
        
        # Validate feature generation
        if new_df.empty:
            logger.warning(f"No features generated for column {column}")
        
        # Return only the new feature columns
        return new_df
    
    def _filter_publishers(self, publishers: List[str]) -> List[str]:
        """Filter publishers to only allowed ones."""
        if isinstance(publishers, list):
            return [p for p in publishers if p in self.ALLOWED_PUBLISHER_NAMES]
        elif hasattr(publishers, '__array__'):  # Handle numpy arrays
            return [p for p in publishers.tolist() if p in self.ALLOWED_PUBLISHER_NAMES]
        return []
    
    def _filter_families(self, families: List[str]) -> List[str]:
        """Filter families based on allow/remove patterns."""
        # Convert numpy array to list if needed
        if hasattr(families, '__array__'):
            families = families.tolist()
        elif not isinstance(families, list):
            return []
        
        remove_pattern = re.compile("|".join(self.FAMILY_REMOVE_PATTERNS))
        allow_pattern = re.compile("|".join(self.FAMILY_ALLOW_PATTERNS))
        
        filtered = []
        for family in families:
            if family is not None:
                # Skip if it matches the remove pattern
                if remove_pattern.search(family):
                    continue
                # Include if it matches the allow pattern
                if allow_pattern.search(family):
                    filtered.append(family)
        
        return filtered
    
    def fit(self, X: pd.DataFrame, y=None) -> 'BGGSklearnPreprocessor':
        """Fit the preprocessor.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the preprocessor on.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : BGGSklearnPreprocessor
            The fitted preprocessor.
        """
        X = X.copy()
        
        # Fit array features
        if self.create_category_features:
            self.frequent_categories_ = self._fit_array_features(
                X, "categories", self.category_min_freq, self.max_category_features
            )
        
        if self.create_mechanic_features:
            self.frequent_mechanics_ = self._fit_array_features(
                X, "mechanics", self.mechanic_min_freq, self.max_mechanic_features
            )
        
        if self.create_designer_features:
            self.frequent_designers_ = self._fit_array_features(
                X, "designers", self.designer_min_freq, self.max_designer_features
            )
        
        if self.create_artist_features:
            self.frequent_artists_ = self._fit_array_features(
                X, "artists", self.artist_min_freq, self.max_artist_features
            )
        
        if self.create_publisher_features:
            # Filter publishers first
            X_filtered = X.copy()
            X_filtered["publishers"] = X_filtered["publishers"].apply(self._filter_publishers)
            self.frequent_publishers_ = self._fit_array_features(
                X_filtered, "publishers", self.publisher_min_freq, self.max_publisher_features
            )
        
        if self.create_family_features:
            # Filter families first
            X_filtered = X.copy()
            X_filtered["families"] = X_filtered["families"].apply(self._filter_families)
            self.frequent_families_ = self._fit_array_features(
                X_filtered, "families", self.family_min_freq, self.max_family_features
            )
        
        # Generate feature names
        self._generate_feature_names()
        
        return self
    
    def _generate_feature_names(self):
        """Generate the list of feature names."""
        feature_names = []
        
        # Detailed logging for feature generation
        logger = logging.getLogger(__name__)
        logger.info("Generating feature names:")
        
        # ALWAYS include base numeric features
        # Unconditionally add year features
        year_features = [
            "year_published_centered",
            "year_published_normalized", 
            "year_published_transformed"
        ]
        
        # Debug: Add explicit checks for year feature generation
        logger.info("Debug: Year Feature Generation")
        logger.info(f"  transform_year: {self.transform_year}")
        logger.info(f"  include_base_numeric: {self.include_base_numeric}")
        logger.info(f"  year_transform_type: {self.year_transform_type}")
        
        # Unconditional addition of year features
        feature_names.extend(year_features)
        logger.info(f"  Year features (ALWAYS ADDED): {year_features}")
        
        # Explicit check to ensure year features are added
        for feat in year_features:
            if feat not in feature_names:
                logger.error(f"CRITICAL: {feat} was NOT added to feature names!")
                feature_names.append(feat)
        
        # Base numeric features
        base_numeric_features = [
            "mechanics_count",
            "min_age",
            "min_playtime",
            "max_playtime"
        ]
        feature_names.extend(base_numeric_features)
        logger.info(f"  Base numeric features: {base_numeric_features}")
        
        # ALWAYS add missingness features, regardless of flag
        missingness_features = [
            "min_age_missing",
            "min_playtime_missing",
            "max_playtime_missing",
            "year_published_missing"
        ]
        feature_names.extend(missingness_features)
        logger.info(f"  Missingness features (ALWAYS ADDED): {missingness_features}")
        logger.info(f"  create_missingness_features flag: {self.create_missingness_features}")
        
        # Average weight
        if self.include_average_weight:
            feature_names.append("average_weight")
            logger.info("  Average weight feature added")
        
        # Player count features
        if self.create_player_dummies:
            player_dummy_features = [f"player_count_{count}" for count in range(1, self.max_player_count + 1)]
            feature_names.extend(player_dummy_features)
            logger.info(f"  Player dummy features: {player_dummy_features}")
        
        # Array features with detailed logging
        array_feature_types = [
            ('category', self.create_category_features, self.frequent_categories_),
            ('mechanic', self.create_mechanic_features, self.frequent_mechanics_),
            ('designer', self.create_designer_features, self.frequent_designers_),
            ('artist', self.create_artist_features, self.frequent_artists_),
            ('publisher', self.create_publisher_features, self.frequent_publishers_),
            ('family', self.create_family_features, self.frequent_families_)
        ]
        
        for prefix, enabled, frequent_values in array_feature_types:
            if enabled and frequent_values:
                array_features = [f"{prefix}_{self._safe_column_name(val)}" for val in frequent_values]
                feature_names.extend(array_features)
                logger.info(f"  {prefix.capitalize()} features: {len(array_features)} features")
                # Optionally log first few feature names to avoid overwhelming log
                if len(array_features) > 10:
                    logger.info(f"    First 10 {prefix} features: {array_features[:10]}")
        
        # Final logging
        logger.info(f"Total generated features: {len(feature_names)}")
        
        # Validate feature names
        logger.info("Validating feature names:")
        for feature in feature_names:
            logger.info(f"  {feature}")
        
        # CRITICAL: Ensure year features are ALWAYS present
        missing_year_features = [
            feat for feat in [
                "year_published_centered", 
                "year_published_normalized", 
                "year_published_transformed"
            ] if feat not in feature_names
        ]
        
        if missing_year_features:
            logger.critical(f"CRITICAL: Missing year features: {missing_year_features}")
            feature_names.extend(missing_year_features)
        
        self.feature_names_ = feature_names
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.
        
        Returns
        -------
        pd.DataFrame
            The transformed data.
        """
        if self.feature_names_ is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Detailed logging for debugging
        logger = logging.getLogger(__name__)
        logger.info("Starting transform method")
        logger.info(f"Fitted feature names: {self.feature_names_}")
        logger.info(f"Total fitted features: {len(self.feature_names_)}")
        
        # Create a copy of input data for base transformations
        X_base = X.copy()
        
        # Handle missing values and get missingness indicators
        if self.handle_missing_values:
            X_base, missingness_df = self._handle_missing_values(X_base)
        else:
            missingness_df = pd.DataFrame(index=X_base.index)
        
        # Initialize list to store all feature DataFrames
        feature_dfs = []
        
        # ALWAYS add missingness features, regardless of flag
        if not missingness_df.empty:
            feature_dfs.append(missingness_df)
            logger.info("Added missingness features")
            logger.info(f"  create_missingness_features flag: {self.create_missingness_features}")
        
        # Transform year features if enabled
        if self.transform_year and "year_published" in X_base.columns:
            year_df = self._transform_year(X_base[["year_published"]])
            if not year_df.empty:
                feature_dfs.append(year_df)
                logger.info("Added year transformation features")
        
        # Create mechanics count if mechanics column exists
        if "mechanics" in X_base.columns:
            mechanics_df = pd.DataFrame(index=X_base.index)
            mechanics_df["mechanics_count"] = X_base["mechanics"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            feature_dfs.append(mechanics_df)
            logger.info("Added mechanics count feature")
        
        # Create player dummies if enabled
        if self.create_player_dummies:
            player_dummies = pd.DataFrame(index=X_base.index)
            for count in range(1, self.max_player_count + 1):
                player_dummies[f"player_count_{count}"] = (
                    (X_base["min_players"] <= count) & 
                    (X_base["max_players"] >= count)
                ).astype(int)
            feature_dfs.append(player_dummies)
            logger.info("Added player dummy features")
        
        # Create array features
        array_features = [
            ("categories", self.create_category_features, self.frequent_categories_, "category", None),
            ("mechanics", self.create_mechanic_features, self.frequent_mechanics_, "mechanic", None),
            ("designers", self.create_designer_features, self.frequent_designers_, "designer", None),
            ("artists", self.create_artist_features, self.frequent_artists_, "artist", None),
            ("publishers", self.create_publisher_features, self.frequent_publishers_, "publisher", self._filter_publishers),
            ("families", self.create_family_features, self.frequent_families_, "family", self._filter_families)
        ]
        
        for col, enabled, frequent_values, prefix, filter_func in array_features:
            if enabled and frequent_values and col in X_base.columns:
                X_filtered = X_base.copy() if filter_func else X_base
                if filter_func:
                    X_filtered[col] = X_filtered[col].apply(filter_func)
                feature_df = self._create_array_features(X_filtered, col, frequent_values, prefix)
                if not feature_df.empty:
                    feature_dfs.append(feature_df)
                    logger.info(f"Added {prefix} features: {feature_df.columns.tolist()}")
        
        # Add base numeric features if enabled
        if self.include_base_numeric:
            numeric_features = ["min_age", "min_playtime", "max_playtime"]
            base_numeric_df = pd.DataFrame(index=X_base.index)
            missingness_df = pd.DataFrame(index=X_base.index)
            
            for col in numeric_features:
                if col in X_base.columns:
                    base_numeric_df[col] = X_base[col]
                    
                    # Add missingness features if enabled
                    if self.create_missingness_features:
                        missingness_df[f"{col}_missing"] = X_base[col].isna().astype(int)
            
            # Add year published missingness if enabled
            if self.create_missingness_features and "year_published" in X_base.columns:
                missingness_df["year_published_missing"] = X_base["year_published"].isna().astype(int)
            
            if not base_numeric_df.empty:
                feature_dfs.append(base_numeric_df)
                logger.info("Added base numeric features")
            
            if self.create_missingness_features and not missingness_df.empty:
                feature_dfs.append(missingness_df)
                logger.info("Added numeric missingness features")
        
        # Add average weight if enabled
        if self.include_average_weight and "average_weight" in X_base.columns:
            weight_df = pd.DataFrame({"average_weight": X_base["average_weight"]})
            feature_dfs.append(weight_df)
            logger.info("Added average weight feature")
        
        # Concatenate all feature DataFrames
        if feature_dfs:
            result = pd.concat(feature_dfs, axis=1)
            
            # Detailed logging of generated features
            logger.info("Generated features:")
            for col in result.columns:
                logger.info(f"  {col}")
            
            # Ensure we only return the features we want in the correct order
            available_features = [col for col in self.feature_names_ if col in result.columns]
            
            # Log any missing features
            missing_features = set(self.feature_names_) - set(available_features)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            result = result[available_features]
            
            # Log NaN information
            logger.info("NaN value counts in transformed features:")
            nan_counts = result.isna().sum()
            nan_columns = nan_counts[nan_counts > 0]
            if not nan_columns.empty:
                for col, count in nan_columns.items():
                    logger.info(f"  {col}: {count} NaN values ({(count/len(result))*100:.2f}%)")
            else:
                logger.info("  No NaN values found in transformed features")
            
            return result
        else:
            # Return empty DataFrame with correct columns if no features were created
            return pd.DataFrame(columns=self.feature_names_, index=X_base.index)
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit the preprocessor and transform the data.
        
        Parameters
        ----------
        X : pd.DataFrame
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
        if self.feature_names_ is None:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        return self.feature_names_.copy()


# Example usage:
if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from src.data.loader import BGGDataLoader
    from src.data.config import load_config
    
    # Load data
    config = load_config()
    loader = BGGDataLoader(config)
    full_data = loader.load_training_data(end_train_year=2022, min_ratings=0)
    
    # Convert to pandas
    df_pandas = full_data.to_pandas()
    
    # Create and configure the preprocessor with default settings
    preprocessor = BGGSklearnPreprocessor()
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Get pre-computed hurdle target from SQL
    y = df_pandas["hurdle"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df_pandas, y, test_size=0.2, random_state=42
    )
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Print classification report
    print(classification_report(y_test, y_pred))
