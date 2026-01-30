import numpy as np
import pandas as pd
import re
import logging
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer


class ColumnTransformerNoPrefix(ColumnTransformer):
    """
    ColumnTransformer that removes prefixes from feature names.

    This class extends sklearn's ColumnTransformer to provide cleaner feature names
    by removing the transformer name prefix that is normally added.
    """

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation, removing prefixes.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features_out,), dtype=str
            Transformed feature names without prefixes.
        """
        raw_names = super().get_feature_names_out(input_features)
        return np.array([name.split("__")[-1] for name in raw_names])


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to apply log(1+x) transformation to specified columns.

    Parameters
    ----------
    columns : list, optional (default=['min_age', 'min_playtime', 'max_playtime'])
        List of column names to apply log transformation to.

    Attributes
    ----------
    columns_ : list
        Columns that will be log-transformed.
    """

    def __init__(self, columns=None):
        if columns is None:
            columns = ["min_age", "min_playtime", "max_playtime"]
        self.columns = columns
        self.columns_ = None

    def fit(self, X, y=None):
        """
        Validate the columns to be transformed.

        Parameters
        ----------
        X : pandas.DataFrame
            Input data
        y : None
            Ignored

        Returns
        -------
        self : LogTransformer
            Fitted transformer
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Validate columns
        self.columns_ = [col for col in self.columns if col in X.columns]

        return self

    def transform(self, X):
        """
        Apply log(1+x) transformation to specified columns.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Input data to transform

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Transformed data
        """
        # Determine output type based on input
        is_pandas = isinstance(X, pd.DataFrame)

        # Convert to DataFrame if not already
        if not is_pandas:
            X = pd.DataFrame(X)

        # Create a copy to avoid modifying the original
        X_transformed = X.copy()

        # Apply log(1+x) transformation to specified columns
        for col in self.columns_:
            X_transformed[col] = np.log1p(X_transformed[col])

        # Return in the same type as input
        return X_transformed if is_pandas else X_transformed.to_numpy()

    def set_output(self, *, transform=None):
        """
        Set the output configuration for the transformer.

        Parameters
        ----------
        transform : str, optional
            Output format. Typically 'default', 'pandas', or None.

        Returns
        -------
        self : object
            Returns self with updated output configuration.
        """
        if transform is not None and transform not in ["default", "pandas"]:
            raise ValueError(
                f"Invalid transform parameter: {transform}. Must be 'default' or 'pandas'."
            )

        self._output_config = transform
        return self

    def get_feature_names_out(self, input_features=None):
        """
        Return the names of the transformed year features.

        Returns
        -------
        ndarray
            Names of the transformed year features.
        """
        return np.array(["year_published_transformed"])


class YearTransformer(BaseEstimator, TransformerMixin):
    """Transformer for year-related feature engineering.

    Parameters
    ----------
    reference_year : int, default=2000
        Reference year for year transformations.

    normalization_factor : int, default=25
        Normalization factor for year transformations.
    """

    def __init__(self, reference_year: int = 2000, normalization_factor: int = 25):
        """Initialize the year transformer."""
        self.reference_year = reference_year
        self.normalization_factor = normalization_factor
        self.to_drop_ = [
            "year_published",
            "year_published_centered",
            "year_published_normalized",
        ]

    def fit(self, X, y=None):
        """Fit method (does nothing for this transformer)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform year features.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing 'year_published' column.

        Returns
        -------
        pd.DataFrame
            DataFrame with transformed year features.
        """
        # Ensure input is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure 'year_published' column exists
        if "year_published" not in X.columns:
            raise ValueError("Input DataFrame must contain 'year_published' column")

        # Create a copy to avoid modifying the original
        X_copy = X.copy()

        # Create year features (convert to float64 for ML model compatibility)
        X_copy["year_published_centered"] = (
            X_copy["year_published"] - self.reference_year
        ).astype(np.float64)
        X_copy["year_published_normalized"] = (
            (X_copy["year_published"] - self.reference_year) / self.normalization_factor
        ).astype(np.float64)

        # Log-transformed normalized distance (with safe handling for edge cases)
        X_copy["year_published_transformed"] = np.where(
            X_copy["year_published"] <= self.reference_year,
            np.log(
                np.maximum(self.reference_year - X_copy["year_published"] + 1, 1e-8)
            ),
            np.log(
                np.maximum(X_copy["year_published"] - self.reference_year + 1, 1e-8)
            ),
        )

        # Return the entire original DataFrame
        return X_copy.drop(columns=self.to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        """
        Return the names of the transformed year features.

        Returns
        -------
        ndarray
            Names of the transformed year features.
        """
        return np.array(["year_published_transformed"])

    def set_output(self, *, transform=None):
        """
        Set the output configuration for the transformer.

        Parameters
        ----------
        transform : str, optional
            Output format. Typically 'default', 'pandas', or None.

        Returns
        -------
        self : object
            Returns self with updated output configuration.
        """
        if transform is not None and transform not in ["default", "pandas"]:
            raise ValueError(
                f"Invalid transform parameter: {transform}. Must be 'default' or 'pandas'."
            )

        # Set the output configuration attribute
        self._output_config = transform

        # Set the output configuration for scikit-learn compatibility
        if hasattr(self, "output_"):
            self.output_ = transform

        return self


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove features with correlation above a threshold.

    This transformer identifies and removes features that are highly correlated
    with other features. When two features have correlation above the threshold,
    the second one (in column order) is dropped.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold. Features with absolute correlation above this
        value will be dropped. Default is 0.95 to remove highly correlated features.

    Attributes
    ----------
    to_drop_ : list
        List of column names that will be dropped during transform.
    feature_names_in_ : array
        Names of features seen during fit.
    feature_names_out_ : array
        Names of features after dropping correlated ones.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.to_drop_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """Fit the transformer by identifying correlated features to drop.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self
        """
        logger = logging.getLogger(__name__)

        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = np.array(X.columns)

        # Sample for faster correlation computation on large datasets
        if len(X) > 10000:
            X_sample = X.sample(n=10000, random_state=42)
        else:
            X_sample = X

        # Compute correlation matrix
        corr_matrix = X_sample.corr().abs()
        upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper)

        # Find columns to drop
        self.to_drop_ = [
            column
            for column in upper_corr.columns
            if any(upper_corr[column] > self.threshold)
        ]

        self.feature_names_out_ = np.array(
            [col for col in X.columns if col not in self.to_drop_]
        )

        if len(self.to_drop_) > 0:
            logger.info(
                f"CorrelationFilter: dropping {len(self.to_drop_)} features "
                f"with correlation > {self.threshold}"
            )

        return self

    def transform(self, X):
        """Remove correlated features.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : DataFrame
            Data with correlated features removed.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.to_drop_, errors="ignore")

    def set_output(self, *, transform=None):
        self._output_config = transform
        return self

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        if self.feature_names_out_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return self.feature_names_out_


def bgg_numeric_transformer(numeric_columns=None):
    """
    Create a numeric transformer pipeline for BGG data.

    Parameters
    ----------
    numeric_columns : list, optional
        List of numeric column names. Defaults to ['min_age', 'min_playtime', 'max_playtime'].

    Returns
    -------
    ColumnTransformerNoPrefix
        Configured numeric transformer pipeline.
    """
    if numeric_columns is None:
        numeric_columns = ["min_age", "min_playtime", "max_playtime"]

    numeric_transformer = ColumnTransformerNoPrefix(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=False
                ),
                numeric_columns,
            ),
            ("log", LogTransformer(columns=numeric_columns), numeric_columns),
        ],
        remainder="passthrough",
    )

    return numeric_transformer


class BaseBGGTransformer(BaseEstimator, TransformerMixin):
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
        # Logging parameters
        verbose: bool = False,
        # Array feature parameters
        category_min_freq: int = 0,  # Lower default threshold
        mechanic_min_freq: int = 0,  # Lower default threshold
        designer_min_freq: int = 10,  # Lower default threshold
        artist_min_freq: int = 10,  # Lower default threshold
        publisher_min_freq: int = 5,  # Lower default threshold
        family_min_freq: int = 10,  # Lower default threshold
        max_category_features: int = 500,  # More reasonable default
        max_mechanic_features: int = 500,  # More reasonable default
        max_designer_features: int = 250,  # More reasonable default
        max_artist_features: int = 250,  # More reasonable default
        max_publisher_features: int = 150,  # More reasonable default
        max_family_features: int = 250,  # More reasonable default
        # Feature generation flags
        handle_missing_values: bool = True,
        create_player_dummies: bool = True,
        create_category_features: bool = True,
        create_mechanic_features: bool = True,
        create_designer_features: bool = True,
        create_artist_features: bool = True,
        create_publisher_features: bool = True,
        create_family_features: bool = True,
        # Feature selection parameters
        include_base_numeric: bool = True,
        include_count_features: bool = False,
        include_average_weight: bool = False,
        # Column preservation parameters
        preserve_columns: Optional[List[str]] = None,
        # Embedding features
        include_description_embeddings: bool = False,
        # Family pattern filters (regex patterns to match against family names)
        family_allow_patterns: Optional[List[str]] = None,
        family_remove_patterns: Optional[List[str]] = None,
    ):
        """Initialize the preprocessor.

        Parameters
        ----------
        max_player_count : int, optional
            Maximum number of player count dummy variables to create, by default 10

        verbose : bool, optional
            Whether to print detailed logging information, by default False

        *_min_freq : int, optional
            Minimum frequency for including array features (categories, mechanics, etc.)

        max_*_features : int, optional
            Maximum number of features to create for each array type

        create_*_features : bool, optional
            Flag to enable/disable generation of specific feature types
            - create_player_dummies: Create dummy variables for player counts
            - create_category_features: Create one-hot encoded category features
            - create_mechanic_features: Create one-hot encoded mechanic features
            - create_designer_features: Create one-hot encoded designer features
            - create_artist_features: Create one-hot encoded artist features
            - create_publisher_features: Create one-hot encoded publisher features
            - create_family_features: Create one-hot encoded family features

        include_base_numeric : bool, optional
            Include base numeric features like min_age, playtime, by default True

        include_average_weight : bool, optional
            Include average game weight as a feature, by default False

        preserve_columns : list, optional
            Additional columns to preserve in the transformed data, by default ['year_published']
        """
        # Feature generation parameters
        self.max_player_count = max_player_count

        # Logging parameters
        self.verbose = verbose

        # Array feature frequency thresholds
        self.category_min_freq = category_min_freq
        self.mechanic_min_freq = mechanic_min_freq
        self.designer_min_freq = designer_min_freq
        self.artist_min_freq = artist_min_freq
        self.publisher_min_freq = publisher_min_freq
        self.family_min_freq = family_min_freq

        # Maximum feature limits
        self.max_category_features = max_category_features
        self.max_mechanic_features = max_mechanic_features
        self.max_designer_features = max_designer_features
        self.max_artist_features = max_artist_features
        self.max_publisher_features = max_publisher_features
        self.max_family_features = max_family_features

        # Feature generation flags
        self.handle_missing_values = handle_missing_values
        self.create_player_dummies = create_player_dummies
        self.create_category_features = create_category_features
        self.create_mechanic_features = create_mechanic_features
        self.create_designer_features = create_designer_features
        self.create_artist_features = create_artist_features
        self.create_publisher_features = create_publisher_features
        self.create_family_features = create_family_features

        # Feature selection parameters
        self.include_base_numeric = include_base_numeric
        self.include_count_features = include_count_features
        self.include_average_weight = include_average_weight

        # Column preservation parameters
        self.preserve_columns = preserve_columns or ["year_published"]

        # Embedding features
        self.include_description_embeddings = include_description_embeddings

        # Family pattern filters (use defaults if not provided)
        self.family_allow_patterns = family_allow_patterns
        self.family_remove_patterns = family_remove_patterns

        # Fitted attributes (will be populated during fit)
        self.feature_names_ = None
        self.embedding_columns_ = None
        self.frequent_categories_ = None
        self.frequent_mechanics_ = None
        self.frequent_designers_ = None
        self.frequent_artists_ = None
        self.frequent_publishers_ = None
        self.frequent_families_ = None

        # Publisher allow list
        self.ALLOWED_PUBLISHER_NAMES = {
            "Hasbro",
            "Mayfair Games",
            "Decision Games",
            "Multi-Man Publishing",
            "Alderac Entertainment Group",
            "Days of Wonder",
            "Pandasaurus Games",
            "(web published)",
            "(Self-Published)",
            "Splotter Spellen",
            "Asmodee",
            "Ravensburger",
            "Parker Brothers",
            "Pegasus Spiele",
            "KOSMOS",
            "Milton Bradley",
            "Rio Grande Games",
            "Z-Man Games",
            "GMT Games",
            "Fantasy Flight Games",
            "Avalon Hill",
            "(Unknown)",
            "Eagle-Gryphon Games",
            "Matagot",
            "Games Workshop Ltd",
            "Queen Games",
            "Stronghold Games",
            "Steve Jackson Games",
            "Wizards of the Coast",
            "Cryptozoic Entertainment",
            "Plaid Hat Games",
            "CMON Global Limited",
            "Gamewright",
            "WizKids",
            "(Public Domain)",
            "Mattel, Inc",
            "Space Cowboys",
            "Stonemaier Games",
            "Plan B Games",
            "Capstone Games",
            "Chip Theory Games",
            "Ares Games",
            "Greater Than Games",
            "Renegade Games",
            "Restoration Games",
            "Osprey Games",
            "Roxley",
            "Czech Games Edition",
            "Awaken Realms",
            "Compass Games",
            "Button Shy",
            "The Game Crafter",
            "Cheapass Games",
            "alea",
            "NorthStar Game Studio",
            "Bézier Games",
            "Red Raven Games",
            "3W (World Wide Wargames)",
        }

        # Family patterns
        self.FAMILY_REMOVE_PATTERNS = [
            "^Admin:",
            "^Misc:",
            "^Promotional:",
            "^Digital Implementations:",
            "^Crowdfunding: Spieleschmiede",
            "^Crowdfunding: Verkami",
            "^Crowdfunding: Indiegogo",
            "^Contests:",
            "^Game:",
            "^Players: Expansions",
            "^Players: Games with expansions",
        ]

        self.FAMILY_ALLOW_PATTERNS = [
            "^Series: Monopoly-Like",
            "^Series: 18xx",
            "^Series: Cards Against Humanity-Like",
            "^Series: Exit: The Game",
            "^Players: Games with Solitaire Rules",
            "^Players: Wargames with Solitaire Rules",
            "^Players: One versus Many",
            "^Players: Solitaire Only Games",
            "^Players: Solitaire Only Wargames",
            "^Players: Two-Player Only Games",
            "^Players: Three Players Only Games",
            "^Players: Wargames with Rules Supporting Only Two Players",
            "^Players: Solitaire Only Card Games",
            "^Country:",
            "^Animals",
            "^History",
            "^Sports",
            "^Category",
            "^Cities",
            "^Traditional",
            "^Creatures",
            "^TV",
            "^Region",
            "^Card",
            "^Comic",
            "^Ancient",
            "^Brands",
            "^Versions & Editions",
            "^Food",
            "^Movies",
            "^Setting",
            "^Card Games",
            "^Collectible",
            "^Containers",
            "^Crowdfunding: Kickstarter",
            "^Crowdfunding: Gamefound",
            "^Authors",
            "^Characters",
            "^Religious",
            "^Holidays",
            "^Space",
            "^Folk",
            "^Word",
            "^Mythology",
            "^Occupation",
            "^Celebrities",
            "^Toys",
        ]

    def __getattr__(self, name: str):
        """Handle missing attributes for backwards compatibility with older pickled models.

        When models are unpickled, they may be missing attributes that were added
        in newer versions of the class. This method provides sensible defaults
        for those attributes.
        """
        # Attributes added in later versions that older pickled models may lack
        backwards_compat_defaults = {
            "family_remove_patterns": None,
            "family_allow_patterns": None,
            "include_description_embeddings": False,
        }

        if name in backwards_compat_defaults:
            return backwards_compat_defaults[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _safe_column_name(self, name: str) -> str:
        """Create a safe column name from a string."""
        # Convert to string and lowercase
        safe_name = str(name).lower()

        # Replace problematic characters
        safe_name = (
            safe_name.replace(" ", "_")  # Replace spaces
            .replace("-", "_")  # Replace hyphens
            .replace(":", "")  # Remove colons
            .replace("/", "_")  # Replace slashes
            .replace("&", "_and_")  # Replace ampersands
            .replace("+", "_plus_")  # Replace plus signs
        )

        # Remove any remaining non-alphanumeric characters except underscores
        safe_name = re.sub(r"[^a-z0-9_]", "", safe_name)

        # Remove consecutive underscores
        safe_name = re.sub(r"_+", "_", safe_name)

        # Ensure the name doesn't start with a number or underscore
        if safe_name[0].isdigit() or safe_name[0] == "_":
            safe_name = "x_" + safe_name.lstrip("_")

        # Remove leading or trailing underscores
        safe_name = safe_name.strip("_")

        return safe_name

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace zeros with NaN."""
        logger = logging.getLogger(__name__)

        df = df.copy()

        # Replace zeros with NaN
        columns_to_replace = [
            "min_age",
            "min_playtime",
            "max_playtime",
            "min_players",
            "max_players",
        ]

        if self.verbose:
            logger.info("Processing missing values:")

        for col in columns_to_replace:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    if self.verbose:
                        logger.info(
                            f"  {col}: replacing {zero_count} zeros with NaN ({(zero_count / len(df)) * 100:.2f}% of values)"
                        )
                df[col] = df[col].replace(0, np.nan)
                nan_count = df[col].isna().sum()
                if nan_count > 0 and self.verbose:
                    logger.info(
                        f"  {col}: now has {nan_count} NaN values ({(nan_count / len(df)) * 100:.2f}% of values)"
                    )

        return df

    def _create_mechanics_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a feature counting the number of mechanics a game has."""
        if "mechanics" not in df.columns:
            return df

        df = df.copy()

        # More robust mechanics count with logging
        def count_mechanics(mechanics):
            # Handle different possible input types
            if mechanics is None:
                return 0
            elif isinstance(mechanics, list):
                return len(mechanics)
            elif hasattr(mechanics, "__array__"):  # Handle numpy arrays
                return len(mechanics.tolist())
            elif isinstance(mechanics, str):
                # In case mechanics is a string representation of a list
                try:
                    return len(eval(mechanics))
                except (TypeError, SyntaxError, NameError):
                    # Handle potential errors in eval
                    return 0
            else:
                # Log unexpected type for debugging
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Unexpected mechanics type: {type(mechanics)}, value: {mechanics}"
                )
                return 0

        df["mechanics_count"] = df["mechanics"].apply(count_mechanics)

        return df

    def _create_categories_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a feature counting the number of categories a game has."""
        if "categories" not in df.columns:
            return df

        df = df.copy()

        def count_categories(categories):
            # Handle different possible input types
            if categories is None:
                return 0
            elif isinstance(categories, list):
                return len(categories)
            elif hasattr(categories, "__array__"):  # Handle numpy arrays
                return len(categories.tolist())
            elif isinstance(categories, str):
                # In case categories is a string representation of a list
                try:
                    return len(eval(categories))
                except (TypeError, SyntaxError, NameError):
                    # Handle potential errors in eval
                    return 0
            else:
                # Log unexpected type for debugging
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Unexpected categories type: {type(categories)}, value: {categories}"
                )
                return 0

        df["categories_count"] = df["categories"].apply(count_categories)

        return df

    def _create_time_per_player(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time per player feature by dividing max_playtime by max_players."""
        df = df.copy()

        # Safely compute time per player, handling potential division by zero or NaN
        def compute_time_per_player(max_playtime, max_players):
            # Return NaN if either max_playtime or max_players is 0 or NaN
            if (
                pd.isna(max_playtime)
                or pd.isna(max_players)
                or max_playtime == 0
                or max_players == 0
            ):
                return np.nan
            return max_playtime / max_players

        df["time_per_player"] = df.apply(
            lambda row: compute_time_per_player(
                row["max_playtime"], row["max_players"]
            ),
            axis=1,
        )

        return df

    def _create_description_word_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a word count feature based on the description field."""
        df = df.copy()

        def count_words(description):
            # Handle None or NaN values
            if pd.isna(description):
                return 0

            # Convert to string to handle potential non-string types
            description_str = str(description)

            # Split on whitespace and count non-empty words
            words = description_str.split()
            return len(words)

        df["description_word_count"] = df["description"].apply(count_words)

        return df

    def _create_player_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for player counts."""
        if not self.create_player_dummies:
            return df

        df = df.copy()

        for count in range(1, self.max_player_count + 1):
            col_name = f"player_count_{count}"
            df[col_name] = (
                (df["min_players"] <= count) & (df["max_players"] >= count)
            ).astype(int)

        return df

    def _fit_array_features(
        self, df: pd.DataFrame, column: str, min_freq: int, max_features: int
    ) -> List[str]:
        """Fit array features and return frequent values."""
        logger = logging.getLogger(__name__)

        if column not in df.columns:
            logger.warning(f"Column {column} not found in DataFrame")
            return []

        # Explode the array column and count frequencies
        all_values = []
        for values in df[column]:
            if isinstance(values, list):
                all_values.extend(values)
            elif hasattr(values, "__array__"):  # Handle numpy arrays
                # Convert to list and extend
                values_list = (
                    values.tolist() if hasattr(values, "tolist") else list(values)
                )
                # Filter out empty strings and None values
                values_list = [v for v in values_list if v and v != ""]
                all_values.extend(values_list)
            else:
                logger.warning(
                    f"Unexpected value type in {column}: {values} (type: {type(values)})"
                )

        # Count frequencies
        value_counts = pd.Series(all_values).value_counts()

        # Filter by frequency and limit
        frequent_values = (
            value_counts[value_counts >= min_freq].head(max_features).index.tolist()
        )

        if self.verbose:
            logger.info(f"Processing {column}:")
            logger.info(f"  Total values: {len(all_values)}")
            logger.info(f"  Unique values: {len(value_counts)}")
            logger.info(f"  Selected {len(frequent_values)} frequent values")

        return frequent_values

    def _create_array_features(
        self, df: pd.DataFrame, column: str, frequent_values: List[str], prefix: str
    ) -> pd.DataFrame:
        """Create one-hot encoded features for array column."""
        if not frequent_values or column not in df.columns:
            return df

        # Create a dictionary to store new columns
        new_columns = {}

        # Initialize all columns with zeros first
        for value in frequent_values:
            if value is not None:
                col_name = f"{prefix}_{self._safe_column_name(value)}"
                new_columns[col_name] = pd.Series(0, index=df.index)

        # Then set 1s only for values that were in the frequent_values list
        for idx, values in df[column].items():
            if isinstance(values, list) or hasattr(values, "__array__"):
                values_list = values if isinstance(values, list) else values.tolist()
                for value in values_list:
                    if value in frequent_values:
                        col_name = f"{prefix}_{self._safe_column_name(value)}"
                        new_columns[col_name][idx] = 1

        # Create a new DataFrame with only the columns from frequent_values
        new_df = pd.DataFrame(new_columns, index=df.index)

        # Return only the new feature columns
        return new_df

    def _filter_publishers(self, publishers: List[str]) -> List[str]:
        """Filter publishers to only allowed ones."""
        if isinstance(publishers, list):
            return [p for p in publishers if p in self.ALLOWED_PUBLISHER_NAMES]
        elif hasattr(publishers, "__array__"):  # Handle numpy arrays
            return [p for p in publishers.tolist() if p in self.ALLOWED_PUBLISHER_NAMES]
        return []

    def _filter_families(self, families: List[str]) -> List[str]:
        """Filter families based on allow/remove patterns."""
        # Convert numpy array to list if needed
        if hasattr(families, "__array__"):
            families = families.tolist()
        elif not isinstance(families, list):
            return []

        # Use instance patterns if provided, otherwise use class defaults
        remove_patterns = self.family_remove_patterns if self.family_remove_patterns is not None else self.FAMILY_REMOVE_PATTERNS
        allow_patterns = self.family_allow_patterns if self.family_allow_patterns is not None else self.FAMILY_ALLOW_PATTERNS

        remove_pattern = re.compile("|".join(remove_patterns))
        allow_pattern = re.compile("|".join(allow_patterns))

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

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the preprocessor.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the preprocessor on.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BaseBGGTransformer
            The fitted preprocessor.
        """
        X = X.copy()

        # Reset all frequent features to None
        self.frequent_categories_ = None
        self.frequent_mechanics_ = None
        self.frequent_designers_ = None
        self.frequent_artists_ = None
        self.frequent_publishers_ = None
        self.frequent_families_ = None

        # Detect embedding columns if enabled
        self.embedding_columns_ = []
        if self.include_description_embeddings:
            self.embedding_columns_ = [col for col in X.columns if col.startswith("emb_")]
            if self.embedding_columns_:
                logger = logging.getLogger(__name__)
                logger.info(f"Detected {len(self.embedding_columns_)} embedding columns")

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
            X_filtered["publishers"] = X_filtered["publishers"].apply(
                self._filter_publishers
            )
            self.frequent_publishers_ = self._fit_array_features(
                X_filtered,
                "publishers",
                self.publisher_min_freq,
                self.max_publisher_features,
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

        # Base numeric features
        if self.include_base_numeric:
            if getattr(self, "include_count_features", False):
                feature_names.extend(
                    ["mechanics_count", "categories_count", "description_word_count"]
                )
            feature_names.extend(
                [
                    "time_per_player",
                    "min_age",
                    "min_playtime",
                    "max_playtime",
                ]
            )

        # Average weight
        if self.include_average_weight:
            feature_names.append("average_weight")

        # Player count features
        if self.create_player_dummies:
            for count in range(1, self.max_player_count + 1):
                feature_names.append(f"player_count_{count}")

        # Array features
        if self.create_category_features and self.frequent_categories_:
            for cat in self.frequent_categories_:
                feature_names.append(f"category_{self._safe_column_name(cat)}")

        if self.create_mechanic_features and self.frequent_mechanics_:
            for mech in self.frequent_mechanics_:
                feature_names.append(f"mechanic_{self._safe_column_name(mech)}")

        if self.create_designer_features and self.frequent_designers_:
            for designer in self.frequent_designers_:
                feature_names.append(f"designer_{self._safe_column_name(designer)}")

        if self.create_artist_features and self.frequent_artists_:
            for artist in self.frequent_artists_:
                feature_names.append(f"artist_{self._safe_column_name(artist)}")

        if self.create_publisher_features and self.frequent_publishers_:
            for pub in self.frequent_publishers_:
                feature_names.append(f"publisher_{self._safe_column_name(pub)}")

        if self.create_family_features and self.frequent_families_:
            for family in self.frequent_families_:
                feature_names.append(f"family_{self._safe_column_name(family)}")

        # Embedding features (passthrough)
        if self.include_description_embeddings and self.embedding_columns_:
            feature_names.extend(self.embedding_columns_)

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
            The transformed data, including engineered features and optionally preserved columns.

        Notes
        -----
        Preserved columns are added after the engineered features. If a preserved
        column is not present in the input DataFrame, it will be skipped.
        """
        if self.feature_names_ is None:
            raise ValueError("Preprocessor must be fitted before transform")

        # Create a copy of input data for base transformations
        X_base = X.copy()

        # Always replace zeros with NaN
        X_base = self._handle_missing_values(X_base)

        # Initialize list to store all feature DataFrames
        feature_dfs = []

        # Create mechanics count if mechanics column exists
        if getattr(self, "include_count_features", False) and "mechanics" in X_base.columns:
            mechanics_df = pd.DataFrame(index=X_base.index)

            # Use the more robust count_mechanics function
            def count_mechanics(mechanics):
                # Handle different possible input types
                if mechanics is None:
                    return 0
                elif isinstance(mechanics, list):
                    return len(mechanics)
                elif hasattr(mechanics, "__array__"):  # Handle numpy arrays
                    return len(mechanics.tolist())
                elif isinstance(mechanics, str):
                    # In case mechanics is a string representation of a list
                    try:
                        return len(eval(mechanics))
                    except (TypeError, SyntaxError, NameError):
                        # Handle potential errors in eval
                        return 0
                else:
                    # Log unexpected type for debugging
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Unexpected mechanics type: {type(mechanics)}, value: {mechanics}"
                    )
                    return 0

            mechanics_df["mechanics_count"] = X_base["mechanics"].apply(count_mechanics)
            feature_dfs.append(mechanics_df)

        # Create categories count if categories column exists
        if getattr(self, "include_count_features", False) and "categories" in X_base.columns:
            categories_df = pd.DataFrame(index=X_base.index)

            def count_categories(categories):
                # Handle different possible input types
                if categories is None:
                    return 0
                elif isinstance(categories, list):
                    return len(categories)
                elif hasattr(categories, "__array__"):  # Handle numpy arrays
                    return len(categories.tolist())
                elif isinstance(categories, str):
                    # In case categories is a string representation of a list
                    try:
                        return len(eval(categories))
                    except (TypeError, SyntaxError, NameError):
                        # Handle potential errors in eval
                        return 0
                else:
                    # Log unexpected type for debugging
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Unexpected categories type: {type(categories)}, value: {categories}"
                    )
                    return 0

            categories_df["categories_count"] = X_base["categories"].apply(
                count_categories
            )
            feature_dfs.append(categories_df)

        # Create time per player feature
        if "max_playtime" in X_base.columns and "max_players" in X_base.columns:
            time_per_player_df = pd.DataFrame(index=X_base.index)

            def compute_time_per_player(max_playtime, max_players):
                # Check for NaN or zero max_players to avoid division by zero
                if pd.isna(max_playtime) or pd.isna(max_players) or max_players == 0:
                    return np.nan
                return max_playtime / max_players

            time_per_player_df["time_per_player"] = X_base.apply(
                lambda row: compute_time_per_player(
                    row["max_playtime"], row["max_players"]
                ),
                axis=1,
            )
            feature_dfs.append(time_per_player_df)

        # Create description word count feature (only if include_count_features is True)
        if (
            "description" in X_base.columns
            and getattr(self, "include_count_features", False)
        ):
            description_word_count_df = pd.DataFrame(index=X_base.index)

            def count_words(description):
                # Handle None or NaN values
                if pd.isna(description):
                    return 0

                # Convert to string to handle potential non-string types
                description_str = str(description)

                # Split on whitespace and count non-empty words
                words = description_str.split()
                return len(words)

            description_word_count_df["description_word_count"] = X_base[
                "description"
            ].apply(count_words)
            feature_dfs.append(description_word_count_df)

        # Create player dummies if enabled
        if self.create_player_dummies:
            player_dummies = pd.DataFrame(index=X_base.index)
            for count in range(1, self.max_player_count + 1):
                player_dummies[f"player_count_{count}"] = (
                    (X_base["min_players"] <= count) & (X_base["max_players"] >= count)
                ).astype(int)
            feature_dfs.append(player_dummies)

        # Create array features
        array_features = [
            (
                "categories",
                self.create_category_features,
                self.frequent_categories_,
                "category",
                None,
            ),
            (
                "mechanics",
                self.create_mechanic_features,
                self.frequent_mechanics_,
                "mechanic",
                None,
            ),
            (
                "designers",
                self.create_designer_features,
                self.frequent_designers_,
                "designer",
                None,
            ),
            (
                "artists",
                self.create_artist_features,
                self.frequent_artists_,
                "artist",
                None,
            ),
            (
                "publishers",
                self.create_publisher_features,
                self.frequent_publishers_,
                "publisher",
                self._filter_publishers,
            ),
            (
                "families",
                self.create_family_features,
                self.frequent_families_,
                "family",
                self._filter_families,
            ),
        ]

        for col, enabled, frequent_values, prefix, filter_func in array_features:
            if enabled and frequent_values and col in X_base.columns:
                X_filtered = X_base.copy() if filter_func else X_base
                if filter_func:
                    X_filtered[col] = X_filtered[col].apply(filter_func)
                feature_df = self._create_array_features(
                    X_filtered, col, frequent_values, prefix
                )
                if not feature_df.empty:
                    feature_dfs.append(feature_df)

        # Add base numeric features if enabled
        if self.include_base_numeric:
            numeric_features = ["min_age", "min_playtime", "max_playtime"]
            base_numeric_df = pd.DataFrame(index=X_base.index)
            for col in numeric_features:
                if col in X_base.columns:
                    base_numeric_df[col] = X_base[col]
            if not base_numeric_df.empty:
                feature_dfs.append(base_numeric_df)

        # Add average weight if enabled
        if self.include_average_weight and "average_weight" in X_base.columns:
            weight_df = pd.DataFrame({"average_weight": X_base["average_weight"]})
            feature_dfs.append(weight_df)

        # Add embedding columns if enabled (passthrough unchanged)
        if self.include_description_embeddings and self.embedding_columns_:
            available_emb_cols = [col for col in self.embedding_columns_ if col in X_base.columns]
            if available_emb_cols:
                embedding_df = X_base[available_emb_cols].copy()
                feature_dfs.append(embedding_df)

        # Concatenate all feature DataFrames
        if feature_dfs:
            result = pd.concat(feature_dfs, axis=1)
            # Ensure we only return the features we want in the correct order
            available_features = [
                col for col in self.feature_names_ if col in result.columns
            ]
            result = result[available_features]

            # Prepare preserved columns with guaranteed order
            guaranteed_cols = ["year_published"]
            preserved_cols = []

            # Always check for guaranteed columns first
            for col in guaranteed_cols:
                if col in X_base.columns and col not in preserved_cols:
                    preserved_cols.append(col)

            # Then add any additional columns from preserve_columns
            for col in self.preserve_columns:
                if col in X_base.columns and col not in preserved_cols:
                    preserved_cols.append(col)

            # Combine preserved columns and engineered features
            if preserved_cols:
                preserved_data = X_base[preserved_cols]
                result = pd.concat([preserved_data, result], axis=1)

            # Log NaN information
            if self.verbose:
                logger = logging.getLogger(__name__)
                logger.info("NaN value counts in transformed features:")
                nan_counts = result.isna().sum()
                nan_columns = nan_counts[nan_counts > 0]
                if not nan_columns.empty:
                    for col, count in nan_columns.items():
                        logger.info(
                            f"  {col}: {count} NaN values ({(count / len(result)) * 100:.2f}%)"
                        )
                else:
                    logger.info("  No NaN values found in transformed features")

            return result
        else:
            # Return empty DataFrame with correct columns if no features were created
            result = pd.DataFrame(columns=self.feature_names_, index=X_base.index)

            # Prepare preserved columns
            preserved_cols = []
            for col in self.preserve_columns:
                if col in X_base.columns:
                    preserved_cols.append(col)

            # Add preserved columns first
            if preserved_cols:
                result = X_base[preserved_cols].copy()

            return result

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

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get the list of feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present for API consistency by convention.

        Returns
        -------
        List[str]
            The list of feature names.
        """
        if self.feature_names_ is None:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        return self.feature_names_.copy()

    def set_output(self, *, transform=None):
        self._output_config = transform
        return self


class DescriptionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X might be a Series or DataFrame with one column
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        X = X.fillna("")  # Replace None/NaN with empty strings

        embeddings = self.model.encode(X.tolist(), show_progress_bar=False)
        return np.array(embeddings)

    def set_output(self, *, transform=None):
        self._output_config = transform
        return self

    def get_feature_names_out(self, input_features=None):
        return [f"embed_{i}" for i in range(384)]
