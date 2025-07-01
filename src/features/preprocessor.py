"""Data preprocessing for BGG predictive models."""
from typing import Dict, List, Optional, Tuple, Union
import re

import polars as pl


class BaseTransformer:
    """Base class for all data transformers."""
    
    def fit(self, df: pl.DataFrame) -> 'BaseTransformer':
        """Fit transformer to data (if needed)."""
        return self
        
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform the data."""
        raise NotImplementedError("Subclasses must implement transform()")
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class MissingValueTransformer(BaseTransformer):
    """Replace zeros with nulls in specific columns."""
    
    def __init__(self, columns_to_replace: List[str] = None):
        self.columns_to_replace = columns_to_replace or [
            "year_published", "average_weight", "average_rating", 
            "min_age", "min_playtime", "max_playtime", 
            "min_players", "max_players"
        ]
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Replace zeros with nulls in specified columns."""
        expressions = []
        for col in self.columns_to_replace:
            if col in df.columns:
                expressions.append(
                    pl.when(pl.col(col) == 0).then(None).otherwise(pl.col(col)).alias(col)
                )
        
        if expressions:
            return df.with_columns(expressions)
        return df


class YearTransformer(BaseTransformer):
    """Transform year_published variable."""
    
    def __init__(
        self, 
        reference_year: int = 2000, 
        normalization_factor: int = 25,
        transformation_strategy: str = 'normalized_log'
    ):
        """Initialize transformer.
        
        Args:
            reference_year: Year to center around (default 2000)
            normalization_factor: Divisor for normalization (default 25)
            transformation_strategy: Strategy for year transformation
                Options:
                - 'normalized_log': Normalized and log-transformed distance from reference year
                - 'era_bins': Categorical binning of years
                - 'centered': Simple centering around reference year
                - 'quadratic': Quadratic transformation
                - 'cubic': Cubic transformation
        """
        self.reference_year = reference_year
        self.normalization_factor = normalization_factor
        self.transformation_strategy = transformation_strategy
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform year_published into selected feature."""
        if "year_published" not in df.columns:
            return df
        
        # Remove rows with missing year_published
        total_rows_before = len(df)
        result_df = df.filter(pl.col("year_published").is_not_null())
        rows_removed = total_rows_before - len(result_df)
        
        if rows_removed > 0:
            print(f"Removed {rows_removed} games with missing year_published (out of {total_rows_before} total games)")
        
        # Base features
        base_features = [
            (pl.col("year_published") - self.reference_year).alias("year_published_centered"),
            ((pl.col("year_published") - self.reference_year) / self.normalization_factor).alias("year_published_normalized"),
        ]
        
        # Strategy-specific transformations
        if self.transformation_strategy == 'normalized_log':
            # Log-transformed normalized distance
            log_feature = (
                pl.when(pl.col("year_published") <= self.reference_year)
                .then((self.reference_year - pl.col("year_published") + 1).log())
                .otherwise((pl.col("year_published") - self.reference_year + 1).log())
                .alias("year_published_transformed")
            )
            base_features.append(log_feature)
        
        elif self.transformation_strategy == 'era_bins':
            # Categorical binning
            era_feature = (
                pl.when(pl.col("year_published") < 1900)
                .then(pl.lit("pre_1900"))
                .when(pl.col("year_published") < 1950)
                .then(pl.lit("1900_1950"))
                .when(pl.col("year_published") < 1975)
                .then(pl.lit("1950_1975"))
                .when(pl.col("year_published") < 1990)
                .then(pl.lit("1975_1990"))
                .when(pl.col("year_published") < 2000)
                .then(pl.lit("1990_2000"))
                .when(pl.col("year_published") < 2010)
                .then(pl.lit("2000_2010"))
                .when(pl.col("year_published") < 2020)
                .then(pl.lit("2010_2020"))
                .when(pl.col("year_published") < 2030)
                .then(pl.lit("2020_2030"))
                .otherwise(pl.lit("post_2030"))
                .alias("year_published_transformed")
            )
            base_features.append(era_feature)
        
        elif self.transformation_strategy == 'quadratic':
            # Quadratic transformation
            quad_feature = (
                ((pl.col("year_published") - self.reference_year) ** 2 / (self.normalization_factor ** 2))
                .alias("year_published_transformed")
            )
            base_features.append(quad_feature)
        
        elif self.transformation_strategy == 'cubic':
            # Cubic transformation
            cubic_feature = (
                ((pl.col("year_published") - self.reference_year) ** 3 / (self.normalization_factor ** 3))
                .alias("year_published_transformed")
            )
            base_features.append(cubic_feature)
        
        # Apply features
        result_df = result_df.with_columns(base_features)
        
        return result_df


class MechanicsCountTransformer(BaseTransformer):
    """Add a feature counting the number of mechanics a game has."""
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add mechanics count feature."""
        if "mechanics" not in df.columns:
            return df
        
        result_df = df.with_columns([
            pl.col("mechanics").list.len().alias("mechanics_count")
        ])
        
        return result_df


class PlayerCountTransformer(BaseTransformer):
    """Create dummy variables for player counts."""
    
    def __init__(self, max_count: int = 20):
        self.max_count = max_count
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create player count dummy variables."""
        result_df = df.clone()
        
        # For each possible player count (1 to max_count)
        for count in range(1, self.max_count + 1):
            # Create a dummy variable indicating if the game supports this player count
            col_name = f"player_count_{count}"
            result_df = result_df.with_columns([
                pl.when((pl.col("min_players") <= count) & (pl.col("max_players") >= count))
                  .then(1)
                  .otherwise(0)
                  .alias(col_name)
            ])
        
        return result_df


class ArrayFeatureTransformer(BaseTransformer):
    """One-hot encode array columns like categories, mechanics, designers, artists."""
    
    def __init__(
        self, 
        column: str, 
        min_freq: int = 10,
        max_features: int = 50,
        feature_prefix: str = None
    ):
        """Initialize transformer.
        
        Args:
            column: Name of the array column to transform
            min_freq: Minimum frequency to include a value
            max_features: Maximum number of features to generate
            feature_prefix: Prefix for feature names (defaults to column name)
        """
        self.column = column
        self.min_freq = min_freq
        self.max_features = max_features
        self.feature_prefix = feature_prefix or column
        self.frequent_values = None
    
    def fit(self, df: pl.DataFrame) -> 'ArrayFeatureTransformer':
        """Find frequent values in the array column."""
        if self.column not in df.columns:
            return self
            
        # Explode the array column
        values = df.select(pl.col(self.column)).explode(self.column)
        
        # Count frequencies
        value_counts = values.group_by(self.column).count()
        
        # Select top features by frequency
        self.frequent_values = (
            value_counts
            .filter(pl.col("count") >= self.min_freq)
            .sort("count", descending=True)
            .head(self.max_features)
            .select(self.column)
            .to_series()
            .to_list()
        )
        
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """One-hot encode the array column."""
        if self.column not in df.columns or not self.frequent_values:
            return df
            
        result_df = df.clone()
        
        for value in self.frequent_values:
            # Safely handle various potential issues
            if value is not None:
                # Create safe column name
                safe_value = str(value).lower().replace(' ', '_').replace('-', '_')
                col_name = f"{self.feature_prefix}_{safe_value}"
                
                result_df = result_df.with_columns([
                    pl.when(
                        (pl.col(self.column).is_not_null()) & 
                        (pl.col(self.column).list.len() > 0)
                    )
                    .then(
                        pl.col(self.column)
                        .list.contains(value)
                        .fill_null(False)
                        .cast(pl.Int8)
                    )
                    .otherwise(0)
                    .alias(col_name)
                ])
        
        return result_df


class PublisherTransformer(BaseTransformer):
    """One-hot encode publishers with an allow list."""
    
    # Publisher IDs from the allow list
    PUBLISHER_ALLOW_LIST = [
        51, 10, 102, 196, 396, 1027, 21847, 1001, 4, 140, 157, 34, 28, 39, 37, 20, 3, 538, 52, 
        17, 5, 3320, 597, 5400, 26, 47, 11652, 19, 13, 12024, 10754, 21608, 108, 221, 171, 93, 
        25842, 23202, 34188, 30958, 22593, 17917, 17543, 28072, 34846, 29313, 21765, 7345, 
        29412, 3929, 26991, 2456, 12, 9, 2164, 5774, 18617, 102, 489
    ]
    
    # Mapping of known publisher IDs to names
    # This would need to be populated from the database
    PUBLISHER_ID_TO_NAME = {
        51: "Hasbro",
        10: "Mayfair Games",
        102: "Decision Games",
        196: "Multi-Man Publishing",
        396: "Alderac Entertainment Group",
        1027: "Days of Wonder",
        21847: "Pandasaurus Games",
        1001: "(web published)",
        4: "(Self-Published)",
        140: "Splotter Spellen",
        157: "Asmodee",
        34: "Ravensburger",
        28: "Parker Brothers",
        39: "Pegasus Spiele",
        37: "KOSMOS",
        20: "Milton Bradley",
        3: "Rio Grande Games",
        538: "Z-Man Games",
        52: "GMT Games",
        17: "Fantasy Flight Games",
        5: "Avalon Hill",
        3320: "(Unknown)",
        597: "Eagle-Gryphon Games",
        5400: "Matagot",
        26: "Games Workshop Ltd",
        47: "Queen Games",
        11652: "Stronghold Games",
        19: "Steve Jackson Games",
        13: "Wizards of the Coast",
        12024: "Cryptozoic Entertainment",
        10754: "Plaid Hat Games",
        21608: "CMON Global Limited",
        108: "Gamewright",
        221: "WizKids",
        171: "(Public Domain)",
        93: "Mattel, Inc",
        25842: "Space Cowboys",
        23202: "Stonemaier Games",
        34188: "Plan B Games",
        30958: "Capstone Games",
        22593: "Chip Theory Games",
        17917: "Ares Games",
        17543: "Greater Than Games",
        28072: "Renegade Games",
        34846: "Restoration Games",
        29313: "Osprey Games",
        21765: "Roxley",
        7345: "Czech Games Edition",
        29412: "Awaken Realms",
        3929: "Compass Games",
        26991: "Button Shy",
        2456: "The Game Crafter",
        12: "Cheapass Games",
        9: "alea",
        2164: "NorthStar Game Studio",
        5774: "Bézier Games",
        18617: "Red Raven Games",
        489: "3W (World Wide Wargames)",
    }
    
    # Create a set of allowed publisher names for faster lookup
    ALLOWED_PUBLISHER_NAMES = set(PUBLISHER_ID_TO_NAME.values())
    
    def __init__(
        self, 
        column: str = "publishers",
        min_freq: int = 10,
        max_features: int = 50,
        feature_prefix: str = "publisher"
    ):
        """Initialize transformer.
        
        Args:
            column: Name of the column containing publisher names
            min_freq: Minimum frequency to include a publisher
            max_features: Maximum number of publisher features to generate
            feature_prefix: Prefix for feature names
        """
        self.column = column
        self.min_freq = min_freq
        self.max_features = max_features
        self.feature_prefix = feature_prefix
        self.frequent_publishers = None
    
    def fit(self, df: pl.DataFrame) -> 'PublisherTransformer':
        """Find frequent publishers that are in the allow list."""
        if self.column not in df.columns:
            return self
        
        # Explode the array column
        values = df.select(pl.col(self.column)).explode(self.column)
        
        # Filter to allowed publishers
        allowed_publishers = []
        for publisher in values[self.column]:
            if publisher is not None and publisher in self.ALLOWED_PUBLISHER_NAMES:
                allowed_publishers.append(publisher)
        
        # Count frequencies
        publisher_df = pl.DataFrame({"publisher": allowed_publishers})
        publisher_counts = publisher_df.group_by("publisher").count()
        
        # Select top features by frequency
        self.frequent_publishers = (
            publisher_counts
            .filter(pl.col("count") >= self.min_freq)
            .sort("count", descending=True)
            .head(self.max_features)
            .select("publisher")
            .to_series()
            .to_list()
        )
        
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """One-hot encode the allowed publishers."""
        if self.column not in df.columns or not self.frequent_publishers:
            return df
            
        result_df = df.clone()
        
        for publisher in self.frequent_publishers:
            # Safely handle various potential issues
            if publisher is not None:
                # Create safe column name
                safe_value = str(publisher).lower().replace(' ', '_').replace('-', '_')
                col_name = f"{self.feature_prefix}_{safe_value}"
                
                result_df = result_df.with_columns([
                    pl.when(
                        (pl.col(self.column).is_not_null()) & 
                        (pl.col(self.column).list.len() > 0)
                    )
                    .then(
                        pl.col(self.column)
                        .list.contains(publisher)
                        .fill_null(False)
                        .cast(pl.Int8)
                    )
                    .otherwise(0)
                    .alias(col_name)
                ])
        
        return result_df


class FamilyTransformer(BaseTransformer):
    """One-hot encode families with allow and remove lists."""
    
    # Patterns to remove
    FAMILY_REMOVE_PATTERNS = [
        "^Admin:", "^Misc:", "^Promotional:", "^Digital Implementations:",
        "^Crowdfunding: Spieleschmiede", "^Crowdfunding: Verkami", 
        "^Crowdfunding: Indiegogo", "^Contests:", "^Game:",
        "^Players: Expansions", "^Players: Games with expansions"
    ]
    
    # Patterns to allow
    FAMILY_ALLOW_PATTERNS = [
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
    
    def __init__(
        self, 
        column: str = "families",
        min_freq: int = 10,
        max_features: int = 50,
        feature_prefix: str = "family"
    ):
        """Initialize transformer.
        
        Args:
            column: Name of the column containing families
            min_freq: Minimum frequency to include a family
            max_features: Maximum number of family features to generate
            feature_prefix: Prefix for feature names
        """
        self.column = column
        self.min_freq = min_freq
        self.max_features = max_features
        self.feature_prefix = feature_prefix
        self.frequent_families = None
        
        # Compile regex patterns
        self.remove_pattern = re.compile("|".join(self.FAMILY_REMOVE_PATTERNS))
        self.allow_pattern = re.compile("|".join(self.FAMILY_ALLOW_PATTERNS))
    
    def fit(self, df: pl.DataFrame) -> 'FamilyTransformer':
        """Find frequent families that match the allow list and not the remove list."""
        if self.column not in df.columns:
            return self
            
        # Explode the array column
        values = df.select(pl.col(self.column)).explode(self.column)
        
        # Filter to allowed families
        allowed_families = []
        for family in values[self.column]:
            if family is not None:
                # Skip if it matches the remove pattern
                if self.remove_pattern.search(family):
                    continue
                
                # Include if it matches the allow pattern
                if self.allow_pattern.search(family):
                    allowed_families.append(family)
        
        # Count frequencies
        family_df = pl.DataFrame({"family": allowed_families})
        family_counts = family_df.group_by("family").count()
        
        # Select top features by frequency
        self.frequent_families = (
            family_counts
            .filter(pl.col("count") >= self.min_freq)
            .sort("count", descending=True)
            .head(self.max_features)
            .select("family")
            .to_series()
            .to_list()
        )
        
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """One-hot encode the allowed families."""
        if self.column not in df.columns or not self.frequent_families:
            return df
            
        result_df = df.clone()
        
        for family in self.frequent_families:
            # Safely handle various potential issues
            if family is not None:
                # Create safe column name
                safe_value = str(family).lower().replace(' ', '_').replace('-', '_').replace(':', '')
                col_name = f"{self.feature_prefix}_{safe_value}"
                
                result_df = result_df.with_columns([
                    pl.when(
                        (pl.col(self.column).is_not_null()) & 
                        (pl.col(self.column).list.len() > 0)
                    )
                    .then(
                        pl.col(self.column)
                        .list.contains(family)
                        .fill_null(False)
                        .cast(pl.Int8)
                    )
                    .otherwise(0)
                    .alias(col_name)
                ])
        
        return result_df


class BGGPreprocessingPipeline:
    """Pipeline for preprocessing BGG data."""
    
    def __init__(self, transformers: List[BaseTransformer] = None):
        """Initialize with list of transformers."""
        self.transformers = transformers or []
    
    def add_transformer(self, transformer: BaseTransformer) -> 'BGGPreprocessingPipeline':
        """Add a transformer to the pipeline."""
        self.transformers.append(transformer)
        return self
    
    def fit(self, df: pl.DataFrame) -> 'BGGPreprocessingPipeline':
        """Fit all transformers to data."""
        for transformer in self.transformers:
            transformer.fit(df)
        return self
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all transformers in sequence."""
        result = df.clone()
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class BGGPreprocessor:
    """Enhanced preprocessor for BGG data."""
    
    def __init__(
        self,
        # Feature generation parameters
        max_player_count: int = 10,
        
        # Year transformation parameters
        reference_year: int = 2000,
        normalization_factor: int = 25,
        
        # Array feature parameters
        category_min_freq: int = 0,
        mechanic_min_freq: int = 0,
        designer_min_freq: int = 10,
        artist_min_freq: int = 10,
        publisher_min_freq: int = 10,
        family_min_freq: int = 10,
        max_category_features: int = 250,
        max_mechanic_features: int = 250,
        max_designer_features: int = 100,
        max_artist_features: int = 100,
        max_publisher_features: int = 100,
        max_family_features: int = 100,
        
        # Feature generation flags
        handle_missing_values: bool = True,
        transform_year: bool = True,
        create_player_dummies: bool = True,
        create_category_mechanic_features: bool = False,
        create_designer_artist_features: bool = False,
        create_publisher_features: bool = False,
        create_family_features: bool = False,
        
        # Feature selection parameters
        include_base_numeric: bool = True,
        include_average_weight: bool = False,  # New parameter for average_weight
        include_player_count: bool = True,
        include_categories: bool = True,
        include_mechanics: bool = True,
        include_designers: bool = False,
        include_artists: bool = False,
        include_publishers: bool = False,
        include_families: bool = False,
        custom_feature_patterns: List[str] = None,
        exclude_feature_patterns: List[str] = None,
        always_include_columns: List[str] = None,
        always_exclude_columns: List[str] = None,
    ):
        """Initialize preprocessor with configuration options."""
        # Store feature selection parameters
        self.include_base_numeric = include_base_numeric
        self.include_average_weight = include_average_weight  # Store the new parameter
        self.include_player_count = include_player_count
        self.include_categories = include_categories
        self.include_mechanics = include_mechanics
        self.include_designers = include_designers
        self.include_artists = include_artists
        self.include_publishers = include_publishers
        self.include_families = include_families
        self.custom_feature_patterns = custom_feature_patterns or []
        self.exclude_feature_patterns = exclude_feature_patterns or []
        self.always_include_columns = always_include_columns or ["id", "name"]
        self.always_exclude_columns = always_exclude_columns or []
        
        # Feature registry to track created features
        self.feature_registry = {
            "base_numeric": [],
            "average_weight": [],  # New category for average_weight
            "player_count": [],
            "categories": [],
            "mechanics": [],
            "designers": [],
            "artists": [],
            "publishers": [],
            "families": [],
            "custom": [],
        }
        
        self.pipeline = BGGPreprocessingPipeline()
        
        # Add transformers based on configuration
        if handle_missing_values:
            self.pipeline.add_transformer(MissingValueTransformer())
        
        if transform_year:
            self.pipeline.add_transformer(YearTransformer(
                reference_year=reference_year,
                normalization_factor=normalization_factor
            ))
        
        # Always add mechanics count transformer
        self.pipeline.add_transformer(MechanicsCountTransformer())
        
        if create_player_dummies:
            self.pipeline.add_transformer(PlayerCountTransformer(max_count=max_player_count))
        
        if create_category_mechanic_features:
            self.pipeline.add_transformer(
                ArrayFeatureTransformer(
                    column="categories", 
                    min_freq=category_min_freq, 
                    max_features=max_category_features,
                    feature_prefix="category"
                )
            )
            self.pipeline.add_transformer(
                ArrayFeatureTransformer(
                    column="mechanics", 
                    min_freq=mechanic_min_freq, 
                    max_features=max_mechanic_features,
                    feature_prefix="mechanic"
                )
            )
        
        # Designer features
        if create_designer_artist_features:
            self.pipeline.add_transformer(
                ArrayFeatureTransformer(
                    column="designers", 
                    min_freq=designer_min_freq, 
                    max_features=max_designer_features,
                    feature_prefix="designer"
                )
            )
            self.pipeline.add_transformer(
                ArrayFeatureTransformer(
                    column="artists", 
                    min_freq=artist_min_freq, 
                    max_features=max_artist_features,
                    feature_prefix="artist"
                )
            )
        
        if create_publisher_features:
            self.pipeline.add_transformer(
                PublisherTransformer(
                    column="publishers",
                    min_freq=publisher_min_freq,
                    max_features=max_publisher_features,
                    feature_prefix="publisher"
                )
            )
        
        if create_family_features:
            self.pipeline.add_transformer(
                FamilyTransformer(
                    column="families",
                    min_freq=family_min_freq,
                    max_features=max_family_features,
                    feature_prefix="family"
                )
            )
    
    def fit(self, df: pl.DataFrame) -> 'BGGPreprocessor':
        """Fit the preprocessor to the data."""
        self.pipeline.fit(df)
        return self
    
    def _update_feature_registry(self, df: pl.DataFrame):
        """Update the feature registry based on the processed DataFrame."""
        # Base numeric features - only include fields that are in the DataFrame
        base_numeric = [
            "year_published_transformed", "year_published_centered", 
            "year_published_normalized", "mechanics_count",
            "min_age", "min_playtime", "max_playtime"
        ]
        self.feature_registry["base_numeric"] = [col for col in base_numeric if col in df.columns]
        
        # Average weight feature - only include if it's in the DataFrame
        if "average_weight" in df.columns:
            self.feature_registry["average_weight"] = ["average_weight"]
        else:
            self.feature_registry["average_weight"] = []
        
        # Player count features
        self.feature_registry["player_count"] = [
            col for col in df.columns if col.startswith("player_count_")
        ]
        
        # Category features
        self.feature_registry["categories"] = [
            col for col in df.columns if col.startswith("category_")
        ]
        
        # Mechanic features
        self.feature_registry["mechanics"] = [
            col for col in df.columns if col.startswith("mechanic_")
        ]
        
        # Designer features
        self.feature_registry["designers"] = [
            col for col in df.columns if col.startswith("designer_")
        ]
        
        # Artist features
        self.feature_registry["artists"] = [
            col for col in df.columns if col.startswith("artist_")
        ]
        
        # Publisher features
        self.feature_registry["publishers"] = [
            col for col in df.columns if col.startswith("publisher_")
        ]
        
        # Family features
        self.feature_registry["families"] = [
            col for col in df.columns if col.startswith("family_")
        ]
        
        # Custom features
        for pattern in self.custom_feature_patterns:
            matching_cols = [col for col in df.columns if re.search(pattern, col)]
            self.feature_registry["custom"].extend(matching_cols)
    
    def _get_selected_features(self) -> List[str]:
        """Get the list of selected features based on configuration."""
        selected_features = []
        
        # Add features based on configuration
        if self.include_base_numeric:
            selected_features.extend(self.feature_registry["base_numeric"])
        
        if self.include_average_weight:
            selected_features.extend(self.feature_registry["average_weight"])
        
        if self.include_player_count:
            selected_features.extend(self.feature_registry["player_count"])
        
        if self.include_categories:
            selected_features.extend(self.feature_registry["categories"])
        
        if self.include_mechanics:
            selected_features.extend(self.feature_registry["mechanics"])
        
        if self.include_designers:
            selected_features.extend(self.feature_registry["designers"])
        
        if self.include_artists:
            selected_features.extend(self.feature_registry["artists"])
        
        if self.include_publishers:
            selected_features.extend(self.feature_registry["publishers"])
        
        if self.include_families:
            selected_features.extend(self.feature_registry["families"])
        
        # Add custom features
        selected_features.extend(self.feature_registry["custom"])
        
        # Add always-include columns
        selected_features.extend(self.always_include_columns)
        
        # Remove duplicates
        selected_features = list(dict.fromkeys(selected_features))
        
        # Remove excluded features
        for pattern in self.exclude_feature_patterns:
            selected_features = [col for col in selected_features if not re.search(pattern, col)]
        
        # Remove always-exclude columns
        selected_features = [col for col in selected_features if col not in self.always_exclude_columns]
        
        return selected_features
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """Get all available features by category."""
        return self.feature_registry
    
    def get_selected_features(self) -> List[str]:
        """Get the list of selected features based on current configuration."""
        return self._get_selected_features()
    
    def transform(
        self, 
        df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, Dict[str, pl.Series]]:
        """Transform raw data into features and targets."""
        # Apply preprocessing pipeline
        processed = self.pipeline.transform(df)
        
        # Update feature registry
        self._update_feature_registry(processed)
        
        # Get selected features based on configuration
        feature_columns = self._get_selected_features()
        
        # Filter to only include columns that exist in the processed DataFrame
        valid_columns = [col for col in feature_columns if col in processed.columns]
        
        # Select features
        features = processed.select(valid_columns)
        
        # Create targets using pre-computed columns from SQL
        targets = {
            "hurdle": df.select("hurdle").to_series(),
            "complexity": df.select("complexity").to_series(),
            "rating": df.select("rating").to_series(),
            "users_rated": df.select("log_users_rated").to_series()
        }
        
        return features, targets
    
    def fit_transform(
        self, 
        df: pl.DataFrame,
    ) -> Tuple[pl.DataFrame, Dict[str, pl.Series]]:
        """Fit to data and transform in one step."""
        return self.fit(df).transform(df)
