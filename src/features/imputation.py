"""Imputation transformers for BGG predictive models."""
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class BGGImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in BGG data.
    
    This transformer imputes missing values in numeric columns using different strategies
    based on the column type. It's designed to be used in a scikit-learn pipeline.
    
    Parameters
    ----------
    numeric_strategy : str, default='median'
        The imputation strategy for numeric columns. Options are:
        - 'mean': Replace missing values with the mean of the column.
        - 'median': Replace missing values with the median of the column.
        - 'most_frequent': Replace missing values with the most frequent value in the column.
        - 'constant': Replace missing values with a constant value.
    
    numeric_fill_value : float, optional
        The constant value to use when numeric_strategy is 'constant'.
    
    categorical_strategy : str, default='most_frequent'
        The imputation strategy for categorical columns. Options are:
        - 'most_frequent': Replace missing values with the most frequent value in the column.
        - 'constant': Replace missing values with a constant value.
    
    categorical_fill_value : str, optional
        The constant value to use when categorical_strategy is 'constant'.
    
    numeric_columns : List[str], optional
        List of numeric columns to impute. If None, all numeric columns will be imputed.
    
    categorical_columns : List[str], optional
        List of categorical columns to impute. If None, all categorical columns will be imputed.
    
    Attributes
    ----------
    numeric_imputer_ : SimpleImputer
        The imputer for numeric columns.
    
    categorical_imputer_ : SimpleImputer
        The imputer for categorical columns.
    
    numeric_columns_ : List[str]
        The list of numeric columns that were imputed.
    
    categorical_columns_ : List[str]
        The list of categorical columns that were imputed.
    """
    
    def __init__(
        self,
        numeric_strategy: str = 'median',
        numeric_fill_value: Optional[float] = None,
        categorical_strategy: str = 'most_frequent',
        categorical_fill_value: Optional[str] = None,
        numeric_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
    ):
        self.numeric_strategy = numeric_strategy
        self.numeric_fill_value = numeric_fill_value
        self.categorical_strategy = categorical_strategy
        self.categorical_fill_value = categorical_fill_value
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer on.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : BGGImputer
            The fitted imputer.
        """
        # Determine numeric and categorical columns if not provided
        if self.numeric_columns is None:
            self.numeric_columns_ = X.select_dtypes(include=['number']).columns.tolist()
        else:
            self.numeric_columns_ = self.numeric_columns
        
        if self.categorical_columns is None:
            self.categorical_columns_ = X.select_dtypes(exclude=['number']).columns.tolist()
        else:
            self.categorical_columns_ = self.categorical_columns
        
        # Create and fit the numeric imputer
        if self.numeric_columns_:
            self.numeric_imputer_ = SimpleImputer(
                strategy=self.numeric_strategy,
                fill_value=self.numeric_fill_value
            )
            self.numeric_imputer_.fit(X[self.numeric_columns_])
        
        # Create and fit the categorical imputer
        if self.categorical_columns_:
            self.categorical_imputer_ = SimpleImputer(
                strategy=self.categorical_strategy,
                fill_value=self.categorical_fill_value
            )
            self.categorical_imputer_.fit(X[self.categorical_columns_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to impute.
        
        Returns
        -------
        pd.DataFrame
            The imputed data.
        """
        X_transformed = X.copy()
        
        # Impute numeric columns
        if hasattr(self, 'numeric_imputer_') and self.numeric_columns_:
            numeric_cols_to_impute = [col for col in self.numeric_columns_ if col in X.columns]
            if numeric_cols_to_impute:
                X_numeric_imputed = self.numeric_imputer_.transform(X[numeric_cols_to_impute])
                X_transformed[numeric_cols_to_impute] = X_numeric_imputed
        
        # Impute categorical columns
        if hasattr(self, 'categorical_imputer_') and self.categorical_columns_:
            cat_cols_to_impute = [col for col in self.categorical_columns_ if col in X.columns]
            if cat_cols_to_impute:
                X_cat_imputed = self.categorical_imputer_.transform(X[cat_cols_to_impute])
                X_transformed[cat_cols_to_impute] = X_cat_imputed
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit the imputer on X and then transform X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer on and then impute.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        pd.DataFrame
            The imputed data.
        """
        return self.fit(X, y).transform(X)


class ColumnSpecificImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in specific columns with different strategies.
    
    This transformer allows for different imputation strategies for different columns.
    It's designed to be used in a scikit-learn pipeline.
    
    Parameters
    ----------
    imputation_strategies : Dict[str, Dict[str, Union[str, float, int]]]
        A dictionary mapping column names to their imputation strategies.
        Each strategy is a dictionary with the following keys:
        - 'strategy': The imputation strategy. Options are:
            - 'mean': Replace missing values with the mean of the column.
            - 'median': Replace missing values with the median of the column.
            - 'most_frequent': Replace missing values with the most frequent value in the column.
            - 'constant': Replace missing values with a constant value.
        - 'fill_value': The constant value to use when strategy is 'constant'.
    
    Attributes
    ----------
    imputers_ : Dict[str, SimpleImputer]
        A dictionary mapping column names to their fitted imputers.
    """
    
    def __init__(self, imputation_strategies: Dict[str, Dict[str, Union[str, float, int]]]):
        self.imputation_strategies = imputation_strategies
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the imputer on X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer on.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : ColumnSpecificImputer
            The fitted imputer.
        """
        self.imputers_ = {}
        
        for column, strategy_dict in self.imputation_strategies.items():
            if column in X.columns:
                strategy = strategy_dict.get('strategy', 'mean')
                fill_value = strategy_dict.get('fill_value', None)
                
                imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                # Reshape to 2D array for SimpleImputer
                imputer.fit(X[[column]])
                self.imputers_[column] = imputer
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to impute.
        
        Returns
        -------
        pd.DataFrame
            The imputed data.
        """
        X_transformed = X.copy()
        
        for column, imputer in self.imputers_.items():
            if column in X.columns:
                # Reshape to 2D array for SimpleImputer
                X_transformed[column] = imputer.transform(X[[column]])
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit the imputer on X and then transform X.
        
        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer on and then impute.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        pd.DataFrame
            The imputed data.
        """
        return self.fit(X, y).transform(X)


# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame with missing values
    df = pd.DataFrame({
        'numeric_col1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric_col2': [np.nan, 2.0, 3.0, 4.0, 5.0],
        'categorical_col1': ['a', 'b', np.nan, 'd', 'e'],
        'categorical_col2': ['a', np.nan, 'c', 'd', 'e']
    })
    
    # Example 1: Using BGGImputer
    imputer = BGGImputer(
        numeric_strategy='median',
        categorical_strategy='most_frequent'
    )
    df_imputed = imputer.fit_transform(df)
    print("Example 1: Using BGGImputer")
    print(df_imputed)
    
    # Example 2: Using ColumnSpecificImputer
    column_imputer = ColumnSpecificImputer({
        'numeric_col1': {'strategy': 'mean'},
        'numeric_col2': {'strategy': 'median'},
        'categorical_col1': {'strategy': 'most_frequent'},
        'categorical_col2': {'strategy': 'constant', 'fill_value': 'missing'}
    })
    df_column_imputed = column_imputer.fit_transform(df)
    print("\nExample 2: Using ColumnSpecificImputer")
    print(df_column_imputed)
