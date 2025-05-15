# daanish/utils/preprocessing/scaler.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaler:
    """
    A flexible scaler for numerical features in a DataFrame, with optional skewness correction.

    This class supports:
    - Z-score standardization (StandardScaler)
    - Min-max normalization (MinMaxScaler)
    - Optional automatic handling of skewed variables via 'log', 'sqrt', or 'none'

    Parameters:
        method (str): Scaling method. Either 'zscore' or 'minmax'. Default is 'zscore'.
        handle_skew (bool): Whether to automatically transform skewed variables. Default is False.
        skew_method (str): Method to reduce skewness. Options: 'log', 'sqrt'. Ignored if handle_skew is False.
        skew_threshold (float): Absolute skewness value above which variables are transformed. Default is 1.0.

    Methods:
        fit_transform(df, columns): Fit and transform selected columns of DataFrame.
        transform(df): Transform the data using the fitted scaler.
        fit(df): Fit the scaler on selected columns.
        inverse_transform(df_scaled): Convert scaled data back to original scale.
    """

    def __init__(self, method='zscore', handle_skew=False, skew_method='log', skew_threshold=1.0):
        if method == 'zscore':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        if skew_method not in ['log', 'sqrt']:
            raise ValueError("skew_method must be 'log' or 'sqrt'")

        self.method = method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold
        self.columns = None
        self.skewed_cols = []

    def _reduce_skewness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Internal method to reduce skewness in the specified numerical columns.

        Applies a transformation (log1p or square root) to variables whose 
        absolute skewness exceeds the defined threshold. Transformed columns 
        are stored in self.skewed_cols.

        Args:
            df (pd.DataFrame): The input DataFrame containing original data.

        Returns:
            pd.DataFrame: A copy of the DataFrame with skewed columns transformed.
        """
        df_copy = df.copy()
        skewness = df_copy[self.columns].skew()

        for col in self.columns:
            if abs(skewness[col]) > self.skew_threshold:
                if self.skew_method == 'log':
                    df_copy[col] = np.log1p(df_copy[col])  # log(1 + x)
                elif self.skew_method == 'sqrt':
                    df_copy[col] = np.sqrt(df_copy[col])
                self.skewed_cols.append(col)
        return df_copy

    def fit_transform(self, df: pd.DataFrame, columns: list = None):
        """
        Fit the scaler to the DataFrame and return the scaled values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of numerical columns to scale. If None, uses all numeric columns.

        Returns:
            pd.DataFrame: Scaled DataFrame (only selected columns).
        """
        self.columns = columns if columns is not None else df.select_dtypes(
            include=np.number).columns.tolist()

        df_processed = self._reduce_skewness(
            df) if self.handle_skew else df.copy()
        scaled_values = self.scaler.fit_transform(df_processed[self.columns])
        return pd.DataFrame(scaled_values, columns=self.columns, index=df.index)

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame using the already-fitted scaler.

        Args:
            df (pd.DataFrame): New data with same structure.

        Returns:
            pd.DataFrame: Scaled DataFrame (only selected columns).
        """
        df_copy = self._reduce_skewness(df) if self.handle_skew else df.copy()
        scaled_values = self.scaler.transform(df_copy[self.columns])
        return pd.DataFrame(scaled_values, columns=self.columns, index=df.index)

    def fit(self, df: pd.DataFrame, columns: list = None):
        """
        Fit the scaler to the DataFrame without transforming it.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): Columns to fit. If None, uses all numeric columns.
        """
        self.columns = columns if columns is not None else df.select_dtypes(
            include=np.number).columns.tolist()
        df_copy = self._reduce_skewness(df) if self.handle_skew else df.copy()
        self.scaler.fit(df_copy[self.columns])
        return self

    def inverse_transform(self, df_scaled: pd.DataFrame):
        """
        Convert scaled data back to original scale (approximate).

        Args:
            df_scaled (pd.DataFrame): Scaled DataFrame to revert.

        Returns:
            pd.DataFrame: Data in original scale.
        """
        original_values = self.scaler.inverse_transform(df_scaled)
        return pd.DataFrame(original_values, columns=self.columns, index=df_scaled.index)
