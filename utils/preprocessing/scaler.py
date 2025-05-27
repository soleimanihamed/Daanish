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
        inverse_transform(df_scaled): Convert scaled data back to original scale,
                                    including reversing skewness transformations.
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
        self.columns = None  # Stores the names of the columns that the scaler is fitted on
        self.skewed_cols = []  # Stores names of columns that had skewness transformation applied

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
        # Calculate skewness only for the columns the scaler is supposed to handle
        skewness = df_copy[self.columns].skew()

        current_skewed_cols = []  # Keep track for this fit/transform pass
        for col in self.columns:
            if abs(skewness[col]) > self.skew_threshold:
                if self.skew_method == 'log':
                    df_copy[col] = np.log1p(df_copy[col])
                elif self.skew_method == 'sqrt':
                    df_copy[col] = np.sqrt(df_copy[col])
                current_skewed_cols.append(col)

        # Only update self.skewed_cols if this is part of a fit operation
        # This check assumes _reduce_skewness is called by fit or fit_transform
        # If called by transform, self.skewed_cols should already be set by fit.
        if not self.skewed_cols:  # Or a more robust way to check if fitting
            self.skewed_cols = current_skewed_cols
        elif set(self.skewed_cols) != set(current_skewed_cols) and any(col in self.skewed_cols for col in current_skewed_cols):
            # This condition might indicate calling fit_transform after fit with different skew outcomes,
            # or applying to new data in transform where skewness differs.
            # For simplicity, we assume skew characteristics are stable from fit.
            # If transform needs to re-evaluate skew, logic would be more complex.
            # Current design: skewness decisions are fixed at fit time.
            pass

        return df_copy

    def fit(self, df: pd.DataFrame, columns: list = None):
        """
        Fit the scaler to the DataFrame without transforming it.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): Columns to fit. If None, uses all numeric columns from df.
        """
        if columns is not None:
            self.columns = list(columns)  # Ensure it's a list copy
        else:
            self.columns = df.select_dtypes(include=np.number).columns.tolist()

        # Work on a copy of relevant columns
        df_to_fit = df[self.columns].copy()
        self.skewed_cols = []  # Reset for fitting

        if self.handle_skew:
            # This will populate self.skewed_cols
            df_to_fit = self._reduce_skewness(df_to_fit)

        self.scaler.fit(df_to_fit)
        return self

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame using the already-fitted scaler.

        Args:
            df (pd.DataFrame): New data with same structure (must contain self.columns).

        Returns:
            pd.DataFrame: Scaled DataFrame (only selected columns).
        """
        if self.columns is None:
            raise ValueError(
                "Scaler has not been fitted yet. Call fit() or fit_transform() first.")

        df_to_transform = df[self.columns].copy()

        if self.handle_skew and self.skewed_cols:  # Use self.skewed_cols determined at fit time
            for col in self.skewed_cols:
                if col in df_to_transform.columns:
                    if self.skew_method == 'log':
                        df_to_transform[col] = np.log1p(df_to_transform[col])
                    elif self.skew_method == 'sqrt':
                        df_to_transform[col] = np.sqrt(df_to_transform[col])

        scaled_values = self.scaler.transform(df_to_transform)
        return pd.DataFrame(scaled_values, columns=self.columns, index=df.index if isinstance(df, pd.DataFrame) else None)

    def fit_transform(self, df: pd.DataFrame, columns: list = None):
        """
        Fit the scaler to the DataFrame and return the scaled values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (list): List of numerical columns to scale. If None, uses all numeric columns from df.

        Returns:
            pd.DataFrame: Scaled DataFrame (only selected columns).
        """
        self.fit(df, columns=columns)  # This sets self.columns and self.skewed_cols
        return self.transform(df)     # Transform uses the fitted parameters

    def inverse_transform(self, data_scaled):
        """
        Convert scaled data back to original scale.
        This includes reversing skewness transformations if they were applied.

        Args:
            data_scaled (pd.DataFrame or np.ndarray): Scaled data to revert.
                                                   If DataFrame, columns should match self.columns.
                                                   If ndarray, number of columns should match len(self.columns).

        Returns:
            pd.DataFrame: Data in original scale, with columns as self.columns.
        """
        if self.columns is None:
            raise ValueError(
                "Scaler has not been fitted yet or columns are not set.")

        original_index = None
        if isinstance(data_scaled, pd.DataFrame):
            # Ensure we're using the correct columns in the correct order for inverse_transform
            if not all(col in data_scaled.columns for col in self.columns):
                raise ValueError(
                    "data_scaled DataFrame is missing some of the fitted columns.")
            values_to_inverse = data_scaled[self.columns].values
            original_index = data_scaled.index
        elif isinstance(data_scaled, np.ndarray):
            values_to_inverse = data_scaled
        else:
            raise TypeError(
                "Input data_scaled must be a pandas DataFrame or a NumPy ndarray.")

        # Handle 1D array case (e.g. single sample)
        if values_to_inverse.ndim == 1:
            values_to_inverse = values_to_inverse.reshape(1, -1)
        if values_to_inverse.shape[1] != len(self.columns):
            raise ValueError(
                f"Number of features in data_scaled ({values_to_inverse.shape[1]}) "
                f"does not match number of fitted columns ({len(self.columns)})."
            )

        # Step 1: Inverse the primary scaling (z-score or min-max)
        unscaled_values = self.scaler.inverse_transform(values_to_inverse)

        # Create a DataFrame from the unscaled values to easily work with columns
        df_reverted = pd.DataFrame(
            unscaled_values, columns=self.columns, index=original_index)

        # Step 2: Inverse the skewness transformations for relevant columns
        # Use self.skewed_cols which was determined during the fit process
        if self.handle_skew and self.skewed_cols:
            for col in self.skewed_cols:
                if col in df_reverted.columns:
                    if self.skew_method == 'log':
                        # Inverse of log1p(x) = log(1+x) is expm1(y) = exp(y) - 1
                        df_reverted[col] = np.expm1(df_reverted[col])
                    elif self.skew_method == 'sqrt':
                        # Inverse of sqrt(x) is x^2
                        df_reverted[col] = np.square(df_reverted[col])

        return df_reverted
