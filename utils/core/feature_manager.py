# utils/core/feature_manager.py

import pandas as pd
from utils.data_io.loader import load_data


class FeatureManager:
    """
    Handles feature configuration, including identifying numerical, nominal, and ordinal features,
    as well as detecting the target variable and related metadata.
    """

    def __init__(self, source_type, input_path=None, query=None, global_config=None):
        """
        Initialize FeatureManager.

        Args:
            source_type (str): 'csv', 'excel', or 'sql', indicating the data source type.
            source_path (str, optional): Path to the CSV/Excel file.
            query (str, optional): SQL query to fetch feature configuration.
            global_config (dict, optional): Global configuration for database credentials (if using SQL).

        Raises:
            ValueError: If the required parameters are missing or invalid.
        """
        self.nominal_features = []
        self.ordinal_features = []
        self.numerical_features = []
        self.target_variable = None
        self.all_features = []
        self.missing_value_strategies = {}  # Dictionary to store missing value strategies
        self.missing_fill_values = {}      # Dictionary to store missing fill values
        self.display_names = {}             # Dictionary to store display names
        self.outlier_value_strategies = {}  # Dictionary to store outlier value strategies
        self.outlier_fill_values = {}      # Dictionary to store outlier fill values

        df = load_data(source_type, input_path=input_path,
                       query=query, global_config=global_config)

        if df is None or 'feature' not in df.columns or 'type' not in df.columns:
            raise ValueError("Data must contain 'feature' and 'type' columns.")

        self._process_feature_dataframe(df)

    def _process_feature_dataframe(self, df):
        """Processes the feature DataFrame and categorizes feature types."""
        self.nominal_features = df[df['type'] == 'nominal']['feature'].tolist()
        self.ordinal_features = df[df['type'] == 'ordinal']['feature'].tolist()
        self.numerical_features = df[df['type']
                                     == 'numerical']['feature'].tolist()

        # Target variable(s)
        if 'target_variable' in df.columns:
            self.target_variable = df[df['target_variable']
                                      == 1]['feature'].tolist()

        # Store all features in a single list
        self.all_features_list = (
            self.nominal_features + self.ordinal_features + self.numerical_features
        )

        # Load missing value strategies
        if 'miss_val_strategy' in df.columns:
            self.missing_value_strategies = dict(
                zip(df['feature'], df['miss_val_strategy']))

        if 'miss_fill_value' in df.columns:
            self.missing_fill_values = dict(
                zip(df['feature'], df['miss_fill_value']))

        # Load Display names
        if 'display_name' in df.columns:
            self.display_names = dict(zip(df['feature'], df['display_name']))

        # Load outlier value strategies
        if 'outlier_val_strategy' in df.columns:
            self.outlier_value_strategies = dict(
                zip(df['feature'], df['outlier_val_strategy']))

        if 'outlier_fill_value' in df.columns:
            self.outlier_fill_values = dict(
                zip(df['feature'], df['outlier_fill_value']))

    def get_nominal_features(self):
        """Return a list of nominal features."""
        return self.nominal_features

    def get_ordinal_features(self):
        """Return a list of ordinal features."""
        return self.ordinal_features

    def get_numerical_features(self):
        """Return a list of numerical features."""
        return self.numerical_features

    def get_target_variable(self):
        """Return the target variable."""
        return self.target_variable

    def get_all_features(self):
        """Return a dictionary of all feature categories."""
        return self.all_features_list

    def get_missing_value_strategies(self):
        """Return a dictionary of missing value strategies."""
        return self.missing_value_strategies

    def get_missing_fill_values(self):
        """Return a dictionary of missing fill values."""
        return self.missing_fill_values

    def get_display_names(self):
        """Return a dictionary of display names."""
        return self.display_names

    def get_outlier_value_strategies(self):
        """Return a dictionary of outlier value strategies."""
        return self.outlier_value_strategies

    def get_outlier_fill_values(self):
        """Return a dictionary of outlier fill values."""
        return self.outlier_fill_values
