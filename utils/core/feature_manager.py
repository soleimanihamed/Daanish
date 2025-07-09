# utils/core/feature_manager.py

import pandas as pd
from utils.data_io.loader import load_data
import json


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
        # Dictionary to store outlier detection strategies
        self.outlier_detection_strategies = {}
        # Dictionary to store outlier strategies parameters
        self.outlier_detection_params = {}
        # Dictionary to store imputation methods to treat outliers
        self.outlier_imputation_methods = {}
        # Dictionary to store custom value if outlier imputation method is 'Custom'
        self.outlier_imputation_values = {}
        # Dictionary to store custom binning if binning method is 'manual'
        self.binning_config = {}

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

        # Load outlier detection strategy
        if 'outlier_strategy' in df.columns:
            self.outlier_detection_strategies = dict(
                zip(df['feature'], df['outlier_strategy'])
            )

        # Load outlier detection params (as strings, to be parsed later)
        if 'outlier_params' in df.columns:
            self.outlier_detection_params = dict(
                zip(df['feature'], df['outlier_params'])
            )

        # Load outlier imputation method (e.g., fill_mean, drop)
        if 'outlier_imputation_method' in df.columns:
            self.outlier_imputation_methods = dict(
                zip(df['feature'], df['outlier_imputation_method'])
            )

        # Load outlier imputation value (if required)
        if 'outlier_imputation_value' in df.columns:
            self.outlier_imputation_values = dict(
                zip(df['feature'], df['outlier_imputation_value'])
            )

        # Load binning cofig to manually bin data
        if 'binning_config' in df.columns:
            self.binning_configs = dict(
                zip(df['feature'], df['binning_config']))
        else:
            self.binning_configs = {}

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

    def get_outlier_detection_strategies(self):
        return self.outlier_detection_strategies

    def get_outlier_detection_params(self):
        return self.outlier_detection_params

    def get_outlier_imputation_methods(self):
        return self.outlier_imputation_methods

    def get_outlier_imputation_values(self):
        return self.outlier_imputation_values

    def _parse_outlier_params(self):
        """Parses outlier_params JSON strings per feature into dictionaries."""
        parsed = {}
        for feature, param_str in self.outlier_detection_params.items():
            try:
                parsed[feature] = json.loads(param_str) if param_str else {}
            except Exception:
                print(
                    f"Warning: invalid JSON in outlier_params for feature '{feature}'")
                parsed[feature] = {}
        return parsed

    def get_outlier_config_bundle(self):
        """
        Builds a full config dictionary for outlier detection and treatment.

        Returns:
            dict: {
                feature: {
                    "method": str,              # detection method
                    "params": dict,             # detection parameters
                    "imputation_method": str,   # how to handle outliers
                    "imputation_value": any     # optional value if method is "replace_value"
                }
            }
        """
        parsed_params = self._parse_outlier_params()
        config = {}

        for feature, method in self.outlier_detection_strategies.items():
            config[feature] = {
                "method": method,
                "params": parsed_params.get(feature, {}),
                "imputation_method": self.outlier_imputation_methods.get(feature, "none"),
                "imputation_value": self.outlier_imputation_values.get(feature)
            }

        return config

    def get_binning_configs(self):
        """Return a dictionary of raw binning_config strings."""
        return self.binning_configs

    def get_binning_config_bundle(self):
        """
        Builds a full binning config dictionary for use in the Binner class.

        Returns:
            dict: {
                feature: {
                    "type": "numerical" or "categorical",
                    "method": "manual_numerical" or "manual_categorical",
                    "bin_edges" or "mapping": list
                }
            }
        """

        config = {}
        for feature, raw_value in self.binning_configs.items():
            if pd.isna(raw_value):
                continue
            try:
                parsed = json.loads(raw_value)
            except Exception:
                print(
                    f"Warning: Invalid JSON for binning_config in feature '{feature}'")
                continue

            if feature in self.nominal_features:
                feature_type = "categorical"
                method = "manual_categorical"
                config[feature] = {
                    "type": feature_type,
                    "method": method,
                    "mapping": parsed
                }
            else:  # numerical or ordinal
                feature_type = "numerical"
                method = "manual_numerical"
                config[feature] = {
                    "type": feature_type,
                    "method": method,
                    "bin_edges": parsed
                }

        return config
