# utils/data_variable_encoder.py

from utils.feature_configuration_reader import FeatureConfigurationReader
from utils.feature_selector import FeatureSelector
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import OrdinalEncoder


class DataVariableEncoder:
    def __init__(self, feature_config_reader):
        """
        Initializes the DataVariableEncoder with specific features and encoders.

        Args:
            feature_config_reader (FeatureConfigurationReader): Instance of FeatureConfigurationReader containing feature types.
        """
        # Set feature types from the configuration reader
        self.nominal_features = feature_config_reader.nominal_features
        self.ordinal_features = feature_config_reader.ordinal_features
        self.numerical_features = feature_config_reader.numerical_features

        # Initialize encoders and scaler
        self.nominal_encoder = OneHotEncoder(handle_unknown='ignore')
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()

    def determine_variable_type(self, data):
        """
        Determines the type of each feature in the dataset.

        Args:
            data (pd.DataFrame): The dataset containing the features.

        Returns:
            dict: Dictionary with feature names as keys and types (nominal, ordinal, numerical) as values.
        """
        feature_types = {}
        for column in data.columns:
            if column in self.nominal_features:
                feature_types[column] = 'nominal'
            elif column in self.ordinal_features:
                feature_types[column] = 'ordinal'
            elif column in self.numerical_features:
                feature_types[column] = 'numerical'
            else:
                feature_types[column] = 'unknown'
        return feature_types

    def encode_nominal_features(self, data):
        """
        Encodes nominal features using OneHotEncoder.

        Args:
            data (pd.DataFrame): The dataset containing the nominal features.

        Returns:
            pd.DataFrame: DataFrame with nominal features encoded.
        """
        nominal_data = self.nominal_encoder.fit_transform(
            data[self.nominal_features]).toarray()
        nominal_df = pd.DataFrame(
            nominal_data, columns=self.nominal_encoder.get_feature_names_out(self.nominal_features))
        return nominal_df

    def encode_ordinal_features(self, data):
        """
        Encodes ordinal features using OrdinalEncoder.

        Args:
            data (pd.DataFrame): The dataset containing the ordinal features.

        Returns:
            pd.DataFrame: DataFrame with ordinal features encoded.
        """
        ordinal_data = self.ordinal_encoder.fit_transform(
            data[self.ordinal_features])
        ordinal_df = pd.DataFrame(ordinal_data, columns=self.ordinal_features)
        return ordinal_df

    def scale_numerical_features(self, data):
        """
        Scales numerical features using StandardScaler.

        Args:
            data (pd.DataFrame): The dataset containing the numerical features.

        Returns:
            pd.DataFrame: DataFrame with numerical features scaled.
        """
        numerical_data = self.scaler.fit_transform(
            data[self.numerical_features])
        numerical_df = pd.DataFrame(
            numerical_data, columns=self.numerical_features)
        return numerical_df

    def preprocess_data(self, data, selected_features=None, exclude_features=None):
        """
        Preprocesses the data by encoding nominal and ordinal features and scaling numerical features.

        Args:
            data (pd.DataFrame): The dataset to preprocess.
            selected_features (list, optional): A list of features to include in the preprocessing.
            exclude_features (list, optional): A list of features to exclude from the preprocessing.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        # Use FeatureSelector to determine the features to process
        feature_selector = FeatureSelector(data)
        selected_data = feature_selector.select_features(
            selected_features, exclude_features)

        try:
            nominal_df = self.encode_nominal_features(selected_data) if any(
                f in self.nominal_features for f in selected_data.columns) else pd.DataFrame()
            ordinal_df = self.encode_ordinal_features(selected_data) if any(
                f in self.ordinal_features for f in selected_data.columns) else pd.DataFrame()
            numerical_df = self.scale_numerical_features(selected_data) if any(
                f in self.numerical_features for f in selected_data.columns) else pd.DataFrame()

            # Concatenate all preprocessed features
            processed_data = pd.concat(
                [nominal_df, ordinal_df, numerical_df], axis=1)
            return processed_data
        except ValueError as e:
            print(f"Error during data preprocessing: {e}")
            return None
