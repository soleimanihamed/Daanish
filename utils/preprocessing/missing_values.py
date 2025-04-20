# utils/preprocessing/missing_values.py

import pandas as pd


class MissingValueHandler:
    """Handles missing value imputation or removal strategies."""

    def __init__(self, data):
        self.data = data.copy()

    def handle(self, features, strategies, fill_values=None):
        """
        Handle missing values for multiple features with individual strategies.

        Args:
            features (list): List of feature names to handle.
            strategies (list or dict): Strategies for each feature.
                                     If a list, it should have the same length as features.
                                     If a dict, keys should be feature names.
                                     Allowed strategies:
                                         - "drop": Remove rows with missing values.
                                         - "fill_mean": Fill with the mean (numeric features only).
                                         - "fill_median": Fill with the median (numeric features only).
                                         - "fill_mode": Fill with the mode (most frequent value).
                                         - "fill_value": Fill with a specified value.
                                         - "ffill" (fill_forward): Propagate last valid observation forward.
                                         - "bfill" (fill_backward): Propagate next valid observation backward.
                                         - "fill_interpolate": Fill using linear interpolation.
                                         - "none" or "keep": Keep missing values as is.
            fill_values (list or dict, optional): Fill values for "fill_value" strategy.
                                                 If a list, it should have the same length as features.
                                                 If a dict, keys should be feature names.
            Returns:
            pandas.DataFrame: DataFrame containing records with replaced null values,
                              with an added 'affected_features' column.
        """
        null_records_before = self.data[self.data[features].isnull().any(
            axis=1)].copy()
        affected_features_dict = {idx: [] for idx in null_records_before.index}
        original_record_count = len(self.data)  # store original record count
        imputed_record_count = 0  # Initialize imputed record count

        if isinstance(strategies, list):
            if len(strategies) != len(features):
                raise ValueError(
                    "Length of strategies list must match length of features list.")
            for feature, strategy in zip(features, strategies):
                self._apply_strategy(
                    feature, strategy, fill_values, affected_features_dict)
        elif isinstance(strategies, dict):
            for feature in features:
                if feature in strategies:
                    self._apply_strategy(
                        feature, strategies[feature], fill_values, affected_features_dict)
                else:
                    print(f"Missing strategy for feature: {feature}")
        else:
            raise ValueError("strategies must be a list or a dictionary.")

        imputed_records = self.data.loc[list(
            affected_features_dict.keys())].copy()
        if imputed_records.empty:
            return pd.DataFrame()

        imputed_records['affected_features'] = [
            ', '.join(affected_features_dict[idx]) for idx in imputed_records.index]

        # get the length of replaced_records.
        imputed_record_count = len(imputed_records)

        removed_record_count = original_record_count - \
            len(self.data)  # Calculate removed records

        print(f"Total number of records removed: {removed_record_count}")
        print(f"Total number of records imputed: {imputed_record_count}")

        imputed_dataset = self.data

        return imputed_records, imputed_dataset

    def _apply_strategy(self, feature, strategy, fill_values, affected_features_dict):
        """Helper function to apply a strategy to a single feature."""
        null_indices = self.data[self.data[feature].isnull()].index

        if strategy == "drop":
            self.data.dropna(subset=[feature], inplace=True)
            for idx in null_indices:
                if idx in affected_features_dict:
                    affected_features_dict.pop(idx, None)
        elif strategy == "fill_mean":
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                self.data[feature] = self.data[feature].fillna(
                    self.data[feature].mean())
                for idx in null_indices:
                    if idx in affected_features_dict:
                        affected_features_dict[idx].append(feature)
            else:
                print(f"Skipping fill_mean for non-numeric feature: {feature}")
        elif strategy == "fill_median":
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                self.data[feature] = self.data[feature].fillna(
                    self.data[feature].median())
                for idx in null_indices:
                    if idx in affected_features_dict:
                        affected_features_dict[idx].append(feature)
            else:
                print(
                    f"Skipping fill_median for non-numeric feature: {feature}")
        elif strategy == "fill_mode":
            self.data[feature] = self.data[feature].fillna(
                self.data[feature].mode()[0])
            for idx in null_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)
        elif strategy == "fill_value":
            if fill_values is None:
                raise ValueError(
                    "fill_values must be provided for fill_value strategy."
                )
            if isinstance(fill_values, list):
                if len(fill_values) != len(self.data.columns):
                    raise ValueError(
                        "Length of fill_values list must match length of the dataframe columns."
                    )
                self.data[feature] = self.data[feature].fillna(
                    fill_values[self.data.columns.get_loc(feature)]
                )
                for idx in null_indices:
                    if idx in affected_features_dict:
                        affected_features_dict[idx].append(feature)
            elif isinstance(fill_values, dict):
                if feature in fill_values:
                    self.data[feature] = self.data[feature].fillna(
                        fill_values[feature])
                    for idx in null_indices:
                        if idx in affected_features_dict:
                            affected_features_dict[idx].append(feature)
                else:
                    print(f"Missing fill_value for feature: {feature}")
            else:
                raise ValueError("fill_values must be a list or a dictionary.")
        elif strategy == "ffill":
            self.data[feature] = self.data[feature].fillna(
                method='ffill')
            for idx in null_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)
        elif strategy == "bfill":
            self.data[feature] = self.data[feature].fillna(
                method='bfill')
            for idx in null_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)
        elif strategy == "fill_interpolate":

            self.data[feature] = self.data[feature].interpolate()
            for idx in null_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)
        elif strategy == "none" or strategy == "keep":
            print(f"Keeping missing values for feature: {feature}")
            pass  # do nothing

        else:
            raise ValueError(
                f"Invalid missing value handling strategy: {strategy}")
