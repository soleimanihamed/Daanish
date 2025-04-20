# utils/preprocessing/outlier_treatment.py

import pandas as pd


class OutlierHandler:
    """Handles treatment or removal of outlier values."""

    def __init__(self, data):
        self.data = data.copy()

    def handle(self, features, strategies, outliers_df, replace_values=None):
        """
        Handle outliers for multiple features using individual strategies.

        Args:
            features (list): List of feature names.
            strategies (list or dict): Strategy per feature. Can be list (same length) or dict.
                                    Strategies include:
                                        - "remove"           : Remove records containing outliers.
                                        - "replace_mean"     : Replace outliers with the mean.
                                        - "replace_median"   : Replace outliers with the median.
                                        - "replace_mode"     : Replace outliers with the mode.
                                        - "replace_value"    : Replace outliers with a specified value.
                                        - "ffill"            : Forward fill from previous value.
                                        - "bfill"            : Backward fill from next value.
                                        - "fill_interpolate" : Replace using linear interpolation.
                                        - "none" / "keep"    : Keep outliers as is.
            outliers_df (pd.DataFrame): A DataFrame with detected outlier values (same columns as self.data).
            replace_values (dict, optional): Custom values for "replace_value" strategy.

        Returns:
            pd.DataFrame: DataFrame with handled outlier records (with 'affected_features').
            pd.DataFrame: The cleaned dataset.
        """
        if outliers_df.empty:
            print("No outliers to handle.")
            return pd.DataFrame(), self.data.copy()

        affected_features_dict = {idx: [] for idx in outliers_df.index}
        original_record_count = len(self.data)
        imputed_record_count = 0

        if isinstance(strategies, list):
            if len(strategies) != len(features):
                raise ValueError(
                    "Length of strategies list must match length of features list.")
            for feature, strategy in zip(features, strategies):
                self._apply_outlier_strategy(
                    feature, strategy, outliers_df, replace_values, affected_features_dict)
        elif isinstance(strategies, dict):
            for feature in features:
                if feature in strategies:
                    self._apply_outlier_strategy(
                        feature, strategies[feature], outliers_df, replace_values, affected_features_dict)
                else:
                    print(f"Missing strategy for feature: {feature}")
        else:
            raise ValueError("strategies must be a list or a dictionary.")

        handled_indices = list(affected_features_dict.keys())
        handled_records = self.data.loc[handled_indices].copy()

        if not handled_records.empty:
            handled_records['affected_features'] = [
                ', '.join(affected_features_dict[idx]) for idx in handled_records.index
            ]

        removed_record_count = original_record_count - len(self.data)
        imputed_record_count = len(handled_records)

        print(f"Total number of records removed: {removed_record_count}")
        print(
            f"Total number of records handled (imputed or altered): {imputed_record_count}")

        return handled_records, self.data

    def _apply_outlier_strategy(self, feature, strategy, outliers_df, replace_values, affected_features_dict):
        """Apply strategy to a single feature's outliers."""
        if feature not in self.data.columns:
            print(f"Feature '{feature}' not in dataset.")
            return

        outlier_values = outliers_df[feature].dropna().unique()
        outlier_indices = self.data[self.data[feature].isin(
            outlier_values)].index

        if strategy == "remove":
            self.data.drop(index=outlier_indices, inplace=True)
            for idx in outlier_indices:
                affected_features_dict.pop(idx, None)

        elif strategy == "replace_mean":
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                replacement = self.data[feature].mean()
                self.data.loc[outlier_indices, feature] = replacement
                for idx in outlier_indices:
                    affected_features_dict[idx].append(feature)

        elif strategy == "replace_median":
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                replacement = self.data[feature].median()
                self.data.loc[outlier_indices, feature] = replacement
                for idx in outlier_indices:
                    affected_features_dict[idx].append(feature)

        elif strategy == "replace_mode":
            replacement = self.data[feature].mode()[0]
            self.data.loc[outlier_indices, feature] = replacement
            for idx in outlier_indices:
                affected_features_dict[idx].append(feature)

        elif strategy == "replace_value":
            if replace_values and feature in replace_values:
                replacement = replace_values[feature]
                self.data.loc[outlier_indices, feature] = replacement
                for idx in outlier_indices:
                    affected_features_dict[idx].append(feature)

            else:
                print(f"No replace_value provided for feature '{feature}'.")

        elif strategy == "ffill":
            self.data.loc[outlier_indices,
                          feature] = self.data[feature].ffill()
            for idx in outlier_indices:
                affected_features_dict[idx].append(feature)

        elif strategy == "bfill":
            self.data.loc[outlier_indices,
                          feature] = self.data[feature].bfill()
            for idx in outlier_indices:
                affected_features_dict[idx].append(feature)

        elif strategy == "fill_interpolate":
            self.data.loc[outlier_indices,
                          feature] = self.data[feature].interpolate()
            for idx in outlier_indices:
                affected_features_dict[idx].append(feature)

        elif strategy == "none" or strategy == "keep":
            print(f"Keeping outliers for feature: {feature}")

        else:
            raise ValueError(
                f"Invalid strategy '{strategy}' for feature '{feature}'.")
