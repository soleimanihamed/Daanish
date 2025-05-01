# utils/preprocessing/outlier_treatment.py

import pandas as pd


class OutlierHandler:
    """Handles treatment or removal of outlier values."""

    def __init__(self, data):
        self.data = data.copy()

    def handle_from_config(self, outlier_config_bundle, outliers_df):
        """
        Simplified handler using unified configuration structure.

        Args:
            outlier_config_bundle (dict): Dictionary with structure:
                {
                    feature_name: {
                        "method": ...,  # detection method (ignored here)
                        "params": ...,  # detection params (ignored here)
                        "imputation_method": str,
                        "imputation_value": optional custom value
                    },
                    ...
                }
            outliers_df (pd.DataFrame): DataFrame of detected outlier records.

        Returns:
            pd.DataFrame: Outlier-imputed records.
            pd.DataFrame: Cleaned dataset.
        """

        invalid = {None, "", "nan", "NaN", "NAN"}

        features = list(outlier_config_bundle.keys())

        imputation_methods = {
            f: (cfg.get("imputation_method") if cfg.get(
                "imputation_method") not in invalid else "keep")
            for f, cfg in outlier_config_bundle.items()
        }

        imputation_values = {
            f: cfg.get("imputation_value")
            for f, cfg in outlier_config_bundle.items()
            if imputation_methods[f] == "replace_value"
        }

        return self.handle(
            features,
            imputation_method=imputation_methods,
            outliers_df=outliers_df,
            replace_values=imputation_values
        )

    def handle(self, features, imputation_method, outliers_df, replace_values=None):
        """
        Handle outliers for multiple features using individual imputation method.

        Args:
            features (list): List of feature names.
            imputation_method (list or dict): Imputation method per feature. Can be list (same length) or dict.
                                    Imputation_ methods include:
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

        if isinstance(imputation_method, list):
            if len(imputation_method) != len(features):
                raise ValueError(
                    "Length of imputation method list must match length of features list.")
            for feature, method in zip(features, imputation_method):
                self._apply_outlier_imputation_method(
                    feature, method, outliers_df, replace_values, affected_features_dict)
        elif isinstance(imputation_method, dict):
            for feature in features:
                if feature in imputation_method:
                    self._apply_outlier_imputation_method(
                        feature, imputation_method[feature], outliers_df, replace_values, affected_features_dict)
                else:
                    print(
                        f"Missing outlier imputation method for feature: {feature}")
        else:
            raise ValueError(
                "Outlier imputation method must be a list or a dictionary.")

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

    def _apply_outlier_imputation_method(self, feature, imputation_method, outliers_df, replace_values, affected_features_dict):
        """Apply imputation method to a single feature's outliers."""
        if feature not in self.data.columns:
            print(f"Feature '{feature}' not in dataset.")
            return

        # Identify rows where this feature is marked as an outlier in 'outlier_source'
        relevant_rows = outliers_df[
            outliers_df["outlier_source"].str.contains(rf"\b{feature}\b")
        ]

        # Indexes of rows to be updated for this feature
        outlier_indices = relevant_rows.index

        if imputation_method == "remove":
            self.data.drop(index=outlier_indices, inplace=True)
            for idx in outlier_indices:
                affected_features_dict.pop(idx, None)

        elif imputation_method == "replace_mean":
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                replacement = self.data[feature].mean()
                self.data.loc[outlier_indices, feature] = replacement
                for idx in outlier_indices:
                    if idx in affected_features_dict:
                        affected_features_dict[idx].append(feature)

        elif imputation_method == "replace_median":
            if pd.api.types.is_numeric_dtype(self.data[feature]):
                replacement = self.data[feature].median()
                self.data.loc[outlier_indices, feature] = replacement
                for idx in outlier_indices:
                    if idx in affected_features_dict:
                        affected_features_dict[idx].append(feature)

        elif imputation_method == "replace_mode":
            replacement = self.data[feature].mode()[0]
            self.data.loc[outlier_indices, feature] = replacement
            for idx in outlier_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)

        elif imputation_method == "replace_value":
            if replace_values and feature in replace_values:
                replacement = replace_values[feature]
                self.data.loc[outlier_indices, feature] = replacement
                for idx in outlier_indices:
                    if idx in affected_features_dict:
                        affected_features_dict[idx].append(feature)
            else:
                print(f"No replace_value provided for feature '{feature}'.")

        elif imputation_method == "ffill":
            self.data.loc[outlier_indices,
                          feature] = self.data[feature].ffill()
            for idx in outlier_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)

        elif imputation_method == "bfill":
            self.data.loc[outlier_indices,
                          feature] = self.data[feature].bfill()
            for idx in outlier_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)

        elif imputation_method == "fill_interpolate":
            self.data.loc[outlier_indices,
                          feature] = self.data[feature].interpolate()
            for idx in outlier_indices:
                if idx in affected_features_dict:
                    affected_features_dict[idx].append(feature)

        elif imputation_method == "none" or imputation_method == "keep":
            print(f"Keeping outliers for feature: {feature}")

        else:
            raise ValueError(
                f"Invalid imputation method '{imputation_method}' for feature '{feature}'.")

    def filter_outlier_heavy_rows(self, outliers_df, threshold=0.5):
        """
        Removes rows where the proportion of outlier features exceeds the threshold.

        Args:
            outliers_df (pd.DataFrame): DataFrame with 'outlier_source' column listing outlier features per row.
            threshold (float): Proportion of outlier-affected features above which to remove a row (0.0 - 1.0).

        Returns:
            pd.DataFrame: A new DataFrame with outlier-heavy rows removed.
        """
        if outliers_df.empty or "outlier_source" not in outliers_df.columns:
            print("⚠️ No outliers to process for row-wise filtering.")
            return self.data.copy()

        # Determine numerical features directly from the data
        numerical_features = [col for col in self.data.columns
                              if pd.api.types.is_numeric_dtype(self.data[col])]

        total_numerical = len(numerical_features)
        if total_numerical == 0:
            print("❌ No numerical features found in the dataset.")
            return self.data.copy()

        # Compute number of outlier features per row
        outlier_feature_counts = outliers_df["outlier_source"].str.split(
            ",").apply(len)
        outlier_ratios = outlier_feature_counts / total_numerical

        # Identify rows exceeding the threshold
        heavy_outlier_indices = outlier_ratios[outlier_ratios >
                                               threshold].index

        # Drop those rows
        filtered_data = self.data.drop(index=heavy_outlier_indices)

        print(
            f"🧹 Removed {len(heavy_outlier_indices)} rows with > {threshold*100:.0f}% outlier features.")

        return filtered_data
