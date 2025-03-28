# utils/data_preprocessor.py
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (norm, expon, lognorm, gamma, beta, weibull_min, chi2, pareto, uniform, t, gumbel_r, burr, invgauss,
                         triang, laplace, logistic, genextreme, skewnorm, genpareto, burr12, fatiguelife, geninvgauss, halfnorm, exponpow)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils.eda_statistical_analysis import StatisticalEDAAnalysis


class DataPreprocessor:
    """Handles data cleaning and preprocessing."""

    def __init__(self, data):
        self.data = data.copy()  # Make a copy to avoid modifying original data

    def handle_missing_values(self, features, strategies, fill_values=None):
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

    def detect_outliers_iqr(self, df, features, threshold=1.5):
        """Detect outliers for numerical features using IQR."""
        numerical_outliers = []

        for feature in features:
            if feature not in df.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            # Check if feature is numerical
            if pd.api.types.is_numeric_dtype(df[feature]):
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)

                feature_outliers = df[(df[feature] < lower_bound) | (
                    df[feature] > upper_bound)]

                if not feature_outliers.empty:
                    outliers_df = feature_outliers.copy()
                    outliers_df["outlier_source"] = feature
                    numerical_outliers.append(outliers_df)
            else:
                print(f"Skipping IQR for non-numerical feature: {feature}")

        if not numerical_outliers:
            print("No numerical outliers found.")
            return pd.DataFrame()

        combined_outliers = pd.concat(numerical_outliers)

        grouped_sources = combined_outliers.groupby(combined_outliers.index)[
            "outlier_source"].apply(lambda x: ", ".join(set(x)))

        combined_outliers = combined_outliers.loc[grouped_sources.index].copy()
        combined_outliers["outlier_source"] = grouped_sources

        num_outliers = len(combined_outliers)
        print(f"Total number of IQR outliers found: {num_outliers}")

        return combined_outliers

    def detect_custom_outliers(self, df, features, lower_bounds=None, upper_bounds=None):
        """Detect outliers using custom lower and upper bounds for numerical features."""
        all_outliers = []

        if lower_bounds is None and upper_bounds is None:
            print("Warning: No lower or upper bounds provided.")
            return pd.DataFrame()  # Return empty DataFrame if no bounds are provided

        for feature in features:
            if feature not in df.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            # Check if feature is numerical
            if pd.api.types.is_numeric_dtype(df[feature]):
                lower_bound = float('-inf')  # Default lower bound
                upper_bound = float('inf')  # Default upper bound

                if lower_bounds and feature in lower_bounds:
                    lower_bound = lower_bounds[feature]

                if upper_bounds and feature in upper_bounds:
                    upper_bound = upper_bounds[feature]

                feature_outliers = df[(df[feature] < lower_bound) | (
                    df[feature] > upper_bound)]

                if not feature_outliers.empty:
                    outliers_df = feature_outliers.copy()
                    outliers_df["outlier_source"] = feature
                    all_outliers.append(outliers_df)
            else:
                print(
                    f"Skipping custom bounds for non-numerical feature: {feature}")

        if not all_outliers:
            print("No custom outliers found.")
            return pd.DataFrame()

        combined_outliers = pd.concat(all_outliers)

        grouped_sources = combined_outliers.groupby(combined_outliers.index)[
            "outlier_source"].apply(lambda x: ", ".join(set(x)))

        combined_outliers = combined_outliers.loc[grouped_sources.index].copy()
        combined_outliers["outlier_source"] = grouped_sources

        num_outliers = len(combined_outliers)
        print(f"Total number of custom outliers found: {num_outliers}")

        return combined_outliers

    def detect_outliers_zscore(self, df, features, threshold=3, alpha=0.05):
        """
        Detect outliers using Z-score for numerical features, with normality testing.

        Args:
            df (pd.DataFrame): Input DataFrame.
            features (list): List of feature names.
            threshold (float): Z-score threshold for outlier detection.
            alpha (float): Significance level for the Shapiro-Wilk test.

        Returns:
            pd.DataFrame: DataFrame containing detected outliers.
        """
        all_outliers = []
        eda = StatisticalEDAAnalysis(df)

        if not isinstance(features, list):
            print("Warning: Features must be a list.")
            return pd.DataFrame()

        normality_results = eda.perform_normality_test_Shapiro_Wilk(
            features, alpha)  # call normality test

        for feature in features:
            if feature not in df.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            if pd.api.types.is_numeric_dtype(df[feature]):
                if normality_results[feature]["is_normal"]:  # check if normal
                    mean = df[feature].mean()
                    std = df[feature].std()

                    # Handle cases where std is zero (constant column)
                    if std == 0:
                        print(
                            f"Warning: Standard deviation is zero for feature '{feature}'. Skipping.")
                        continue

                    z_scores = abs((df[feature] - mean) / std)
                    feature_outliers = df[z_scores > threshold]

                    if not feature_outliers.empty:
                        outliers_df = feature_outliers.copy()
                        outliers_df["outlier_source"] = feature
                        all_outliers.append(outliers_df)
                else:
                    print(
                        f"Warning: Feature '{feature}' does not appear to be normally distributed (p-value={normality_results[feature]['shapiro_p_value']}). Z-score may be inaccurate. Consider alternative methods.")
            else:
                print(f"Skipping Z-score for non-numerical feature: {feature}")

        if not all_outliers:
            print("No Z-score outliers found.")
            return pd.DataFrame()

        combined_outliers = pd.concat(all_outliers)

        grouped_sources = combined_outliers.groupby(combined_outliers.index)[
            "outlier_source"].apply(lambda x: ", ".join(set(x)))

        combined_outliers = combined_outliers.loc[grouped_sources.index].copy()
        combined_outliers["outlier_source"] = grouped_sources

        num_outliers = len(combined_outliers)
        print(f"Total number of Z-score outliers found: {num_outliers}")

        return combined_outliers

    def detect_outliers_distribution(self, df, distribution_results, confidence_interval=0.95):
        """
        Detects outliers based on fitted probability distributions for multiple features.

        Args:
            distribution_results (dict): Dictionary mapping feature names to their best-fitting distribution and parameters.
            confidence_interval (float): Confidence interval for outlier detection.

        Returns:
            pd.DataFrame: DataFrame containing combined outliers with an 'outlier_source' column.
        """

        # Map string distribution names to their SciPy equivalents
        distribution_mapping = {
            "norm": norm, "expon": expon, "lognorm": lognorm, "gamma": gamma, "beta": beta, "weibull_min": weibull_min,
            "chi2": chi2, "pareto": pareto, "uniform": uniform, "t": t, "gumbel_r": gumbel_r, "burr": burr,
            "invgauss": invgauss, "triang": triang, "laplace": laplace, "logistic": logistic, "genextreme": genextreme,
            "skewnorm": skewnorm, "genpareto": genpareto, "burr12": burr12, "fatiguelife": fatiguelife,
            "geninvgauss": geninvgauss, "halfnorm": halfnorm, "exponpow": exponpow
        }

        all_outliers = []  # List to collect outliers
        skipped_distributions = []  # List to collect skipped distributions

        for feature, result in distribution_results.items():
            if result["best_distribution"] is None:
                skipped_distributions.append(
                    (feature, "No distribution found"))
                continue

            distribution_name = result["best_distribution"]
            params = result["parameters"]
            filtered_data = df[feature].dropna()
            distribution = distribution_mapping.get(distribution_name)

            if distribution is None:
                skipped_distributions.append(
                    (feature, f"{distribution_name} not found in SciPy"))
                continue

            try:
                # Extract parameters correctly
                if distribution_name == "norm":
                    loc, scale = params["loc"], params["scale"]
                    lower_bound = norm.ppf(
                        (1 - confidence_interval) / 2, loc=loc, scale=scale)
                    upper_bound = norm.ppf(
                        (1 + confidence_interval) / 2, loc=loc, scale=scale)
                elif distribution_name == "lognorm":
                    shape, loc, scale = params["s"], params["loc"], params["scale"]
                    lower_bound = lognorm.ppf(
                        (1 - confidence_interval) / 2, shape, loc=loc, scale=scale)
                    upper_bound = lognorm.ppf(
                        (1 + confidence_interval) / 2, shape, loc=loc, scale=scale)
                elif distribution_name in ["beta", "gamma", "triang", "burr", "weibull_min"]:
                    shape_params = tuple(
                        params[k] for k in params if k not in ["loc", "scale"])
                    loc, scale = params["loc"], params["scale"]
                    lower_bound = distribution.ppf(
                        (1 - confidence_interval) / 2, *shape_params, loc=loc, scale=scale)
                    upper_bound = distribution.ppf(
                        (1 + confidence_interval) / 2, *shape_params, loc=loc, scale=scale)
                else:
                    shape_params = tuple(params.values())
                    # Default for one-sided distributions
                    lower_bound = float("-inf")
                    upper_bound = distribution.ppf(
                        confidence_interval, *shape_params)

                # Identify outliers
                outliers = filtered_data[(filtered_data < lower_bound) | (
                    filtered_data > upper_bound)]

                if not outliers.empty:
                    outliers_df = df.loc[outliers.index]
                    outliers_df["outlier_source"] = feature
                    all_outliers.append(outliers_df)

            except Exception as e:
                skipped_distributions.append(
                    (feature, f"{distribution_name} error: {e}"))
                continue

        if not all_outliers:
            combined_outliers = pd.DataFrame()
        else:
            # Concatenate all outliers into a single DataFrame
            combined_outliers = pd.concat(all_outliers)

            # Group by index and combine outlier sources into a single string
            grouped_sources = combined_outliers.groupby(combined_outliers.index)[
                "outlier_source"].apply(lambda x: ", ".join(set(x)))

            combined_outliers = combined_outliers.loc[grouped_sources.index].copy(
            )
            combined_outliers["outlier_source"] = grouped_sources

        # Print skipped distributions (if any)
        if skipped_distributions:
            print("\nDistributions Skipped:")
            for feature, reason in skipped_distributions:
                print(f"Feature: {feature}, Reason: {reason}")

        return combined_outliers

    def detect_outliers_isolation_forest(self, df, features, contamination='auto', max_samples=0.8, n_estimators=200):
        """Detect outliers using Isolation Forest for numerical features."""

        if not isinstance(features, list):
            print("Warning: Features must be a list.")
            return pd.DataFrame()

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            max_samples=max_samples,
            n_estimators=n_estimators
        )

        outlier_flags = {}

        for feature in features:
            if feature not in df.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            if pd.api.types.is_numeric_dtype(df[feature]):
                data_reshaped = df[[feature]].values
                outlier_flags[feature] = iso_forest.fit_predict(
                    data_reshaped) == -1  # -1 means outlier
            else:
                print(
                    f"Skipping Isolation Forest for non-numerical feature: {feature}")

        outlier_records = pd.DataFrame(index=df.index)

        for feature, flags in outlier_flags.items():
            if feature in df.columns:  # prevent error if non numerical features were passed.
                outlier_records[feature] = flags

        # Identify rows with at least one outlier
        outlier_rows = outlier_records.any(axis=1)

        # Get actual values of all features for outliers
        outlier_data = df.loc[outlier_rows, :].copy()

        if outlier_data.empty:
            print("No Isolation Forest outliers found.")
            return pd.DataFrame()

        outlier_data["outlier_source"] = outlier_records.loc[outlier_rows].apply(
            lambda row: ", ".join(row.index[row]), axis=1
        )

        num_outliers = len(outlier_data)
        print(
            f"Total number of Isolation Forest outliers found: {num_outliers}")

        return outlier_data

    def detect_outliers_lof(self, df, features, n_neighbors=20, contamination=0.01):
        """Detect outliers using Local Outlier Factor (LOF) for numerical features."""
        all_outliers = []

        if not isinstance(features, list):
            print("Warning: Features must be a list.")
            return pd.DataFrame()

        for feature in features:
            if feature not in df.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            if pd.api.types.is_numeric_dtype(df[feature]):
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, contamination=contamination)
                outlier_pred = lof.fit_predict(
                    df[[feature]]) == -1  # -1 means outlier

                feature_outliers = df[outlier_pred]

                if not feature_outliers.empty:
                    outliers_df = feature_outliers.copy()
                    outliers_df["outlier_source"] = feature
                    all_outliers.append(outliers_df)
            else:
                print(f"Skipping LOF for non-numerical feature: {feature}")

        if not all_outliers:
            print("No LOF outliers found.")
            return pd.DataFrame()

        combined_outliers = pd.concat(all_outliers)

        grouped_sources = combined_outliers.groupby(combined_outliers.index)[
            "outlier_source"].apply(lambda x: ", ".join(set(x)))

        combined_outliers = combined_outliers.loc[grouped_sources.index].copy()
        combined_outliers["outlier_source"] = grouped_sources

        num_outliers = len(combined_outliers)
        print(f"Total number of LOF outliers found: {num_outliers}")

        return combined_outliers

    def remove_outliers(self, df, outliers_df):
        """
        Removes outliers from the DataFrame based on the provided outliers DataFrame.

        Args:
            df (pd.DataFrame): The original DataFrame.
            outliers_df (pd.DataFrame): DataFrame containing outlier rows.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed.
        """
        if outliers_df.empty:
            print("No outliers to remove.")
            return df.copy()  # Return a copy to avoid modifying the original

        outlier_indices = outliers_df.index
        cleaned_df = df.drop(index=outlier_indices)

        num_removed = len(outlier_indices)
        print(f"Removed {num_removed} outliers.")

        return cleaned_df
