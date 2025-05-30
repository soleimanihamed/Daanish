# uitls/eda/outlier_detection.py

import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from utils.eda.statistical import StatisticalAnalysis


class OutlierDetector:
    """
    Detects outliers in numerical data using a variety of statistical and machine learning techniques:
        - IQR method
        - Z-score (with Shapiro-Wilk normality test)
        - Fitted statistical distributions
        - Custom threshold-based detection
        - Isolation Forest (planned)
        - Local Outlier Factor (planned)
    """

    def __init__(self, data):
        self.data = data

    def detect_outliers_iqr(self, features, threshold=1.5):
        """
        Detect outliers using the IQR method for each numerical feature.

        Args:
            features (list): List of numerical features to analyze.
            threshold (float): Multiplier for the IQR to define outlier boundaries.

        Returns:
            pd.DataFrame: Outliers with their source feature tagged.
        """

        df = self.data

        all_outliers = []

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
                    all_outliers.append(outliers_df)
            else:
                print(f"Skipping IQR for non-numerical feature: {feature}")

        if not all_outliers:
            print("No numerical outliers found.")
            return pd.DataFrame()

        return self._combine_outlier_frames(all_outliers, message="Total number of IQR-based outliers found")

    def detect_custom_outliers(self, features, lower_bounds=None, upper_bounds=None):
        """
        Detect outliers using user-defined lower and/or upper bounds for each feature.

        Args:
            features (list): List of features to apply custom bounds to.
            lower_bounds (dict): Optional dictionary of lower bounds.
            upper_bounds (dict): Optional dictionary of upper bounds.

        Returns:
            pd.DataFrame: Outliers tagged with the feature name that triggered detection.
        """

        df = self.data

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
                lower = lower_bounds.get(feature, float(
                    '-inf')) if lower_bounds else float('-inf')
                upper = upper_bounds.get(feature, float(
                    'inf')) if upper_bounds else float('inf')

                feature_outliers = df[(df[feature] < lower)
                                      | (df[feature] > upper)]

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

        return self._combine_outlier_frames(all_outliers, message="Total number of Custom-based outliers found")

    def detect_outliers_zscore(self, features, threshold=3, alpha=0.05):
        """
        Detect outliers using the Z-score method, with prior normality check (Shapiro-Wilk).

        Args:
            features (list): List of numerical features.
            threshold (float): Z-score threshold.
            alpha (float): Significance level for the Shapiro-Wilk normality test.

        Returns:
            pd.DataFrame: DataFrame of detected outliers.
        """

        df = self.data

        all_outliers = []
        eda = StatisticalAnalysis(df)

        if not isinstance(features, list):
            print("Warning: Features must be a list.")
            return pd.DataFrame()

        # call normality test
        normality_results = eda.perform_normality_test_Shapiro_Wilk(
            features, alpha)

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
                        f"Warning: Feature '{feature}' does not appear to be normally distributed "
                        f"(p-value={normality_results[feature]['shapiro_p_value']:.4f}). "
                        "Z-score may be inappropriate."
                    )
            else:
                print(f"Skipping Z-score for non-numerical feature: {feature}")

        if not all_outliers:
            print("No Z-score outliers found.")
            return pd.DataFrame()

        return self._combine_outlier_frames(all_outliers, message="Total number of Z-Score-based outliers found")

    def detect_outliers_isolation_forest(self, features, contamination='auto', max_samples=0.8, n_estimators=200, random_state=42):
        """
        Detects outliers in numerical features using the Isolation Forest algorithm.

        Args:
            features (list): List of numerical feature names to evaluate.
            contamination (float or 'auto'): Proportion of expected outliers in the data.
            max_samples (float or int): The number of samples to draw to train each base estimator.
            n_estimators (int): Number of base estimators in the ensemble.
            random_state (int): Random state for reproducibility.

        Returns:
            pd.DataFrame: DataFrame containing the outlier rows with a new 'outlier_source' column.
        """

        df = self.data

        if not isinstance(features, list):
            print("Warning: Features must be a list.")
            return pd.DataFrame()

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
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

        # Combine flags into a DataFrame
        outlier_records = pd.DataFrame(index=df.index)

        for feature, flags in outlier_flags.items():
            if feature in df.columns:  # prevent error if non numerical features were passed.
                outlier_records[feature] = flags

        # Find rows with any outlier flags
        outlier_rows = outlier_records.any(axis=1)

        # Get actual values of all features for outliers
        outlier_data = df.loc[outlier_rows, :].copy()

        if outlier_data.empty:
            print("No Isolation Forest outliers found.")
            return pd.DataFrame()

        outlier_data["outlier_source"] = outlier_records.loc[outlier_rows].apply(
            lambda row: ", ".join(row.index[row]), axis=1
        )

        return self._combine_outlier_frames(
            [outlier_data], message="Total number of Isolation Forest-based outliers found"
        )

    def detect_outliers_lof(self, features, n_neighbors=20, contamination=0.01):
        """
        Detects outliers in numerical features using the Local Outlier Factor (LOF) algorithm.

        Args:
            features (list): List of numerical feature names to evaluate.
            n_neighbors (int): Number of neighbors to use for LOF.
            contamination (float): Estimated proportion of outliers in the data.

        Returns:
            pd.DataFrame: DataFrame containing the outlier rows with a new 'outlier_source' column.
         """

        df = self.data

        if not isinstance(features, list):
            print("Warning: Features must be a list.")
            return pd.DataFrame()

        all_outliers = []

        for feature in features:
            if feature not in df.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            if pd.api.types.is_numeric_dtype(df[feature]):
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors, contamination=contamination)
                predictions = lof.fit_predict(
                    df[[feature]]) == -1  # -1 means outlier

                outlier_subset = df[predictions].copy()

                if not outlier_subset.empty:
                    outlier_subset["outlier_source"] = feature
                    all_outliers.append(outlier_subset)

            else:
                print(f"Skipping non-numeric feature '{feature}' in LOF.")

        if not all_outliers:
            print("No LOF outliers found.")
            return pd.DataFrame()

        return self._combine_outlier_frames(all_outliers, message="Total number of LOF-based outliers found")

    def detect_outliers_distribution(self, distribution_results, confidence_interval=0.95):
        """
        Detects outliers based on fitted probability distributions for multiple features.

        Args:
            distribution_results (dict): Dictionary mapping feature names to their best-fitting distribution and parameters.
            confidence_interval (float): Confidence interval for outlier detection.

        Returns:
            pd.DataFrame: DataFrame containing combined outliers with an 'outlier_source' column.
        """

        distribution_mapping = self._get_all_distributions()
        df = self.data
        all_outliers = []
        skipped_distributions = []

        for feature, result in distribution_results.items():
            if result["best_distribution"] is None:
                skipped_distributions.append(
                    (feature, "No distribution found"))
                continue

            dist_name = result["best_distribution"]
            params = result["parameters"]
            dist = distribution_mapping.get(dist_name)

            if dist is None:
                skipped_distributions.append(
                    (feature, f"{dist_name} not found"))
                continue

            try:
                filtered_data = df[feature].dropna()
                shape_keys = [k for k in params if k not in ["loc", "scale"]]
                shape_vals = [params[k] for k in shape_keys]
                loc = params.get("loc", 0)
                scale = params.get("scale", 1)

                # Use dynamic bounds
                lower = dist.ppf((1 - confidence_interval) / 2,
                                 *shape_vals, loc=loc, scale=scale)
                upper = dist.ppf((1 + confidence_interval) / 2,
                                 *shape_vals, loc=loc, scale=scale)

                outliers = filtered_data[(filtered_data < lower) | (
                    filtered_data > upper)]
                if not outliers.empty:
                    outliers_df = df.loc[outliers.index]
                    outliers_df["outlier_source"] = feature
                    all_outliers.append(outliers_df)

            except Exception as e:
                skipped_distributions.append(
                    (feature, f"{dist_name} error: {e}"))

        combined_outliers = self._combine_outlier_frames(
            all_outliers, message="Total number of distribution-based outliers found"
        )

        if skipped_distributions:
            print("\nDistributions Skipped:")
            for feature, reason in skipped_distributions:
                print(f"Feature: {feature}, Reason: {reason}")

        return combined_outliers

    def _get_all_distributions(self):
        """
        Returns a dictionary mapping distribution names to SciPy distribution objects.
        Filters out any invalid or inaccessible attributes.
        """
        all_dist_names = [
            name for name in dir(stats)
            if isinstance(getattr(stats, name), stats.rv_continuous)
        ]

        return {name: getattr(stats, name) for name in all_dist_names}

    def _combine_outlier_frames(self, outlier_frames, message=None):
        """
        Combines a list of outlier DataFrames, aggregates outlier sources per record.

        Args:
            outlier_frames (list): List of DataFrames, each with an 'outlier_source' column.
            message (str, optional): Optional message to print with the number of combined outliers.

        Returns:
            pd.DataFrame: Combined and aggregated outlier records.
        """
        if not outlier_frames:
            return pd.DataFrame()

        combined_outliers = pd.concat(outlier_frames)

        grouped_sources = combined_outliers.groupby(combined_outliers.index)[
            "outlier_source"].apply(lambda x: ", ".join(set(x)))

        combined_outliers = combined_outliers.loc[grouped_sources.index].copy()
        combined_outliers["outlier_source"] = grouped_sources

        if message:
            print(f"{message}: {len(combined_outliers)}")

        return combined_outliers

    def detect_outliers_featurewise(self, method_config, distribution_results=None):
        """
        Detects outliers per feature using different methods and parameters.

        Args:
            method_config (dict): Mapping of feature name to dict with keys:
                                - 'method': str, detection method name
                                - 'params': dict, method-specific parameters
            distribution_results (dict): Optional pre-fitted distribution results for distribution-based methods.

        Returns:
            pd.DataFrame: Combined outlier DataFrame across all methods.
        """

        all_outliers = []

        for feature, config in method_config.items():
            method = config.get("method")
            params = config.get("params", {})
            outliers = pd.DataFrame()

            if method == "iqr":
                outliers = self.detect_outliers_iqr([feature], **params)

            elif method == "zscore":
                outliers = self.detect_outliers_zscore([feature], **params)

            elif method == "custom":
                outliers = self.detect_custom_outliers([feature],
                                                       lower_bounds={
                                                           feature: params.get("lower")},
                                                       upper_bounds={feature: params.get("upper")})
            elif method == "distribution":
                if distribution_results and feature in distribution_results:
                    dist_result = distribution_results[feature]

                else:
                    eda_stat = StatisticalAnalysis(self.data)
                    dist_result = eda_stat.fit_best_distribution(
                        feature,
                        method='sumsquare_error',
                        common_distributions=True,
                        timeout=60
                    )

                outliers = self.detect_outliers_distribution(
                    {feature: dist_result},
                    confidence_interval=params.get("confidence_interval", 0.95)
                )

            elif method == "isolation":
                outliers = self.detect_outliers_isolation_forest(
                    [feature], **params)

            elif method == "lof":
                outliers = self.detect_outliers_lof([feature], **params)

            else:
                print(f"Unknown method '{method}' for feature '{feature}'")
                continue

            if not outliers.empty:
                all_outliers.append(outliers)

        return self._combine_outlier_frames(all_outliers, message="Total combined outliers found across all methods")
