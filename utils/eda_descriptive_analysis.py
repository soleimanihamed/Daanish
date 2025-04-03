# eda_descriptive_analysis.py
from utils.save_utils import SaveUtils
from tabulate import tabulate
import pandas as pd
import numpy as np
from fitter import Fitter


class DescriptiveEDAAnalysis:
    def __init__(self, data):
        self.data = data
        self.save_utils = SaveUtils()  # Initialize SaveUtils for saving reports

    def get_dataset_summary(self):
        """
        Provides a summary of the dataset, including:
        - Basic info (columns, non-null counts, data types)
        - Number of duplicate records
        - Missing values (count and percentage)
        - Number of duplicate records
        - Unique values per column
        """

        info = {
            "info": str(self.data.info(verbose=False, memory_usage=False)),
            "duplicate_count": self.data.duplicated().sum(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "missing_percentage": (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            "unique_values": self.data.nunique().to_dict()
        }
        return info

    def get_data_samples(self, sample_number=5):
        """Returns a sample of the DataFrame."""
        return self.data.sample(sample_number)

    def get_feature_summary(self, feature):
        """Returns extended descriptive statistics for a feature."""
        feature_data = self.data[feature]
        summary = feature_data.describe().to_dict()

        if pd.api.types.is_numeric_dtype(feature_data):
            summary.update({
                "skewness": float(feature_data.skew()),
                "kurtosis": float(feature_data.kurt()),
                "5%": float(feature_data.quantile(0.05)),
                "95%": float(feature_data.quantile(0.95)),
                "missing_values": int(feature_data.isna().sum()),
                "missing_percentage": float(feature_data.isna().mean() * 100),
                "distinct_count": int(feature_data.nunique()),
                "distinct_percentage": float((feature_data.nunique() / len(feature_data)) * 100),
                "zero_values": int((feature_data == 0).sum()),
                "zero_percentage": float((feature_data == 0).mean() * 100),
                "range": float(feature_data.max() - feature_data.min()),
                "iqr": float(feature_data.quantile(0.75) - feature_data.quantile(0.25)),
                "variance": float(feature_data.var()),
                "sum": float(feature_data.sum()),
                "most_frequent": {str(k): int(v) for k, v in feature_data.value_counts().nlargest(15).to_dict().items()},
                "largest_values": {str(k): int(v) for k, v in feature_data.value_counts().sort_index(ascending=False).head(15).to_dict().items()},
                "smallest_values": {str(k): int(v) for k, v in feature_data.value_counts().sort_index(ascending=True).head(15).to_dict().items()}
            })
        elif pd.api.types.is_categorical_dtype(feature_data) or feature_data.dtype == 'object':
            summary.update({
                "missing_values": int(feature_data.isna().sum()),
                "missing_percentage": float(feature_data.isna().mean() * 100),
                "distinct_count": int(feature_data.nunique()),
                "values": int(len(feature_data)),
                "categories": {str(k): int(v) for k, v in feature_data.value_counts().to_dict().items()}
            })
        elif feature_data.dtype == 'bool':
            feature_data = feature_data.astype(int)
            summary.update({
                "missing_values": int(feature_data.isna().sum()),
                "missing_percentage": float(feature_data.isna().mean() * 100),
                "distinct_count": int(feature_data.nunique()),
                "mode": int(feature_data.mode().iloc[0]),
                "mode_percentage": float((feature_data == feature_data.mode().iloc[0]).mean() * 100)
            })
        return summary

    def get_all_feature_summaries(self):
        """Returns a dictionary of feature summaries."""
        return {feature: self.get_feature_summary(feature) for feature in self.data.columns}

    # def extended_describe(self, feature):
    #     """
    #     Generates extended descriptive statistics for a feature.

    #     Args:
    #         feature (str): Feature name.

    #     Returns:
    #         pd.Series: Series with extended descriptive statistics.
    #     """
    #     feature_data = self.data[feature]
    #     summary = feature_data.describe()

    #     # Additional statistics for numerical features
    #     if feature_data.dtype in ['int64', 'float64']:
    #         summary['skewness'] = feature_data.skew()
    #         summary['kurtosis'] = feature_data.kurt()
    #         summary['5%'] = feature_data.quantile(0.05)
    #         summary['95%'] = feature_data.quantile(0.95)
    #         summary['missing_values'] = feature_data.isna().sum()
    #         summary['missing_percentage'] = feature_data.isna().mean() * 100
    #         summary['distinct_count'] = feature_data.nunique()
    #         summary['distinct_percentage'] = (
    #             feature_data.nunique() / len(feature_data)) * 100
    #         summary['zero_values'] = (feature_data == 0).sum()
    #         summary['zero_percentage'] = (feature_data == 0).mean() * 100
    #         summary['range'] = feature_data.max() - feature_data.min()
    #         summary['iqr'] = feature_data.quantile(
    #             0.75) - feature_data.quantile(0.25)
    #         summary['variance'] = feature_data.var()
    #         summary['sum'] = feature_data.sum()
    #         summary['most_frequent'] = [{
    #             'value': value,
    #             'count': count,
    #             'percentage': (count / len(feature_data)) * 100
    #         } for value, count in feature_data.value_counts().nlargest(15).items()]
    #         summary['largest_values'] = [{
    #             'value': value,
    #             'count': count,
    #             'percentage': (count / len(feature_data)) * 100
    #         } for value, count in feature_data.value_counts().sort_index(ascending=False).head(15).items()]

    #         summary['smallest_values'] = [{
    #             'value': value,
    #             'count': count,
    #             'percentage': (count / len(feature_data)) * 100
    #         } for value, count in feature_data.value_counts().sort_index(ascending=True).head(15).items()]

    #     # Additional statistics for categorical features
    #     elif feature_data.dtype == 'object':
    #         summary['missing_values'] = feature_data.isna().sum()
    #         summary['missing_percentage'] = feature_data.isna().mean() * 100
    #         summary['distinct_count'] = feature_data.nunique()
    #         value_counts = feature_data.value_counts()
    #         summary['values'] = len(feature_data)
    #         summary['categories'] = [{
    #             'category': index,
    #             'count': count,
    #             'percentage': (count / len(feature_data)) * 100
    #         } for index, count in value_counts.items()]

    #     # Additional statistics for boolean features
    #     elif feature_data.dtype == 'bool':
    #         # Convert boolean to int for numerical calculations
    #         feature_data = feature_data.astype(int)
    #         summary['missing_values'] = feature_data.isna().sum()
    #         summary['missing_percentage'] = feature_data.isna().mean() * 100
    #         summary['distinct_count'] = feature_data.nunique()

    #         # Most frequent value
    #         summary['mode'] = feature_data.mode().iloc[0]
    #         summary['mode_percentage'] = (
    #             feature_data == summary['mode']).mean() * 100

    #     return summary

    # def generate_summary(self):
    #     """
    #     Generates extended descriptive statistics for all features in the dataset.

    #     Returns:
    #         dict: Dictionary with feature names as keys and their extended descriptive statistics as values.
    #     """
    #     summary_dict = {}
    #     for feature in self.data.columns:
    #         summary_dict[feature] = self.extended_describe(feature).to_dict()
    #     return summary_dict

    # def save_summary_to_json(self, output_file):
    #     """
    #     Saves the extended descriptive statistics to a JSON file.

    #     Args:
    #         output_file (str): Path to the output JSON file.
    #     """
    #     summary = self.generate_summary()
    #     self.save_utils.save_json(summary, output_file)

    # def print_detailed_summary(self, summary=None):
    #     """
    #     Prints the extended descriptive statistics in a well-formatted way.

    #     Args:
    #         summary (dict, optional): Dictionary containing the summary statistics. If None, generates the summary.
    #     """
    #     if summary is None:
    #         summary = self.generate_summary()

    #     for feature, stats in summary.items():
    #         print(f"\nFeature: {feature}")
    #         if isinstance(stats, dict):
    #             # Print basic statistics in a table
    #             basic_stats = {k: v for k,
    #                            v in stats.items() if not isinstance(v, list)}
    #             print(tabulate(basic_stats.items(), headers=[
    #                 "Statistic", "Value"], tablefmt="pretty"))

    #             # Print most_frequent values in a table
    #             if "most_frequent" in stats:
    #                 print("\nMost Frequent Values:")
    #                 print(tabulate(stats["most_frequent"],
    #                                headers="keys", tablefmt="pretty"))

    #             # Print largest_values in a table
    #             if "largest_values" in stats:
    #                 print("\nLargest Values:")
    #                 print(tabulate(stats["largest_values"],
    #                                headers="keys", tablefmt="pretty"))

    #             # Print smallest_values in a table
    #             if "smallest_values" in stats:
    #                 print("\nSmallest Values:")
    #                 print(tabulate(stats["smallest_values"],
    #                                headers="keys", tablefmt="pretty"))

    # def print_high_level_summary(self, summary=None):
    #     """
    #     Prints a high-level summary of the dataset, with each feature as a column and first-level statistics as rows.

    #     Args:
    #         summary (dict, optional): Dictionary containing the summary statistics. If None, generates the summary.
    #     """
    #     if summary is None:
    #         summary = self.generate_summary()

    #     # Extract first-level statistics for each feature
    #     high_level_stats = {}
    #     for feature, stats in summary.items():
    #         if isinstance(stats, dict):
    #             # Filter only the first-level statistics (from count to sum)
    #             high_level_stats[feature] = {
    #                 k: v for k, v in stats.items()
    #                 if k in [
    #                     'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
    #                     'skewness', 'kurtosis', '5%', '95%', 'missing_values',
    #                     'missing_percentage', 'distinct_count', 'distinct_percentage',
    #                     'zero_values', 'zero_percentage', 'range', 'iqr', 'variance', 'sum'
    #                 ]
    #             }

    #     # Convert the high-level stats into a tabular format
    #     table_data = []
    #     for stat in high_level_stats[next(iter(high_level_stats))].keys():
    #         row = [stat]  # First column is the statistic name
    #         for feature in high_level_stats.keys():
    #             row.append(high_level_stats[feature].get(
    #                 stat, 'N/A'))  # Add feature values
    #         table_data.append(row)

    #     # Print the table
    #     headers = ["Statistic"] + list(high_level_stats.keys())
    #     print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    # def save_high_level_summary_to_csv(self, output_file, summary=None):
    #     """
    #     Saves a high-level summary of the dataset to a CSV file with features as columns and statistics as rows.

    #     Args:
    #         output_file (str): Path to the output CSV file.
    #         summary (dict, optional): Dictionary containing the summary statistics. If None, generates the summary.
    #     """
    #     if summary is None:
    #         summary = self.generate_summary()

    #     # Extract first-level statistics for each feature
    #     high_level_stats = {}
    #     for feature, stats in summary.items():
    #         if isinstance(stats, dict):
    #             # Select only relevant numerical summary statistics
    #             high_level_stats[feature] = {
    #                 k: v for k, v in stats.items()
    #                 if k in [
    #                     'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
    #                     'skewness', 'kurtosis', '5%', '95%', 'missing_values',
    #                     'missing_percentage', 'distinct_count', 'distinct_percentage',
    #                     'zero_values', 'zero_percentage', 'range', 'iqr', 'variance', 'sum'
    #                 ]
    #             }

    #     # Convert dictionary to DataFrame
    #     # Features are now **columns**, stats are **rows**
    #     df_summary = pd.DataFrame(high_level_stats)

    #     # Save to CSV
    #     df_summary.to_csv(output_file, index=True)

    #     print(f"High-level summary saved to {output_file}")

    def fit_best_distribution(self, numeric_variables: list, method='sumsquare_error', common_distributions=True, timeout=60):
        """
        Fits the best probability distribution for each numeric variable in the input list.

        Parameters:
            numeric_variables (list): List of column names to analyze.
            method (str): Method to find the best fit ('sumsquare_error', 'aic', 'bic').
            common_distributions (bool): If True, uses only common distributions to avoid long runtime.
            timeout (int): Timeout for fitting each distribution (in seconds).

        Returns:
            dict: A dictionary containing the best-fitting distribution and its parameters for each variable.
        """

        # Dictionary to store results
        results = {}

        # Common distributions (to avoid very slow ones)
        common_dists = [
            'norm', 'expon', 'lognorm', 'gamma', 'beta', 'weibull_min', 'chi2', 'pareto',
            'uniform', 't', 'gumbel_r', 'burr', 'invgauss', 'triang', 'laplace', 'logistic',
            'genextreme', 'skewnorm', 'genpareto', 'burr12', 'fatiguelife', 'geninvgauss',
            'halfnorm', 'exponpow'
        ]

        # Iterate over the input variables
        for variable in numeric_variables:
            # Check if the variable is numeric
            if pd.api.types.is_numeric_dtype(self.data[variable]):
                print(f"\n=== Analyzing {variable} ===")

                # Drop missing values
                data = self.data[variable].dropna()

                # Select distributions based on user choice
                distributions = common_dists if common_distributions else None

                # Initialize Fitter
                fitter = Fitter(
                    data, distributions=distributions, timeout=timeout)

                try:
                    # Fit the distributions
                    fitter.fit()

                    # Get the best-fitting distribution
                    # You can also use 'aic' or 'bic'
                    best_fit = fitter.get_best(method)

                # Check if a best fit was found
                    if best_fit:
                        # Store results
                        results[variable] = {
                            # Get the name of the best-fitting distribution
                            "best_distribution": list(best_fit.keys())[0],
                            # Get the parameters of the best-fitting distribution
                            "parameters": list(best_fit.values())[0]
                        }

                        # Print results
                        print(
                            f"Best-fitting distribution: {list(best_fit.keys())[0]}")
                        print(f"Parameters: {list(best_fit.values())[0]}")
                    else:
                        results[variable] = {
                            "best_distribution": None,
                            "parameters": None
                        }
                        print(
                            f"No suitable distribution found for {variable}.")
                except Exception as e:
                    print(f"Error fitting distributions for {variable}: {e}")
                    results[variable] = {
                        "best_distribution": None, "parameters": None}

            else:
                print(f"\n=== Skipping {variable} (not numeric) ===")

        # Convert results dictionary to a well-structured DataFrame
        if results:
            results_df = pd.DataFrame([
                {"Feature": feature, "Best Distribution":
                    data["best_distribution"], "Parameters": data["parameters"]}
                for feature, data in results.items()
            ])

            # Print formatted results
            print("\n=== Distribution Fitting Results ===")
            # Removes default index for clarity
            print(results_df.to_string(index=False))
        else:
            print("No valid distributions were found for the provided numeric variables.")

        return results
