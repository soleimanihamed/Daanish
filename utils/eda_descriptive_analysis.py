# eda_descriptive_analysis.py

from utils.save_utils import SaveUtils
from tabulate import tabulate
import pandas as pd


class DescriptiveEDAAnalysis:
    def __init__(self, data):
        self.data = data
        self.save_utils = SaveUtils()  # Initialize SaveUtils for saving reports

    def print_data_samples(self, data, sample_number=5):
        """
        Print samples of the DataFrame.

        Args:
            data: The dataset.

        """

        print("Sample of the DataFrame:")
        print(data.sample(sample_number))

    def extended_describe(self, feature):
        """
        Generates extended descriptive statistics for a feature.

        Args:
            feature (str): Feature name.

        Returns:
            pd.Series: Series with extended descriptive statistics.
        """
        feature_data = self.data[feature]
        summary = feature_data.describe()

        # Additional statistics for numerical features
        if feature_data.dtype in ['int64', 'float64']:
            summary['skewness'] = feature_data.skew()
            summary['kurtosis'] = feature_data.kurt()
            summary['5%'] = feature_data.quantile(0.05)
            summary['95%'] = feature_data.quantile(0.95)
            summary['missing_values'] = feature_data.isna().sum()
            summary['missing_percentage'] = feature_data.isna().mean() * 100
            summary['distinct_count'] = feature_data.nunique()
            summary['distinct_percentage'] = (
                feature_data.nunique() / len(feature_data)) * 100
            summary['zero_values'] = (feature_data == 0).sum()
            summary['zero_percentage'] = (feature_data == 0).mean() * 100
            summary['range'] = feature_data.max() - feature_data.min()
            summary['iqr'] = feature_data.quantile(
                0.75) - feature_data.quantile(0.25)
            summary['variance'] = feature_data.var()
            summary['sum'] = feature_data.sum()
            summary['most_frequent'] = [{
                'value': value,
                'count': count,
                'percentage': (count / len(feature_data)) * 100
            } for value, count in feature_data.value_counts().nlargest(15).items()]
            summary['largest_values'] = [{
                'value': value,
                'count': count,
                'percentage': (count / len(feature_data)) * 100
            } for value, count in feature_data.value_counts().sort_index(ascending=False).head(15).items()]

            summary['smallest_values'] = [{
                'value': value,
                'count': count,
                'percentage': (count / len(feature_data)) * 100
            } for value, count in feature_data.value_counts().sort_index(ascending=True).head(15).items()]

        # Additional statistics for categorical features
        elif feature_data.dtype == 'object':
            summary['missing_values'] = feature_data.isna().sum()
            summary['missing_percentage'] = feature_data.isna().mean() * 100
            summary['distinct_count'] = feature_data.nunique()
            value_counts = feature_data.value_counts()
            summary['values'] = len(feature_data)
            summary['categories'] = [{
                'category': index,
                'count': count,
                'percentage': (count / len(feature_data)) * 100
            } for index, count in value_counts.items()]

        return summary

    def generate_summary(self):
        """
        Generates extended descriptive statistics for all features in the dataset.

        Returns:
            dict: Dictionary with feature names as keys and their extended descriptive statistics as values.
        """
        summary_dict = {}
        for feature in self.data.columns:
            summary_dict[feature] = self.extended_describe(feature).to_dict()
        return summary_dict

    def save_summary_to_json(self, output_file):
        """
        Saves the extended descriptive statistics to a JSON file.

        Args:
            output_file (str): Path to the output JSON file.
        """
        summary = self.generate_summary()
        self.save_utils.save_json(summary, output_file)

    def print_detailed_summary(self, summary=None):
        """
        Prints the extended descriptive statistics in a well-formatted way.

        Args:
            summary (dict, optional): Dictionary containing the summary statistics. If None, generates the summary.
        """
        if summary is None:
            summary = self.generate_summary()

        for feature, stats in summary.items():
            print(f"\nFeature: {feature}")
            if isinstance(stats, dict):
                # Print basic statistics in a table
                basic_stats = {k: v for k,
                               v in stats.items() if not isinstance(v, list)}
                print(tabulate(basic_stats.items(), headers=[
                    "Statistic", "Value"], tablefmt="pretty"))

                # Print most_frequent values in a table
                if "most_frequent" in stats:
                    print("\nMost Frequent Values:")
                    print(tabulate(stats["most_frequent"],
                                   headers="keys", tablefmt="pretty"))

                # Print largest_values in a table
                if "largest_values" in stats:
                    print("\nLargest Values:")
                    print(tabulate(stats["largest_values"],
                                   headers="keys", tablefmt="pretty"))

                # Print smallest_values in a table
                if "smallest_values" in stats:
                    print("\nSmallest Values:")
                    print(tabulate(stats["smallest_values"],
                                   headers="keys", tablefmt="pretty"))

    def print_high_level_summary(self, summary=None):
        """
        Prints a high-level summary of the dataset, with each feature as a column and first-level statistics as rows.

        Args:
            summary (dict, optional): Dictionary containing the summary statistics. If None, generates the summary.
        """
        if summary is None:
            summary = self.generate_summary()

        # Extract first-level statistics for each feature
        high_level_stats = {}
        for feature, stats in summary.items():
            if isinstance(stats, dict):
                # Filter only the first-level statistics (from count to sum)
                high_level_stats[feature] = {
                    k: v for k, v in stats.items()
                    if k in [
                        'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                        'skewness', 'kurtosis', '5%', '95%', 'missing_values',
                        'missing_percentage', 'distinct_count', 'distinct_percentage',
                        'zero_values', 'zero_percentage', 'range', 'iqr', 'variance', 'sum'
                    ]
                }

        # Convert the high-level stats into a tabular format
        table_data = []
        for stat in high_level_stats[next(iter(high_level_stats))].keys():
            row = [stat]  # First column is the statistic name
            for feature in high_level_stats.keys():
                row.append(high_level_stats[feature].get(
                    stat, 'N/A'))  # Add feature values
            table_data.append(row)

        # Print the table
        headers = ["Statistic"] + list(high_level_stats.keys())
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))

    def save_high_level_summary_to_csv(self, output_file, summary=None):
        """
        Saves a high-level summary of the dataset to a CSV file with features as columns and statistics as rows.

        Args:
            output_file (str): Path to the output CSV file.
            summary (dict, optional): Dictionary containing the summary statistics. If None, generates the summary.
        """
        if summary is None:
            summary = self.generate_summary()

        # Extract first-level statistics for each feature
        high_level_stats = {}
        for feature, stats in summary.items():
            if isinstance(stats, dict):
                # Select only relevant numerical summary statistics
                high_level_stats[feature] = {
                    k: v for k, v in stats.items()
                    if k in [
                        'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
                        'skewness', 'kurtosis', '5%', '95%', 'missing_values',
                        'missing_percentage', 'distinct_count', 'distinct_percentage',
                        'zero_values', 'zero_percentage', 'range', 'iqr', 'variance', 'sum'
                    ]
                }

        # Convert dictionary to DataFrame
        # Features are now **columns**, stats are **rows**
        df_summary = pd.DataFrame(high_level_stats)

        # Save to CSV
        df_summary.to_csv(output_file, index=True)

        print(f"High-level summary saved to {output_file}")
