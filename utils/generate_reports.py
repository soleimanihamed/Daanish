# generate_reports.py

from tabulate import tabulate
import pandas as pd
from utils.save_utils import SaveUtils


class ReportGenerator:
    """
    A utility class for generating formatted reports from data analysis summaries.
    """

    @staticmethod
    def print_dataset_summary(summary):
        """
        Prints a summary of the entire dataset to the console.

        Args:
            summary (dict): A dictionary containing dataset summary statistics.
                           Expected keys include 'info', 'duplicate_count', 'missing_values',
                           'missing_percentage', and 'unique_values'.
        """
        print("=== Dataset Summary ===")
        print(summary["info"])
        print(f"\nDuplicate Count: {summary['duplicate_count']}")
        print("\nMissing Values:")
        print(tabulate(summary["missing_values"].items(),
              headers=["Feature", "Count"]))
        print("\nMissing Percentages:")
        print(tabulate(summary["missing_percentage"].items(), headers=[
              "Feature", "Percentage"]))
        print("\nUnique Values:")
        print(tabulate(summary["unique_values"].items(),
              headers=["Feature", "Count"]))

    @staticmethod
    def print_feature_summary(feature, summary):
        """
        Prints a detailed summary of a single feature to the console.

        Args:
            feature (str): The name of the feature.
            summary (dict): A dictionary containing summary statistics for the feature.
                           Expected keys include basic statistics (e.g., 'mean', 'std'),
                           'most_frequent', 'largest_values', and 'smallest_values' (if available).
        """
        print(f"\nFeature: {feature}")
        basic_stats = {k: v for k,
                       v in summary.items() if not isinstance(v, dict)}
        print(tabulate(basic_stats.items(), headers=[
              "Statistic", "Value"], tablefmt="pretty"))

        if "most_frequent" in summary:
            print("\nMost Frequent Values:")
            print(
                tabulate(summary["most_frequent"].items(), headers=["Value", "Count"]))
        if "largest_values" in summary:
            print("\nLargest Values:")
            print(
                tabulate(summary["largest_values"].items(), headers=["Value", "Count"]))
        if "smallest_values" in summary:
            print("\nSmallest Values:")
            print(
                tabulate(summary["smallest_values"].items(), headers=["Value", "Count"]))

    @staticmethod
    def print_high_level_summary(all_summaries):
        """
        Prints a high-level summary of all features in a tabular format to the console.

        Args:
            all_summaries (dict): A dictionary where keys are feature names and values are
                                  dictionaries of summary statistics.
        """
        high_level_stats = {}
        first_feature = next(iter(all_summaries))
        for stat in all_summaries[first_feature]:
            if not isinstance(all_summaries[first_feature][stat], dict) and not isinstance(all_summaries[first_feature][stat], dict):
                high_level_stats[stat] = [all_summaries[feature].get(
                    stat, "N/A") for feature in all_summaries]

        table_data = [["Statistic"] + list(all_summaries.keys())]
        for stat, values in high_level_stats.items():
            table_data.append([stat] + values)
        print(tabulate(table_data[1:],
              headers=table_data[0], tablefmt="pretty"))

    @staticmethod
    def save_high_level_summary_to_csv(summary, output_file):
        """
        Saves a high-level summary to a CSV file with features as columns and statistics as rows.

        Args:
            summary (dict): A dictionary where keys are feature names and values are
                            dictionaries of summary statistics.
            output_file (str): The path to the output CSV file.
        """
        high_level_stats = {}
        for feature, stats in summary.items():
            if isinstance(stats, dict):
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
        df_summary = pd.DataFrame(high_level_stats)

        df_summary = df_summary.reset_index()

        df_summary = df_summary.rename(columns={'index': 'Statistic'})

        save_utils = SaveUtils()
        save_utils.save_dataframe_to_csv(
            df_summary, output_file, overwrite=True)
