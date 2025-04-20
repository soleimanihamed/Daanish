# generate_reports.py

from tabulate import tabulate
import pandas as pd
from utils.core.save_manager import SaveUtils


class ReportGenerator:
    """
    A utility class for generating formatted reports from data analysis summaries.
    """

    @staticmethod
    def high_level_summary_to_dataframe(summary: dict) -> pd.DataFrame:
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
