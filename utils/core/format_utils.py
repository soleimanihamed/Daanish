# utils/core/format_utils.py

import pandas as pd


class FormatUtils:
    @staticmethod
    def high_level_summary_to_dataframe(summary: dict) -> pd.DataFrame:
        """
        Converts high-level summary dict to a structured pandas DataFrame for saving or displaying.

        Args:
            summary (dict): Summary dict from EDA.

        Returns:
            pd.DataFrame: Formatted DataFrame.
        """
        selected_keys = [
            'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max',
            'skewness', 'kurtosis', '5%', '95%', 'missing_values',
            'missing_percentage', 'distinct_count', 'distinct_percentage',
            'zero_values', 'zero_percentage', 'range', 'iqr', 'variance', 'sum'
        ]

        high_level_stats = {
            feature: {k: v for k, v in stats.items() if k in selected_keys}
            for feature, stats in summary.items()
            if isinstance(stats, dict)
        }

        df_summary = pd.DataFrame(high_level_stats)
        df_summary = df_summary.reset_index().rename(
            columns={'index': 'Statistic'})

        return df_summary
