import pandas as pd
import numpy as np
from utils.save_utils import SaveUtils


class CustomEDAAnalysis:
    def __init__(self, data):
        self.data = data
        self.save_utils = SaveUtils()  # Initialize SaveUtils for saving reports

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
