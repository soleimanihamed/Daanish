# eda_descriptive_analysis.py
from utils.save_utils import SaveUtils
from tabulate import tabulate
import pandas as pd
import numpy as np


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
