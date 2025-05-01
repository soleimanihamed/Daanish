# utils/features/selector.py

class FeatureSelector:
    """
    Performs feature selection and removal based on configurable criteria.
    """

    def __init__(self, data):
        self.data = data.copy()

    def drop_features(self, features):
        """
        Drops the specified features from the dataset.
        """
        self.data.drop(columns=features, inplace=True)
        return self.data

    def select_by_missing_rate(self, threshold=0.3):
        """
        Returns features with missing rate above threshold.
        """
        return [
            col for col in self.data.columns
            if self.data[col].isnull().mean() > threshold
        ]

    def select_by_outlier_rate(self, outliers_df, threshold=0.2):
        """
        Returns features with outlier rate above threshold.
        """
        total = len(self.data)
        return [
            col for col in outliers_df.columns
            if outliers_df[col].notna().sum() / total > threshold
        ]
