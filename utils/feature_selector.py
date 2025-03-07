# utils/feature_selector.py

class FeatureSelector:
    def __init__(self, data):
        """
        Initializes the FeatureSelector with the dataset.

        Args:
            data (pd.DataFrame): The dataset containing the features.
        """
        self.data = data

    def select_features(self, selected_features=None, exclude_features=None):
        """
        Selects a subset of features from the dataset.

        Args:
            selected_features (list, optional): A list of features to include.
            exclude_features (list, optional): A list of features to exclude.

        Returns:
            pd.DataFrame: Dataset containing only the selected features.
        """
        if selected_features:
            return self.data[selected_features]
        elif exclude_features:
            return self.data[[f for f in self.data.columns if f not in exclude_features]]
        else:
            return self.data
