# utils/importance/woe_encoder.py

import numpy as np
import pandas as pd


class WOEEncoder:
    """
    Weight of Evidence (WOE) encoder for binned categorical/numerical features.

    This class maps each bin to its corresponding WOE value based on the target variable
    and stores WOE values for later use in modeling or interpretation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing binned features and the target variable.

    target : str
        Name of the binary target variable (0/1).

    Attributes
    ----------
    woe_maps : dict
        Dictionary mapping each feature to a dictionary of {bin: WOE value}.

    df : pd.DataFrame
        A copy of the input dataframe with optional new WOE columns after transformation.

    Notes
    -----
    - Bins must be discrete integers or categories created via previous binning steps.
    - Target must be binary, where 1 = "bad" (e.g., default), 0 = "good".
    """

    def __init__(self, df, target):
        self.df = df.copy()
        self.target = target
        self.woe_maps = {}

    def fit(self, binned_features):
        """
        Fit WOE encoder by computing WOE values for each bin in each feature.

        Parameters
        ----------
        binned_features : list of str
            List of binned feature column names (e.g., ['loan_amnt_binned']).
        """
        for feature in binned_features:
            grouped = self.df.groupby(
                feature)[self.target].agg(['sum', 'count'])
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']

            total_good = grouped['good'].sum()
            total_bad = grouped['bad'].sum()

            grouped['dist_good'] = grouped['good'] / total_good
            grouped['dist_bad'] = grouped['bad'] / total_bad

            # Add small value to avoid log(0)
            grouped['WOE'] = np.log(
                (grouped['dist_good'] + 1e-6) / (grouped['dist_bad'] + 1e-6))

            # Save the map
            self.woe_maps[feature] = grouped['WOE'].to_dict()

    def transform(self, features=None):
        """
        Apply the WOE encoding to the specified binned features.

        Parameters
        ----------
        features : list of str, optional
            List of features to transform. If None, transforms all fitted features.
        """
        if features is None:
            features = list(self.woe_maps.keys())

        for feature in features:
            woe_map = self.woe_maps.get(feature)
            if woe_map is None:
                print(f"No WOE map found for {feature}. Skipping.")
                continue
            self.df[f"{feature}_woe"] = self.df[feature].map(woe_map)

    def get_woe_mapping(self, feature=None):
        """
        Retrieve the WOE mapping for a specific feature or all features.

        Parameters
        ----------
        feature : str, optional
            Name of the feature to retrieve the mapping for. 
            If None, returns all WOE mappings.
        -------
        dict
            - If feature is provided: {bin: WOE value} for that feature.
            - If feature is None: {feature_name: {bin: WOE value}, ...} for all features.
        """
        if feature:
            return self.woe_maps.get(feature, None)
        return self.woe_maps

    def get_transformed_data(self):
        """
        Get the DataFrame with WOE-transformed features.

        Returns
        -------
        pd.DataFrame
            DataFrame including original and WOE-encoded columns.
        """
        return self.df
