# utils/features/binning.py

import pandas as pd
import numpy as np
from optbinning import BinningProcess, OptimalBinning


class Binner:
    """
    A class to handle binning of numerical and categorical features in a DataFrame.

    Supports various binning methods including equal width, quantile, decile, optimal (supervised),
    and manual categorical binning.

    Attributes:
        df (pd.DataFrame): The input dataset.
        config (dict): Configuration specifying binning methods and parameters.
        binning_info (dict): Metadata about the applied binning operations.
    """

    def __init__(self, df, config):
        self.df = df.copy()
        self.config = config
        self.binning_info = {}

    def suggest_and_apply_bins(self, features_to_bin=None, apply=False):
        """
        Suggest binning strategies with or without applying them to the dataset.

        Parameters:
        features_to_bin (list, optional): List of feature names to generate bin suggestions for.
                                          If None, uses all features in config.
        """
        selected = features_to_bin if features_to_bin else self.config.keys()

        for feature in selected:
            params = self.config.get(feature, {})
            method = params.get("method")
            feature_type = params.get("type")

            if feature_type == "numerical":
                if method == "equal_width":
                    self._equal_width_binning(feature, params, apply=apply)
                elif method == "quantile":
                    self._quantile_binning(feature, params, apply=apply)
                elif method == "decile":
                    self._decile_binning(feature, params, apply=apply)
                elif method == "optimal":
                    self._optimal_binning(feature, params, apply=apply)
                elif method == "manual_numerical":
                    self._manual_numerical_binning(
                        feature, params, apply=apply)

            elif feature_type == "categorical":
                if method == "manual_categorical":
                    self._manual_categorical_binning(
                        feature, params, apply=apply)
        return self.df

    def apply_bins(self):
        """
        Apply previously suggested binning to the DataFrame.

        Returns:
        pd.DataFrame: Updated DataFrame with new binned feature columns.
        """
        for feature in self.binning_info:
            params = self.config.get(feature, {})
            method = self.binning_info[feature].get("method")

            if method == "equal_width":
                self._equal_width_binning(feature, params, apply=True)
            elif method == "quantile":
                self._quantile_binning(feature, params, apply=True)
            elif method == "decile":
                self._decile_binning(feature, params, apply=True)
            elif method == "optimal":
                self._optimal_binning(feature, params, apply=True)
            elif method == "manual_numerical":
                self._manual_numerical_binning(feature, params, apply=True)
            elif method == "manual_categorical":
                self._manual_categorical_binning(feature, params, apply=True)

        return self.df

    def _equal_width_binning(self, feature, params, apply=True):
        """
        Perform equal width binning on a numerical feature.

        Parameters:
        feature (str): The feature name.
        params (dict): Parameters including number of bins.
        apply : bool, optional (default=True)
            If True, immediately adds a new column with binned values to the DataFrame. 
            If False, stores binning logic only for review or delayed application via `apply_bins()`.
        """
        bins = params.get("bins", 5)
        binned, bin_edges = pd.cut(
            self.df[feature], bins=bins, retbins=True, labels=False)
        self.binning_info[feature] = {
            "method": "equal_width",
            "bin_edges": bin_edges.tolist(),
            "description": f"Equal width binning into {bins} bins",
            "binned_values": binned if apply else None
        }
        if apply:
            self.df[f"{feature}_binned"] = binned

    def _quantile_binning(self, feature, params, apply=True):
        """
        Perform quantile-based binning on a numerical feature.

        Parameters:
        feature (str): The feature name.
        params (dict): Parameters including number of quantiles.
        apply : bool, optional (default=True)
            If True, immediately adds a new column with binned values to the DataFrame. 
            If False, stores binning logic only for review or delayed application via `apply_bins()`.
        """
        q = params.get("quantiles", 4)
        binned, bin_edges = pd.qcut(
            self.df[feature], q=q, retbins=True, labels=False, duplicates='drop')
        self.binning_info[feature] = {
            "method": "quantile",
            "bin_edges": bin_edges.tolist(),
            "description": f"Quantile binning into {q} bins",
            "binned_values": binned if apply else None
        }
        if apply:
            self.df[f"{feature}_binned"] = binned

    def _decile_binning(self, feature, params, apply=True):
        """
        Perform decile-based binning (10 quantiles) on a numerical feature.

        Parameters:
        feature (str): The feature name.
        params (dict): Parameters (not required).
        apply : bool, optional (default=True)
            If True, immediately adds a new column with binned values to the DataFrame. 
            If False, stores binning logic only for review or delayed application via `apply_bins()`.
        """
        binned, bin_edges = pd.qcut(
            self.df[feature], q=10, retbins=True, labels=False, duplicates='drop')
        self.binning_info[feature] = {
            "method": "decile",
            "bin_edges": bin_edges.tolist(),
            "description": "Decile binning (10 quantiles)",
            "binned_values": binned if apply else None
        }
        if apply:
            self.df[f"{feature}_binned"] = binned

    def _optimal_binning(self, feature, params, apply=True):
        """
        Perform optimal binning using the optbinning package.

        Parameters:
        feature (str): The feature name.
        params (dict): Parameters including target variable.
        apply : bool, optional (default=True)
            If True, immediately adds a new column with binned values to the DataFrame. 
            If False, stores binning logic only for review or delayed application via `apply_bins()`.
        """
        target = params["target"]
        optb = OptimalBinning(name=feature, dtype="numerical", solver="cp")
        optb.fit(self.df[feature], self.df[target])
        bin_indexes = optb.transform(self.df[feature], metric="indices")
        self.binning_info[feature] = {
            "method": "optimal",
            "bin_edges": optb.splits.tolist() if optb.splits is not None else [],
            "description": "Optimal binning using target supervision",
            "binned_values": bin_indexes if apply else None
        }
        if apply:
            self.df[f"{feature}_binned"] = bin_indexes

    def _manual_categorical_binning(self, feature, params, apply=True):
        """
        Apply manual binning to a categorical feature based on user-defined mapping.

        Parameters:
        feature (str): The feature name.
        params (dict): Parameters including category-to-bin mapping.
        apply : bool, optional (default=True)
            If True, immediately adds a new column with binned values to the DataFrame. 
            If False, stores binning logic only for review or delayed application via `apply_bins()`.
        """
        mapping = params.get("mapping")
        binned = self.df[feature].map(mapping)
        self.binning_info[feature] = {
            "method": "manual_categorical",
            "mapping": mapping,
            "description": "Manual categorical mapping",
            "binned_values": binned if apply else None
        }
        if apply:
            self.df[f"{feature}_binned"] = binned

    def _manual_numerical_binning(self, feature, params, apply=True):
        """
        Apply manual binning to a numerical feature using defined bin edges.

        Parameters:
        feature (str): The feature name.
        params (dict): Parameters including bin_edges.
        apply : bool, optional (default=True)
        """
        bin_edges = params.get("bin_edges")

        if bin_edges is None:
            raise ValueError(f"No bin_edges specified for feature {feature}.")

        binned = pd.cut(self.df[feature], bins=bin_edges,
                        labels=False, include_lowest=True)

        bin_edges = sorted(bin_edges)

        # Cover full range of values, i.e. less than the first edge and greater than the last edge
        bin_edges = [float('-inf')] + bin_edges + [float('inf')]

        binned = pd.cut(self.df[feature], bins=bin_edges,
                        labels=False, include_lowest=True)

        # Ensure integer labels
        if not pd.api.types.is_integer_dtype(binned):
            binned = binned.astype("Int64")

        self.binning_info[feature] = {
            "method": "manual_numerical",
            "bin_edges": bin_edges,
            "description": f"Manual numerical binning with edges {bin_edges}",
            "binned_values": binned if apply else None
        }
        if apply:
            self.df[f"{feature}_binned"] = binned

    def get_binning_info(self):
        """
        Retrieve stored binning metadata.

        Returns:
        dict: Dictionary of feature names and binning method descriptions.
        """
        return self.binning_info

    def get_bin_edges(self, feature):
        """
        Returns the bin edges for a given feature if available.
        """
        return self.binning_info.get(feature, {}).get("bin_edges")
