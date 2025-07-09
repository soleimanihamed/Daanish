# utils/importance/iv_calculator.py

import numpy as np
import pandas as pd
from typing import List, Dict


class IVCalculator:
    """
    Information Value (IV) Calculator for evaluating predictive power of binned features.

    Supports both:
    - Precomputed WoE values (e.g., using WOEEncoder)
    - Internal WoE computation directly from binned features and target

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing binned features and either WoE-transformed features or raw target.

    target : str
        Name of the binary target variable column (0 = good, 1 = bad).

    Attributes
    ----------
    iv_scores : dict
        Dictionary storing IV value for each feature.
    """

    def __init__(self, df: pd.DataFrame, target: str):
        if isinstance(target, list):
            if len(target) == 1:
                target = target[0]
            else:
                raise ValueError(
                    "Target variable must be a string, not a list.")

        self.df = df.copy()
        self.target = target
        self.iv_scores = {}

    def calculate_iv(
        self,
        binned_features: List[str],
        woe_column_map: Dict[str, str] = None,
        use_precomputed_woe: bool = True
    ):
        """
        Compute IV values for multiple features.

        Parameters
        ----------
        binned_features : List[str]
            List of binned feature names.

        woe_column_map : dict, optional
            A dictionary mapping binned feature names to their corresponding WoE column names.
            Example: {'loan_amnt_binned': 'loan_amnt_woe'}

        use_precomputed_woe : bool, default=True
            If False, the method calculates WoE internally for each feature.
        """
        for feature in binned_features:
            woe_col = woe_column_map.get(feature) if woe_column_map else None
            self._calculate_iv_single(feature, woe_col, use_precomputed_woe)

    def _calculate_iv_single(self, feature: str, woe_col: str = None, use_precomputed_woe: bool = True):
        """
        Compute IV value for a single feature.

        Parameters
        ----------
        feature : str
            The name of the binned feature.

        woe_col : str, optional
            The name of the corresponding WoE column. Only used if `use_precomputed_woe=True`.

        use_precomputed_woe : bool
            Whether to use an existing WoE column or compute WoE internally.
        """
        if use_precomputed_woe:
            if woe_col is None:
                woe_col = f"{feature}_woe"

            if woe_col not in self.df.columns:
                raise ValueError(
                    f"WOE column '{woe_col}' not found in the DataFrame.")

            temp = self.df[[feature, woe_col, self.target]].copy()

        else:
            # Compute WoE from scratch
            grouped = self.df.groupby(
                feature)[self.target].agg(['sum', 'count'])
            grouped.columns = ['bad', 'total']
            grouped['good'] = grouped['total'] - grouped['bad']
            total_good = grouped['good'].sum()
            total_bad = grouped['bad'].sum()
            grouped['dist_good'] = grouped['good'] / total_good
            grouped['dist_bad'] = grouped['bad'] / total_bad
            grouped['WOE'] = np.log(
                (grouped['dist_good'] + 1e-6) / (grouped['dist_bad'] + 1e-6))

            woe_map = grouped['WOE'].to_dict()

            temp = self.df[[feature, self.target]].copy()
            temp['woe_tmp'] = temp[feature].map(woe_map)
            woe_col = 'woe_tmp'

        # Calculate IV components
        grouped_iv = temp.groupby(feature).agg(
            bad_count=(self.target, 'sum'),
            total_count=(self.target, 'count'),
            woe=(woe_col, 'first')
        )
        grouped_iv['good_count'] = grouped_iv['total_count'] - \
            grouped_iv['bad_count']
        total_good = grouped_iv['good_count'].sum()
        total_bad = grouped_iv['bad_count'].sum()
        grouped_iv['dist_good'] = grouped_iv['good_count'] / total_good
        grouped_iv['dist_bad'] = grouped_iv['bad_count'] / total_bad
        grouped_iv['IV'] = (grouped_iv['dist_good'] -
                            grouped_iv['dist_bad']) * grouped_iv['woe']

        self.iv_scores[feature] = grouped_iv['IV'].sum()

    def get_iv_scores(self) -> Dict[str, float]:
        """
        Retrieve calculated IV values.

        Returns
        -------
        dict
            Dictionary of {feature: IV value}
        """
        return self.iv_scores

    def as_dataframe(self) -> pd.DataFrame:
        """
        Get the IV scores as a sorted DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and IV values.
        """
        return pd.DataFrame({
            'Feature': list(self.iv_scores.keys()),
            'IV': list(self.iv_scores.values())
        }).sort_values(by='IV', ascending=False).reset_index(drop=True)
