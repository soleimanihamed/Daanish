# utils/eda/multicollinearity.py

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


class MulticollinearityDetector:
    """
    Detects multicollinearity among numerical features using Variance Inflation Factor (VIF)
    and pairwise correlation metrics.
    """

    def __init__(self, data, correlation_analyzer=None, correlation_matrix=None, correlation_method="pearson"):
        """
        Initializes the MulticollinearityDetector.

        Args:
            data (pd.DataFrame): The input dataset (numerical columns will be selected).
            correlation_analyzer (CorrelationAnalyzer, optional): An optional CorrelationAnalyzer instance.
            correlation_matrix (pd.DataFrame, optional): Precomputed correlation matrix.
            correlation_method (str): Method to compute correlation if not provided ("pearson", "spearman", etc.).
        """
        self.data = data.select_dtypes(
            include='number')  # Only numerical features
        self.columns = self.data.columns
        self.correlation_method = correlation_method

        if correlation_matrix is not None:
            self.corr_matrix = correlation_matrix
        elif correlation_analyzer is not None:
            self.corr_matrix = correlation_analyzer._calculate_numerical_numerical_correlation(
                self.columns, method=self.correlation_method
            )
        else:
            self.corr_matrix = self.data.corr(method=self.correlation_method)

    def compute_vif(self):
        """
        Calculates the Variance Inflation Factor (VIF) for each numerical feature.

        Returns:
            pd.DataFrame: A DataFrame with 'feature' and 'VIF' columns.
        """
        X = add_constant(self.data)
        vif_data = pd.DataFrame()
        vif_data["feature"] = self.columns
        vif_data["VIF"] = [variance_inflation_factor(
            X.values, i + 1) for i in range(len(self.columns))]  # +1 to skip const
        return vif_data

    def high_correlation_pairs(self, threshold=0.9):
        """
        Identifies pairs of features with absolute correlation greater than the specified threshold.

        Args:
            threshold (float): Correlation threshold to identify strong relationships.

        Returns:
            List[Tuple[str, str, float]]: Sorted list of (feature1, feature2, correlation) tuples.
        """
        high_corrs = []
        for i in range(len(self.columns)):
            for j in range(i + 1, len(self.columns)):
                var1, var2 = self.columns[i], self.columns[j]
                corr = self.corr_matrix.loc[var1, var2]
                if abs(corr) > threshold:
                    high_corrs.append((var1, var2, corr))
        return sorted(high_corrs, key=lambda x: -abs(x[2]))

    def suggest_features_to_drop(self, corr_threshold=0.9, vif_threshold=10.0):
        """
        Suggests features to drop based on:
        - High pairwise correlation.
        - High VIF values (indicating multicollinearity).

        Args:
            corr_threshold (float): Threshold for considering correlation as high.
            vif_threshold (float): Threshold above which VIF is considered problematic.

        Returns:
            dict: {
                "high_corr_drops": list of features due to correlation,
                "high_vif_drops": list of features due to VIF,
                "both": features flagged by both methods
            }
        """
        vif_df = self.compute_vif()
        high_vif_features = vif_df[vif_df["VIF"]
                                   > vif_threshold]["feature"].tolist()

        high_corr_pairs = self.high_correlation_pairs(threshold=corr_threshold)
        high_corr_drops = set()

        for var1, var2, _ in high_corr_pairs:
            vif1 = vif_df.loc[vif_df["feature"] == var1, "VIF"].values[0]
            vif2 = vif_df.loc[vif_df["feature"] == var2, "VIF"].values[0]
            drop = var1 if vif1 > vif2 else var2
            high_corr_drops.add(drop)

        both = list(set(high_corr_drops) & set(high_vif_features))

        return {
            "high_corr_drops": list(high_corr_drops),
            "high_vif_drops": high_vif_features,
            "both": both
        }
