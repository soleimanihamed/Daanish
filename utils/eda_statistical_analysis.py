# uitls/eda_statistical_analysis.py
from scipy import stats
from dython.nominal import associations
import pandas as pd


class StatisticalEDAAnalysis:
    def __init__(self, data):
        self.data = data

    def perform_normality_test_Shapiro_Wilk(self, features, alpha=0.05):
        """
        Performs Shapiro-Wilk normality test on multiple features and returns statistical attributes.

        Args:
            features (list): List of feature names.
            alpha (float): Significance level for the Shapiro-Wilk test.

        Returns:
            dict: Dictionary containing statistical attributes and normality test results for each feature.
        """
        results = {}

        for feature in features:
            if feature not in self.data.columns:
                print(
                    f"Warning: Feature '{feature}' not found in the DataFrame.")
                continue

            if pd.api.types.is_numeric_dtype(self.data[feature]):
                stat, p_value = stats.shapiro(self.data[feature].dropna())

                results[feature] = {
                    "mean": self.data[feature].mean(),
                    "std": self.data[feature].std(),
                    "min": self.data[feature].min(),
                    "max": self.data[feature].max(),
                    "median": self.data[feature].median(),
                    "q1": self.data[feature].quantile(0.25),
                    "q3": self.data[feature].quantile(0.75),
                    "shapiro_stat": stat,
                    "shapiro_p_value": p_value,
                    "is_normal": p_value > alpha,
                }
            else:
                print(
                    f"Skipping normality test for non-numerical feature: {feature}")

        return results

    def categorical_correlation_analysis(self):
        """
        Performs categorical correlation analysis using dython.
        """
        associations(self.data, nominal_columns='all', figsize=(10, 6))

    def crosstab(self, var1, var2, normalize=None, margins=True, percent=True):
        """
        Generates a cross-tabulation (contingency table) of two categorical variables.

        Parameters:
            var1 (str): The first variable (rows).
            var2 (str): The second variable (columns).
            normalize (str, optional): 'index', 'columns', or None (default: None).
            margins (bool): Whether to add row/column totals.
            percent (bool): Format output as percentages.

        Returns:
            pd.DataFrame: The crosstabulated table.
        """
        if var1 not in self.data.columns or var2 not in self.data.columns:
            raise ValueError(
                f"One or both variables: {var1}, {var2} not found in dataset.")

        table = pd.crosstab(
            self.data[var1], self.data[var2], normalize=normalize, margins=margins)

        if percent:
            return table.style.format("{:.0%}")  # Convert to percentage format

        return table

    def crosstab_three_way(self, var1, var2, var3, normalize="columns", percent=True):
        """
        Generates a three-way cross-tabulation table.

        Parameters:
            var1 (str): The first variable (rows).
            var2 (str): The second variable (grouped columns).
            var3 (str): The third variable (sub-columns within var2).
            normalize (str, optional): 'index', 'columns', or None (default: 'columns').
            percent (bool): Format output as percentages.

        Returns:
            pd.DataFrame: The three-way crosstabulated table.
        """
        if var1 not in self.data.columns or var2 not in self.data.columns or var3 not in self.data.columns:
            raise ValueError(
                f"One or more variables: {var1}, {var2}, {var3} not found in dataset.")

        table = pd.crosstab(
            self.data[var1], [self.data[var2], self.data[var3]], normalize=normalize)

        if percent:
            return table.round(4).style.format("{:.0%}")

        return table
