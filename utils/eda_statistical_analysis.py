from scipy import stats
from dython.nominal import associations


class StatisticalEDAAnalysis:
    def __init__(self, data):
        self.data = data

    def perform_normality_test(self, feature):
        """
        Performs Shapiro-Wilk normality test on a feature.

        Args:
            feature (str): Feature name.

        Returns:
            tuple: Test statistic and p-value.
        """
        stat, p_value = stats.shapiro(self.data[feature])
        print(f"Shapiro-Wilk test statistic: {stat}, p-value: {p_value}")
        return stat, p_value

    def categorical_correlation_analysis(self):
        """
        Performs categorical correlation analysis using dython.
        """
        associations(self.data, nominal_columns='all', figsize=(10, 6))
