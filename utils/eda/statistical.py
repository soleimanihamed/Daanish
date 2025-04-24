# uitls/eda/statistical.py

from scipy import stats
from dython.nominal import associations
import pandas as pd
from fitter import Fitter


class StatisticalAnalysis:
    """
    StatisticalAnalysis Class

    This class provides a collection of statistical methods designed to support exploratory data analysis (EDA). 
    It includes tools for:

    - Performing normality tests (e.g., Shapiro-Wilk)
    - Fitting the best probability distribution to numerical features
    - Computing correlation between variables
    - Creating two-way and three-way cross-tabulations for categorical variables

    These methods help uncover underlying data patterns and relationships for both numerical and categorical features.
    """

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

    def fit_best_distribution(self, numeric_variables: list, method='sumsquare_error', common_distributions=True, timeout=60):
        """
        Fits the best probability distribution for each numeric variable in the input list.

        Parameters:
            numeric_variables (list): List of column names to analyze.
            method (str): Method to find the best fit ('sumsquare_error', 'aic', 'bic').
            common_distributions (bool): If True, uses only common distributions to avoid long runtime.
            timeout (int): Timeout for fitting each distribution (in seconds).

        Returns:
            dict: A dictionary containing the best-fitting distribution and its parameters for each variable.
        """

        # Dictionary to store results
        results = {}

        # Common distributions (to avoid very slow ones)
        common_dists = [
            'norm', 'expon', 'lognorm', 'gamma', 'beta', 'weibull_min', 'chi2', 'pareto',
            'uniform', 't', 'gumbel_r', 'burr', 'invgauss', 'triang', 'laplace', 'logistic',
            'genextreme', 'skewnorm', 'genpareto', 'burr12', 'fatiguelife', 'geninvgauss',
            'halfnorm', 'exponpow'
        ]

        # Iterate over the input variables
        for variable in numeric_variables:
            # Check if the variable is numeric
            if pd.api.types.is_numeric_dtype(self.data[variable]):
                print(f"\n=== Analyzing {variable} ===")

                # Drop missing values
                data = self.data[variable].dropna()

                # Select distributions based on user choice
                distributions = common_dists if common_distributions else None

                # Initialize Fitter
                fitter = Fitter(
                    data, distributions=distributions, timeout=timeout)

                try:
                    # Fit the distributions
                    fitter.fit()

                    # Get the best-fitting distribution
                    # You can also use 'aic' or 'bic'
                    best_fit = fitter.get_best(method)

                # Check if a best fit was found
                    if best_fit:
                        # Store results
                        results[variable] = {
                            # Get the name of the best-fitting distribution
                            "best_distribution": list(best_fit.keys())[0],
                            # Get the parameters of the best-fitting distribution
                            "parameters": list(best_fit.values())[0]
                        }

                        # Print results
                        print(
                            f"Best-fitting distribution: {list(best_fit.keys())[0]}")
                        print(f"Parameters: {list(best_fit.values())[0]}")
                    else:
                        results[variable] = {
                            "best_distribution": None,
                            "parameters": None
                        }
                        print(
                            f"No suitable distribution found for {variable}.")
                except Exception as e:
                    print(f"Error fitting distributions for {variable}: {e}")
                    results[variable] = {
                        "best_distribution": None, "parameters": None}

            else:
                print(f"\n=== Skipping {variable} (not numeric) ===")

        # Convert results dictionary to a well-structured DataFrame
        if results:
            results_df = pd.DataFrame([
                {"Feature": feature, "Best Distribution":
                    data["best_distribution"], "Parameters": data["parameters"]}
                for feature, data in results.items()
            ])

            # Print formatted results
            print("\n=== Distribution Fitting Results ===")
            # Removes default index for clarity
            print(results_df.to_string(index=False))
        else:
            print("No valid distributions were found for the provided numeric variables.")

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
