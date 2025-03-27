# Visualisation Class
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pandas as pd
from scipy.stats import linregress


class Visualisation:
    def __init__(self, data):
        """
        Initialise the Visualisation class.

        Parameters:
            data (pd.DataFrame): The dataset to visualise.
        """
        self.data = data

    def plot_distributions(self, fitted_distributions, variables=None, bins="auto"):
        """
        Plot histograms and fitted probability distributions for selected variables.

        Parameters:
            fitted_distributions (dict): {variable_name: {"best_distribution": str, "parameters": dict}}
            variables (list, optional): List of variables to visualize.
            bins (str or int): Number of bins ('auto' for automatic selection).
            show_plot (bool): Whether to display the plot.
        """

        if variables is None:
            # Default to all fitted variables
            variables = list(fitted_distributions.keys())

        # Map string names to actual scipy.stats distributions
        distribution_mapping = {dist: getattr(
            st, dist, None) for dist in st.__all__}

        for variable in variables:
            if variable not in self.data.columns:
                print(f"Warning: {variable} is not in the dataset. Skipping.")
                continue

            data_series = self.data[variable].dropna()
            if variable not in fitted_distributions:
                print(
                    f"Warning: No fitted distribution found for {variable}. Skipping.")
                continue

            best_fit_info = fitted_distributions[variable]
            best_dist_name = best_fit_info["best_distribution"]
            params_dict = best_fit_info["parameters"]

            if best_dist_name not in distribution_mapping or distribution_mapping[best_dist_name] is None:
                print(
                    f"Warning: {best_dist_name} is not a valid scipy distribution. Skipping {variable}.")
                continue

            # Get scipy.stats distribution
            best_distribution = distribution_mapping[best_dist_name]

            # Convert dictionary to tuple format expected by scipy.stats
            params = tuple(params_dict.values())

            # Plot histogram
            plt.figure(figsize=(8, 5))
            sns.histplot(data_series, bins=bins, kde=False, stat="density",
                         alpha=0.6, color="skyblue", label="Data Histogram")

            # Create x values for PDF
            x = np.linspace(data_series.min(), data_series.max(), 1000)

            # Extract parameters correctly
            arg = params[:best_distribution.numargs] if best_distribution.numargs else ()
            loc = params[-2] if len(params) > 1 else 0  # Avoid index errors
            scale = params[-1] if len(params) > 1 else 1  # Avoid index errors

            # Compute and plot PDF
            pdf = best_distribution.pdf(x, *arg, loc=loc, scale=scale)
            plt.plot(
                x, pdf, label=f"{best_dist_name} distribution", linewidth=2, color="red")

            # Customize plot
            plt.xlabel(variable)
            plt.ylabel("Density")
            plt.title(f"Distribution of {variable}")
            plt.legend()
            plt.show()

    def plot_histogram(self, variables=None, bins="auto", orientation="vertical"):
        """
        Plot histograms for one or multiple variables (numerical, nominal, or ordinal).

        Parameters:
            variables (list, optional): List of variables to visualize. If None, defaults to all dataset columns.
            bins (int or str): Number of bins for numerical data ('auto' for automatic).
            orientation (str): 'vertical' (default) or 'horizontal' orientation.
        """
        if variables is None:
            # Default to all columns in dataset
            variables = list(self.data.columns)

        for variable in variables:
            if variable not in self.data.columns:
                print(f"Warning: {variable} not found in dataset. Skipping.")
                continue

            plt.figure(figsize=(8, 5))

            # Drop NaN values
            data_series = self.data[variable].dropna()

            if data_series.dtype in ["int64", "float64"]:  # Numerical Data
                sns.histplot(data=data_series,
                             bins=bins, kde=False, stat="count", color="blue", alpha=0.6,
                             orientation="vertical" if orientation == "vertical" else "horizontal")

            else:  # Categorical Data
                sns.countplot(data=data_series.to_frame(), x=variable if orientation == "vertical" else None,
                              y=variable if orientation == "horizontal" else None,
                              color="blue", alpha=0.6)

            # Customize plot
            plt.xlabel("Count" if orientation == "horizontal" else variable)
            plt.ylabel(variable if orientation == "horizontal" else "Count")
            plt.title(f"Histogram of {variable}")
            plt.show()

    def plot_scatter(self, x_var, y_var, hue_var=None, trendline=True):
        """
        Plot a scatter plot with an optional trend line.

        Parameters:
            x_var (str): Name of the variable for the X-axis.
            y_var (str): Name of the variable for the Y-axis.
            hue_var (str, optional): Categorical variable for coloring points.
            trendline (bool): Whether to add a linear regression trend line.
        """
        if x_var not in self.data.columns or y_var not in self.data.columns:
            print(
                f"Warning: {x_var} or {y_var} not found in dataset. Skipping.")
            return

        plt.figure(figsize=(8, 5))

        # Create scatter plot
        sns.scatterplot(data=self.data, x=x_var, y=y_var,
                        hue=hue_var, alpha=0.6, palette="viridis")

        # Add trend line using linear regression
        if trendline:
            x = self.data[x_var].dropna()
            y = self.data[y_var].dropna()

            if len(x) == len(y):  # Ensure equal length
                slope, intercept, r_value, _, _ = linregress(x, y)
                r_squared = r_value ** 2  # Compute R²
                trend_x = np.linspace(x.min(), x.max(), 100)
                trend_y = slope * trend_x + intercept
                plt.plot(trend_x, trend_y, color="red",
                         linestyle="--", linewidth=2.5, label=f"Trend Line (R²={r_squared:.3f})")

        # Customize plot
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title(f"Scatter Plot of {y_var} vs {x_var}")
        plt.legend()
        plt.show()

    def plot_boxplot(self, column, by=None, title='Box Plot'):
        """
        Creates a box plot of a specified column, optionally grouped by another column.

        Parameters:
            column (str): The column to create the box plot for.
            by (str, optional): The column to group the box plots by.
            title (str): The title of the plot.
        """
        if column not in self.data.columns:
            print(f"Warning: Column '{column}' not found in dataset.")
            return

        if by and by not in self.data.columns:
            print(f"Warning: Column '{by}' not found in dataset.")
            return

        plt.figure(figsize=(8, 6))
        if by:
            self.data.boxplot(column=[column], by=by)
        else:
            self.data.boxplot(column=[column])

        plt.title(title)
        plt.show()
