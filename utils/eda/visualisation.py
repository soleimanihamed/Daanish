# uitls/eda/Visualisation Class

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pandas as pd
from scipy.stats import linregress


class Visualisation:
    def __init__(self, data, display_names):
        """
        Initialise the Visualisation class.

        Parameters:
            data (pd.DataFrame): The dataset to visualise.
        """
        self.data = data
        # Use provided mapping or empty dict
        self.column_name_mapping = display_names

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

            # Use display names for plot titles and labels
            # Get display name or original name if not found
            display_name = self.column_name_mapping.get(variable, variable)

            # Customize plot
            plt.xlabel(display_name)
            plt.ylabel("Density")
            plt.title(f"Distribution of {display_name}")
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

                # Use display name for labels
                x_label = "Count" if orientation == "horizontal" else self.column_name_mapping.get(
                    variable, variable)
                y_label = self.column_name_mapping.get(
                    variable, variable) if orientation == "horizontal" else "Count"

            else:  # Categorical Data
                sns.countplot(data=data_series.to_frame(), x=variable if orientation == "vertical" else None,
                              y=variable if orientation == "horizontal" else None,
                              color="blue", alpha=0.6)

                # Use display name for labels
                x_label = "Count" if orientation == "horizontal" else self.column_name_mapping.get(
                    variable, variable)
                y_label = self.column_name_mapping.get(
                    variable, variable) if orientation == "horizontal" else "Count"

            # Customize plot
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(
                f"Histogram of {self.column_name_mapping.get(variable, variable)}")
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
        x_label = self.column_name_mapping.get(x_var, x_var)
        y_label = self.column_name_mapping.get(y_var, y_var)
        hue_label = self.column_name_mapping.get(
            hue_var, hue_var) if hue_var else None
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"Scatter Plot of {y_label} vs {x_label}")
        if hue_label:
            plt.legend(title=hue_label)  # Set legend title if hue is used
        else:
            plt.legend()
        plt.show()

    def plot_boxplot(self, column, by=None):
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

        column_display = self.column_name_mapping.get(
            column, column)  # get display name for column

        if by:
            by_display = self.column_name_mapping.get(
                by, by)  # get display name for by
            self.data.boxplot(column=[column], by=by)
            plt.xlabel(by_display)
            plt.ylabel(column_display)
            plt.title(f"Box Plot of {column_display} by {by_display}")

        else:
            self.data.boxplot(column=[column])
            plt.ylabel(column_display)
            plt.title(f"Box Plot of {column_display}")

        plt.show()

    def plot_explained_variance(self, explained_variance):
        """
        Plot cumulative explained variance by the number of principal components.

        Parameters:
            explained_variance (array-like): The explained variance ratio from PCAAnalyzer.
        """
        plt.figure(figsize=(8, 5))
        cumulative_variance = np.cumsum(explained_variance)
        plt.plot(range(1, len(cumulative_variance) + 1),
                 cumulative_variance, marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_pca_loadings(self, loadings):
        """
        Plot a heatmap of PCA loadings showing feature contributions to each component.

        Parameters:
            loadings (pd.DataFrame): Loadings matrix from PCAAnalyzer.
        """
        plt.figure(figsize=(12, min(0.5 * len(loadings), 10)))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
        plt.title("PCA Component Loadings")
        plt.tight_layout()
        plt.show()

    def plot_pca_scores(self, scores_df, target_column):
        """
        Scatter plot of PCA scores colored by the target variable.

        Parameters:
            scores_df (pd.DataFrame): DataFrame with PCA scores and target column.
            target_column (str): The name of the column to color the plot by.
        """
        if target_column not in scores_df.columns:
            raise ValueError(
                f"{target_column} not found in PCA scores dataframe.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=scores_df, x='PC1', y='PC2',
                        hue=target_column, palette='Set1', alpha=0.7)
        plt.title("PC1 vs PC2 Colored by Target")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap_matrix(matrix: pd.DataFrame, title="Matrix Heatmap", cmap="coolwarm", annot=True):
        """
        Plots a heatmap for any given square matrix.

        Args:
            matrix (pd.DataFrame): A square matrix (e.g., correlation matrix).
            title (str): Title of the plot.
            cmap (str): Color map to use.
            annot (bool): Whether to annotate the heatmap with values.
        """
        if matrix.empty:
            print("Empty matrix provided for plotting.")
            return

        # Dynamically adjust figure size based on number of features
        n_features = matrix.shape[0]
        size = max(8, min(2 + n_features * 0.5, 20))
        figsize = (size, size)

        plt.figure(figsize=figsize)
        sns.heatmap(matrix,
                    annot=annot,
                    fmt=".2f",
                    cmap=cmap,
                    center=0,
                    square=True,
                    cbar_kws={"shrink": .8},
                    linewidths=0.5,
                    linecolor='gray')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mca_explained_inertia(inertia):
        """
        Plot cumulative explained inertia (variance) of the MCA by number of dimensions.
        """

        # Added a check for empty inertia
        if inertia is None or len(inertia) == 0:
            print("Inertia not provided or is empty.")
            return

        if not isinstance(inertia, pd.Series):
            inertia = pd.Series(inertia)
        inertia.index = [f"Dim {i+1}" for i in range(len(inertia))]

        cumulative_inertia = inertia.cumsum()

        plt.figure(figsize=(10, 6))

        # Bar plot for individual explained inertia
        plt.bar(inertia.index, inertia.values,
                label='Individual Explained Inertia', alpha=0.6)

        # Line plot for cumulative explained inertia
        plt.plot(inertia.index, cumulative_inertia.values, marker='o',
                 color='black', label='Cumulative Explained Inertia')

        for i, val in enumerate(cumulative_inertia.values):
            plt.text(i, val + 0.01, f"{val:.2f}", ha='center', fontsize=9)

        plt.xlabel('MCA Dimensions')
        plt.ylabel('Proportion of Explained Inertia')
        plt.title('Explained Inertia by MCA Dimensions')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # @staticmethod
    # def plot_mca_column_coordinates(column_coords):
    #     """
    #     Plot the coordinates of variables on the MCA dimensions.
    #     """

    #     if column_coords is None:
    #         raise ValueError("Column coordinates not provided.")

    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(
    #         x=column_coords[0],
    #         y=column_coords[1],
    #         s=100
    #     )
    #     for i, txt in enumerate(column_coords.index):
    #         plt.text(
    #             column_coords.iloc[i, 0] + 0.01,
    #             column_coords.iloc[i, 1] + 0.01,
    #             txt, fontsize=9
    #         )

    #     plt.axhline(0, color='grey', linestyle='--')
    #     plt.axvline(0, color='grey', linestyle='--')
    #     plt.title("MCA: Column Coordinates on First Two Dimensions")
    #     plt.xlabel("Dimension 1")
    #     plt.ylabel("Dimension 2")
    #     plt.grid(True)
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.show()

    # def plot_mca_row_coordinates(self, row_coords_df: pd.DataFrame, target_column: str, dim_x: str = 'Dim1', dim_y: str = 'Dim2'):
    #     """
    #     Plot MCA row coordinates colored by the target variable.

    #     Parameters:
    #     row_coords_df (pd.DataFrame): DataFrame containing the row coordinates and the target column.
    #     target_column (str): Name of the column containing target labels.
    #     dim_x (str): Column name for x-axis (default: 'Dim1').
    #     dim_y (str): Column name for y-axis (default: 'Dim2').
    #     """
    #     plt.figure(figsize=(8, 6))
    #     sns.scatterplot(data=row_coords_df, x=dim_x, y=dim_y,
    #                     hue=target_column, palette='Set2', alpha=0.7)
    #     plt.title(
    #         f"{dim_x} vs {dim_y} (MCA Row Coordinates) Colored by {target_column}")
    #     plt.xlabel(dim_x)
    #     plt.ylabel(dim_y)
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()

    def plot_mca_row_coordinates(self, row_coords_df: pd.DataFrame, target_column: str, dim_x: str = 'Dim1', dim_y: str = 'Dim2'):
        """
        Plot MCA row coordinates colored by the target variable.

        Parameters:
        row_coords_df (pd.DataFrame): DataFrame containing the row coordinates and the target column.
        target_column (str): Name of the column containing target labels.
        dim_x (str): Column name for x-axis (e.g., 'Dim1').
        dim_y (str): Column name for y-axis (e.g., 'Dim2').
        """
        if dim_x not in row_coords_df.columns or dim_y not in row_coords_df.columns:
            raise ValueError(
                f"Columns {dim_x} or {dim_y} not found in row_coords_df. Available: {list(row_coords_df.columns)}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=row_coords_df, x=dim_x, y=dim_y,
                        hue=target_column, palette='Set2', alpha=0.7)
        plt.title(
            f"{dim_x} vs {dim_y} (MCA Row Coordinates) Colored by {target_column}")
        plt.xlabel(dim_x)
        plt.ylabel(dim_y)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mca_column_coordinates(column_coords: pd.DataFrame, dim_x: int = 0, dim_y: int = 1):
        """
        Plot the coordinates of variables on specified MCA dimensions.

        Parameters:
        column_coords (pd.DataFrame): DataFrame of MCA column coordinates.
        dim_x (int): Index of the dimension for x-axis (e.g., 0 for Dim1).
        dim_y (int): Index of the dimension for y-axis (e.g., 1 for Dim2).
        """
        if column_coords is None or column_coords.shape[1] <= max(dim_x, dim_y):
            raise ValueError(
                "Invalid dimensions or column coordinates not available.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=column_coords.iloc[:, dim_x],
            y=column_coords.iloc[:, dim_y],
            s=100
        )

        for i, txt in enumerate(column_coords.index):
            plt.text(
                column_coords.iloc[i, dim_x] + 0.01,
                column_coords.iloc[i, dim_y] + 0.01,
                txt, fontsize=9
            )

        plt.axhline(0, color='grey', linestyle='--')
        plt.axvline(0, color='grey', linestyle='--')
        plt.title(
            f"MCA: Column Coordinates (Dimension {dim_x + 1} vs Dimension {dim_y + 1})")
        plt.xlabel(f"Dimension {dim_x + 1}")
        plt.ylabel(f"Dimension {dim_y + 1}")
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
