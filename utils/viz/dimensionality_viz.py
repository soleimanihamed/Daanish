# uitls/eda/visualisation/dimensionality_viz.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import pandas as pd
from scipy.stats import linregress


class DimensionalityViz:
    """
    A visualization class for plotting outputs from dimensionality reduction techniques,
    including PCA, MCA, and FAMD.

    This class provides reusable and modular visual tools to help interpret the results
    of dimensionality reduction, such as explained inertia, row/column projections,
    and feature/category contributions.

    Supported Techniques:
    - PCA: Principal Component Analysis (numerical data)
    - MCA: Multiple Correspondence Analysis (categorical data)
    - FAMD: Factor Analysis of Mixed Data (mixed data types)

    Typical visualizations include:
    - Cumulative explained inertia plots
    - Row coordinate scatter plots (colored by target)
    - Column coordinate projections
    - Contribution heatmaps for features or categories

    Attributes:
        data (pd.DataFrame): Original dataset for reference (optional).
        column_name_mapping (dict): Optional mapping from original to display-friendly column names.
    """

    def __init__(self, data=None, display_names=None):
        self.data = data
        self.column_name_mapping = display_names or {}

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

    def plot_mca_column_contributions(self, contributions_df, cmap="viridis"):
        """
        Plot a heat-map of MCA category contributions (%) to each dimension.

        Parameters
        ----------
        contributions_df : pd.DataFrame
            Output of `mca_analyzer.get_column_contributions_df()`.
            Rows = categories (e.g., 'home_ownership_RENT'),
            Columns = MCA dimensions (e.g., 'Dim 1', 'Dim 2', …).
        cmap : str, optional
            Matplotlib / seaborn colormap. Default is "viridis".
        """
        if not isinstance(contributions_df, pd.DataFrame):
            raise ValueError("contributions_df must be a pandas DataFrame.")

        # Order categories by their highest contribution for readability (optional)
        ordered = contributions_df.copy()
        ordered["max_contrib"] = ordered.max(axis=1)
        ordered = ordered.sort_values(
            "max_contrib", ascending=False).drop(columns="max_contrib")

        plt.figure(figsize=(12, min(0.5 * len(ordered), 10)))
        sns.heatmap(
            ordered,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            cbar_kws={"label": "Contribution (%)"},
        )
        plt.title("MCA Category Contributions to Dimensions")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_famd_explained_inertia(inertia):
        """
        Plot cumulative explained inertia (variance) of the FAMD dimensions.
        """
        if inertia is None or len(inertia) == 0:
            print("Inertia not provided or is empty.")
            return

        if not isinstance(inertia, pd.Series):
            inertia = pd.Series(inertia)
        inertia.index = [f"Dim {i+1}" for i in range(len(inertia))]

        cumulative_inertia = inertia.cumsum()

        plt.figure(figsize=(10, 6))
        plt.bar(inertia.index, inertia.values,
                label='Individual Explained Inertia', alpha=0.6)
        plt.plot(inertia.index, cumulative_inertia.values, marker='o',
                 color='black', label='Cumulative Explained Inertia')

        for i, val in enumerate(cumulative_inertia.values):
            plt.text(i, val + 0.01, f"{val:.2f}", ha='center', fontsize=9)

        plt.xlabel('FAMD Dimensions')
        plt.ylabel('Proportion of Explained Inertia')
        plt.title('Explained Inertia by FAMD Dimensions')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_famd_row_coordinates(self, row_coords_df: pd.DataFrame, target_column: str, dim_x: str = 'Dim1', dim_y: str = 'Dim2'):
        """
        Plot FAMD row coordinates colored by the target variable.

        Parameters:
        row_coords_df (pd.DataFrame): DataFrame with row coordinates and the target column.
        target_column (str): Name of the column with target labels.
        dim_x (str): Column name for x-axis.
        dim_y (str): Column name for y-axis.
        """
        if dim_x not in row_coords_df.columns or dim_y not in row_coords_df.columns:
            raise ValueError(
                f"Columns {dim_x} or {dim_y} not found in row_coords_df.")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=row_coords_df, x=dim_x, y=dim_y,
                        hue=target_column, palette='Set1', alpha=0.7)
        plt.title(
            f"{dim_x} vs {dim_y} (FAMD Row Coordinates) Colored by {target_column}")
        plt.xlabel(dim_x)
        plt.ylabel(dim_y)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_famd_column_coordinates(column_coords: pd.DataFrame, dim_x: int = 0, dim_y: int = 1):
        """
        Plot FAMD column coordinates (for numeric + categorical features).

        Parameters:
        column_coords (pd.DataFrame): DataFrame of FAMD column coordinates.
        dim_x (int): Dimension index for x-axis.
        dim_y (int): Dimension index for y-axis.
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
            f"FAMD: Column Coordinates (Dimension {dim_x + 1} vs Dimension {dim_y + 1})")
        plt.xlabel(f"Dimension {dim_x + 1}")
        plt.ylabel(f"Dimension {dim_y + 1}")
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_famd_column_contributions(self, contributions_df: pd.DataFrame, cmap="coolwarm"):
        """
        Plot heatmap of FAMD feature contributions to each dimension.

        Parameters:
        contributions_df : pd.DataFrame
            Feature contributions (rows = features, columns = FAMD dimensions).
        cmap : str
            Colormap to use.
        """
        if not isinstance(contributions_df, pd.DataFrame):
            raise ValueError("contributions_df must be a pandas DataFrame.")

        ordered = contributions_df.copy()
        ordered["max_contrib"] = ordered.max(axis=1)
        ordered = ordered.sort_values(
            "max_contrib", ascending=False).drop(columns="max_contrib")

        plt.figure(figsize=(12, min(0.5 * len(ordered), 10)))
        sns.heatmap(
            ordered,
            annot=True,
            fmt=".1f",
            cmap=cmap,
            cbar_kws={"label": "Contribution (%)"},
        )
        plt.title("FAMD Feature Contributions to Dimensions")
        plt.tight_layout()
        plt.show()
