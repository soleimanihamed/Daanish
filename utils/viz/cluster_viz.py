# daanish/eda/visualisation/cluster_viz.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ClusterVisualisation:
    """
    A utility class for visualising clustering results in 2D space.

    This class provides static methods for plotting cluster assignments
    after dimensionality reduction (e.g., using PCA or t-SNE).
    """

    @staticmethod
    def plot_clusters_2d(components, labels, centroids=None, title='Clusters in 2D', xlabel='Component 1', ylabel='Component 2'):
        """
        Plots clustered data points in 2D space using any 2D projection (e.g., PCA, MCA, t-SNE).

        Parameters:
            components (np.ndarray or pd.DataFrame): A 2D array (n_samples, 2) representing projected points.
            labels (array-like): Cluster labels.
            centroids (np.ndarray or pd.DataFrame, optional): Optional projected centroids.
            title (str): Plot title.
            xlabel (str): Label for x-axis.
            ylabel (str): Label for y-axis.
        """

        plt.figure(figsize=(8, 6))

        if isinstance(components, pd.DataFrame):
            x_vals = components.iloc[:, 0]
            y_vals = components.iloc[:, 1]
        else:
            x_vals = components[:, 0]
            y_vals = components[:, 1]

        sns.scatterplot(x=x_vals, y=y_vals, hue=labels, palette='Set2', s=60)

        if centroids is not None:
            if isinstance(centroids, pd.DataFrame):
                x_centroids = centroids.iloc[:, 0]
                y_centroids = centroids.iloc[:, 1]
            else:
                x_centroids = centroids[:, 0]
                y_centroids = centroids[:, 1]

            plt.scatter(
                x_centroids, y_centroids,
                c='black', marker='X', s=100, label='Centroids'
            )

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
