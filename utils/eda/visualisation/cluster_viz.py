# daanish/eda/visualisation/cluster_viz.py

import matplotlib.pyplot as plt
import seaborn as sns


class ClusterVisualisation:
    """
    A utility class for visualising clustering results in 2D space.

    This class provides static methods for plotting cluster assignments 
    after dimensionality reduction (e.g., using PCA or t-SNE).
    """

    @staticmethod
    def plot_clusters_2d(pca_components, labels, centroids=None, title='Clusters in 2D'):
        """
        Plots clustered data points in 2D space using PCA-reduced components.

        Parameters:
            pca_components (np.ndarray): A 2D array of shape (n_samples, 2) 
                representing the projected data points.
            labels (array-like): Cluster labels corresponding to each data point.
            centroids (np.ndarray, optional): A 2D array of shape (n_clusters, 2) 
                representing cluster centroids in the same reduced space.
            title (str, optional): Title for the plot. Default is 'Clusters in 2D'.

        Returns:
            None: Displays a matplotlib scatter plot showing the clusters and (optionally) centroids.
        """

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=pca_components[:, 0], y=pca_components[:, 1],
            hue=labels, palette='Set2', s=60
        )
        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        c='black', marker='X', s=100, label='Centroids')
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
