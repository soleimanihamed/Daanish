# daanish/clustering/kmeans_cluster.py

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from utils.preprocessing.scaler import Scaler


class KMeansClustering:
    """
    A class to perform K-Means clustering on selected numerical features
    with optional scaling and PCA projection.

    Attributes:
        df (pd.DataFrame): The full cleaned dataset.
        features (List[str]): List of column names to use for clustering.
        n_clusters (int): Number of clusters to form.
        scaling_method (str): Method for scaling ('zscore' or 'minmax').
        scale (bool): Whether to apply scaling.
        handle_skew (bool): Whether to apply skewness correction.
        skew_method (str): Skewness reduction method ('log' or 'sqrt').
        skew_threshold (float): Threshold for identifying skewed features.
        scaler (Scaler): Instance of the Scaler class used for preprocessing.
        model (KMeans): Fitted scikit-learn KMeans model.
        labels_ (np.ndarray): Cluster labels assigned to each sample.
        centroids_ (np.ndarray): Coordinates of cluster centers in scaled space.
        pca_ (PCA): Fitted PCA instance used for projecting to 2D.
        scaling_applied (bool): Tracks whether scaling was applied.
    """

    def __init__(self, df: pd.DataFrame, features: list, n_clusters=3,
                 scaling_method='zscore', handle_skew=False, skew_method='log',
                 skew_threshold=1.0, random_state=42, scale=True):

        self.df = df
        self.features = features
        self.n_clusters = n_clusters
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold
        self.scaler = Scaler(method=scaling_method,
                             handle_skew=handle_skew,
                             skew_method=skew_method,
                             skew_threshold=skew_threshold)
        self.model = KMeans(n_clusters=self.n_clusters,
                            random_state=random_state)
        self.labels_ = None
        self.centroids_ = None
        self.pca_ = None
        self.scale = scale
        self.scaling_applied = False
        self._validate_features()

    def fit(self):
        """Fit the K-Means model to scaled data and store labels, centroids, and PCA model."""

        X = self.df[self.features]

        if self.scale:
            X_scaled = self.scaler.fit_transform(X)
            self.scaling_applied = True
        else:
            X_scaled = X

        self.model.fit(X_scaled)
        self.labels_ = self.model.labels_
        self.centroids_ = self.model.cluster_centers_
        self.pca_ = PCA(n_components=2).fit(X_scaled)
        return self.labels_

    def transform(self):
        """
        Predict cluster labels for the data using the trained model.

        Returns:
            np.ndarray: Cluster labels for each observation.
        """

        X = self.df[self.features]
        X_scaled = self.scaler.transform(X) if self.scale else X
        return self.model.predict(X_scaled)

    def fit_predict(self):
        """
        Fit the K-Means model and return the predicted cluster labels.

        Returns:
            np.ndarray: Cluster labels assigned by the model.
        """
        self.fit()
        return self.labels_

    def project_pca(self):
        """
        Project the scaled data into 2D PCA space for visualization.

        Returns:
            np.ndarray: 2D PCA coordinates of each observation.
        """
        X = self.df[self.features]
        X_scaled = self.scaler.transform(X) if self.scale else X
        return self.pca_.transform(X_scaled)

    def get_centroids(self):
        """
        Return the cluster centroids in the scaled feature space.

        Returns:
            np.ndarray: Coordinates of cluster centroids.
        """
        return self.centroids_

    def get_original_centroids(self):
        """
        Return cluster centroids in original feature scale if scaling was applied.

        Returns:
            np.ndarray: Centroids in original scale if scaling applied, else raw centroids.
        """
        if self.scaling_applied:
            return self.scaler.inverse_transform(self.centroids_)
        return self.centroids_

    def _validate_features(self):
        """
        Validates that all selected features exist in the DataFrame and are numeric.
        Also drops rows with nulls in the selected columns.
        """
        for col in self.features:
            if col not in self.df.columns:
                raise ValueError(f"Feature '{col}' is not in the DataFrame.")
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise TypeError(f"Feature '{col}' is not numeric.")

        # Drop rows with nulls in selected features
        self.df = self.df.dropna(subset=self.features).reset_index(drop=True)

    def profile_clusters(self, pivot=True):
        """
        Generate a descriptive profile of each cluster using original (unscaled) feature values.

        Args:
            pivot (bool): Whether to pivot the output with statistics as rows and features as columns.

        Returns:
            pd.DataFrame: Profiled cluster summary.
        """
        df_with_labels = self.df.copy()
        df_with_labels['Cluster'] = self.labels_

        # Compute summary statistics per cluster
        cluster_summary = df_with_labels.groupby('Cluster')[self.features].agg(
            ['mean', 'std', 'min', 'max', 'count'])

        if pivot:
            # Flatten MultiIndex columns: (feature, stat) => 'feature - stat'
            cluster_summary.columns = [
                f"{feature} - {stat}" for feature, stat in cluster_summary.columns]

            # Transpose to have stats as rows (grouped by cluster)
            cluster_profile = cluster_summary.transpose()

            cluster_profile.columns = [
                f"Cluster {c}" for c in cluster_profile.columns]

            return cluster_profile

        return cluster_summary
