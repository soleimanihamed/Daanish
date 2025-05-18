# daanish/utils/modelling/clustering/hierarchical_cluster.py

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from utils.preprocessing.scaler import Scaler


class HierarchicalClustering:
    """
    A class to perform Agglomerative (Hierarchical) Clustering on selected numerical features
    with optional scaling and PCA projection.

    Attributes:
        df (pd.DataFrame): The full cleaned dataset.
        features (List[str]): List of column names to use for clustering.
        n_clusters (int): Number of clusters to form.
        linkage (str): Linkage criterion to use ('ward', 'complete', 'average', 'single').
        metric (str): Metric used to compute linkage ('euclidean', 'manhattan', l1, l2, cosine, precomputed).
        scaling_method (str): Method for scaling ('zscore' or 'minmax').
        scale (bool): Whether to apply scaling.
        handle_skew (bool): Whether to apply skewness correction.
        skew_method (str): Skewness reduction method ('log' or 'sqrt').
        skew_threshold (float): Threshold for identifying skewed features.
        scaler (Scaler): Instance of the Scaler class used for preprocessing.
        model (AgglomerativeClustering): Fitted AgglomerativeClustering model.
        labels_ (np.ndarray): Cluster labels assigned to each sample.
        pca_ (PCA): Fitted PCA instance used for projecting to 2D.
        scaling_applied (bool): Tracks whether scaling was applied.

        Note: If you use linkage='ward', the only valid metric is 'euclidean' (this is a requirement).
    """

    def __init__(self, df: pd.DataFrame, features: list, n_clusters=3,
                 linkage='ward', metric='euclidean',
                 scaling_method='zscore', handle_skew=False, skew_method='log',
                 skew_threshold=1.0, scale=True):

        self.df = df
        self.features = features
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold
        self.scaler = Scaler(method=scaling_method,
                             handle_skew=handle_skew,
                             skew_method=skew_method,
                             skew_threshold=skew_threshold)
        self.scale = scale
        self.model = None
        self.labels_ = None
        self.pca_ = None
        self.scaling_applied = False

        self._validate_features()

    def fit(self):
        """
        Fit the Agglomerative Clustering model to scaled data and store labels and PCA projection.
        """
        X = self.df[self.features]

        if self.scale:
            X_scaled = self.scaler.fit_transform(X)
            self.scaling_applied = True
        else:
            X_scaled = X

        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             linkage=self.linkage,
                                             metric=self.metric)
        self.labels_ = self.model.fit_predict(X_scaled)
        self.pca_ = PCA(n_components=2).fit(X_scaled)
        return self.labels_

    def transform(self):
        """
        Hierarchical clustering does not support `transform` after fitting like KMeans,
        but we return the existing labels.

        Returns:
            np.ndarray: Cluster labels for each observation.
        """
        return self.labels_

    def fit_predict(self):
        """
        Fit the model and return cluster labels.

        Returns:
            np.ndarray: Cluster labels.
        """
        return self.fit()

    def project_pca(self):
        """
        Project the scaled data into 2D PCA space for visualization.

        Returns:
            np.ndarray: 2D PCA coordinates of each observation.
        """
        X = self.df[self.features]
        X_scaled = self.scaler.transform(X) if self.scale else X
        return self.pca_.transform(X_scaled)

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

        cluster_summary = df_with_labels.groupby('Cluster')[self.features].agg(
            ['mean', 'std', 'min', 'max', 'count'])

        if pivot:
            cluster_summary.columns = [
                f"{feature} - {stat}" for feature, stat in cluster_summary.columns]
            cluster_profile = cluster_summary.transpose()
            cluster_profile.columns = [
                f"Cluster {c}" for c in cluster_profile.columns]
            return cluster_profile

        return cluster_summary

    def get_cluster_means_in_pca_space(self, pca_components):
        """
        Compute the mean PCA position (2D) of each cluster to simulate centroids for visualization.

        Args:
            pca_components (np.ndarray): 2D PCA projection of the data.

        Returns:
            np.ndarray: 2D coordinates representing the mean location of each cluster.
        """
        df_temp = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
        df_temp['Cluster'] = self.labels_
        centroids = df_temp.groupby('Cluster')[['PC1', 'PC2']].mean().values
        return centroids
