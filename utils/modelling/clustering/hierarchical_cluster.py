# daanish/utils/modelling/clustering/hierarchical_cluster.py

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from utils.preprocessing.scaler import Scaler
from prince import MCA
import pandas as pd
import numpy as np


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
        tune_mode (str or None): If 'grid', performs tuning over cluster numbers.
        cluster_range (list or None): List of integers for possible cluster counts.
        grid_search_results (pd.DataFrame): DataFrame of tuning results (if applicable).
        mca_mode (bool): Whether to treat the input features as MCA coordinates (skips scaling and skew correction).

        Note: If you use linkage='ward', the only valid metric is 'euclidean' (this is a requirement).
    """

    def __init__(self, df: pd.DataFrame, features: list, n_clusters=3,
                 linkage='ward', metric='euclidean',
                 scaling_method='zscore', handle_skew=False, skew_method='log',
                 skew_threshold=1.0, scale=True, tune_mode=None, cluster_range=None,
                 mca_mode=False):

        self.df = df
        self.features = features
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold
        self.scale = scale
        self.model = None
        self.labels_ = None
        self.reducer_ = None
        self.scaling_applied = False
        self.tune_mode = tune_mode
        self.cluster_range = cluster_range
        self.grid_search_results = None
        self.mca_mode = mca_mode

        if not self.mca_mode:
            self.scaler = Scaler(method=scaling_method,
                                 handle_skew=handle_skew,
                                 skew_method=skew_method,
                                 skew_threshold=skew_threshold)

        else:
            self.scaler = None

        self._validate_features()

    def fit(self):
        """
        Fit the Agglomerative Clustering model to the data.

        If `mca_mode` is True, applies Multiple Correspondence Analysis to transform categorical data.
        Otherwise, it applies optional scaling and PCA.

        If grid search is enabled (tune_mode='grid'), it searches for the optimal number of clusters
        in the specified cluster_range using the Silhouette score. The best number of clusters is then used
        for final clustering.

        Returns:
            np.ndarray: Cluster labels assigned to each sample.
        """
        X = self.df[self.features]

        if self.mca_mode:
            mca = MCA(n_components=2, random_state=42)
            X_transformed = mca.fit_transform(X)
            self.reducer_ = mca  # store for consistency
            self.scaling_applied = False
        else:
            if self.scale:
                X_transformed = self.scaler.fit_transform(X)
                self.scaling_applied = True
            else:
                X_transformed = X

            # Store PCA projection for plotting/visualisation (numeric case only)
            self.reducer_ = PCA(n_components=2).fit(X_transformed)

        # Run grid search to select optimal number of clusters
        if self.tune_mode == 'grid' and self.cluster_range:
            self.run_grid_search()
            best_row = self.grid_search_results.sort_values(
                by='Silhouette', ascending=False).iloc[0]
            self.n_clusters = int(best_row['n_clusters'])

        self.model = AgglomerativeClustering(n_clusters=self.n_clusters,
                                             linkage=self.linkage,
                                             metric=self.metric)
        self.labels_ = self.model.fit_predict(X_transformed)

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

    def project_2d(self):
        """
        Project the scaled data into 2D PCA or MCA space for visualization.

        Returns:
            np.ndarray: 2D  coordinates of each observation in the reduced space.
        """
        X = self.df[self.features]

        if self.mca_mode:
            X_projected = self.reducer_.transform(X)
        else:
            X_scaled = self.scaler.transform(X) if self.scale else X
            X_projected = self.reducer_.transform(X_scaled)

        return X_projected

    def _validate_features(self):
        """
        Validates that all selected features exist in the DataFrame.

        - If mca_mode is False: ensures features are numeric.
        - If mca_mode is True: allows categorical, boolean, or object types.

        Also drops rows with nulls in the selected columns.
        """
        for col in self.features:
            if col not in self.df.columns:
                raise ValueError(f"Feature '{col}' is not in the DataFrame.")

            if self.mca_mode:
                if not (pd.api.types.is_categorical_dtype(self.df[col]) or
                        pd.api.types.is_object_dtype(self.df[col]) or
                        pd.api.types.is_bool_dtype(self.df[col])):
                    raise TypeError(
                        f"Feature '{col}' must be categorical (object, bool, or category) in mca_mode.")
            else:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise TypeError(
                        f"Feature '{col}' must be numeric when mca_mode is False.")

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

        if self.mca_mode:
            # For categorical features: show mode and frequency
            summary = df_with_labels.groupby('Cluster')[self.features].agg(
                lambda x: x.value_counts().index[0])
            return summary.T if pivot else summary
        else:
            # For numeric features
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

    def get_cluster_means_in_2d_space(self, projection_2d):
        """
        Compute the mean 2D position of each cluster (in PCA, MCA, or any 2D projection).

        Args:
            projection_2d (np.ndarray or pd.DataFrame): 2D projection of the data.

        Returns:
            pd.DataFrame: 2D coordinates representing the mean location of each cluster.
        """

        if isinstance(projection_2d, np.ndarray):
            df_temp = pd.DataFrame(projection_2d, columns=['Dim1', 'Dim2'])
        else:
            df_temp = projection_2d.copy()
            df_temp.columns = ['Dim1', 'Dim2']  # Standardize for clarity

        df_temp['Cluster'] = self.labels_
        centroids = df_temp.groupby('Cluster')[['Dim1', 'Dim2']].mean()

        return centroids

    def run_grid_search(self):
        """
        Perform grid search to evaluate clustering quality over a range of n_clusters.
        Stores results in self.grid_search_results.
        """
        X = self.df[self.features]

        if self.mca_mode:
            mca = MCA(n_components=2, random_state=42)
            X_transformed = mca.fit_transform(X)
            self.reducer_ = mca
            self.scaling_applied = False
        else:
            if self.scale:
                X_transformed = self.scaler.fit_transform(X)
                self.scaling_applied = True
            else:
                X_transformed = X

        results = []

        for k in self.cluster_range:
            model = AgglomerativeClustering(
                n_clusters=k, linkage=self.linkage, metric=self.metric)
            labels = model.fit_predict(X_transformed)

            # Silhouette score can only be computed when there is more than 1 cluster and less than number of samples
            if 1 < len(np.unique(labels)) < len(X_transformed):
                sil_score = silhouette_score(X_transformed, labels)
            else:
                sil_score = np.nan

            results.append({
                'n_clusters': k,
                'Silhouette': sil_score
            })

        self.grid_search_results = pd.DataFrame(results)
        return self.grid_search_results
