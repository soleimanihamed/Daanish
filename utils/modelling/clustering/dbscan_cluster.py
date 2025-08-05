# daanish/utils/modelling/clustering/dbscan_cluster.py

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from utils.preprocessing.scaler import Scaler


class DBSCANClustering:
    """
    A class to perform DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    on selected numerical features with optional scaling and PCA projection.

    Attributes:
        df (pd.DataFrame): The full cleaned dataset.
        features (List[str]): List of column names to use for clustering.
        eps (float): Maximum distance between two samples to be considered in the same neighborhood.
        min_samples (int): Minimum number of samples in a neighborhood for a point to be considered a core point.
        metric (str): The distance metric to use ('euclidean', 'manhattan', etc.).
        scaling_method (str): Method for scaling ('zscore' or 'minmax').
        scale (bool): Whether to apply scaling.
        handle_skew (bool): Whether to apply skewness correction.
        skew_method (str): Skewness reduction method ('log' or 'sqrt').
        skew_threshold (float): Threshold for identifying skewed features.
        scaler (Scaler): Instance of the Scaler class used for preprocessing.
        model (DBSCAN): Fitted DBSCAN model.
        labels_ (np.ndarray): Cluster labels assigned to each sample (-1 means noise).
        pca_ (PCA): Fitted PCA instance used for projecting to 2D.
        scaling_applied (bool): Tracks whether scaling was applied.
        suggest_eps_percentile (int): Percentile for k-distance to auto-suggest eps.
        tune_mode (str or None): Tuning mode. One of:
                            - None: No tuning.
                            - 'eps': Tune only eps using suggest_eps.
                            - 'tune_min_samples': Tune only min_samples using silhouette.
                            - 'grid': Grid search over both eps and min_samples.
        min_samples_range (list or None): Range of min_samples to try during tuning.
        eps_percentiles (list or None): List of percentiles for eps tuning during grid search.                   
    """

    def __init__(self, df: pd.DataFrame, features: list,
                 eps='auto', min_samples=5, metric='euclidean',
                 scaling_method='zscore', handle_skew=False, skew_method='log',
                 skew_threshold=1.0, scale=True, suggest_eps_percentile=90,
                 tune_mode=None,  # Options: None, 'eps', 'tune_min_samples', 'grid'
                 min_samples_range=None, eps_percentiles=None):

        self.df = df
        self.features = features
        self.eps = eps
        self.min_samples = min_samples
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

        self.suggest_eps_percentile = suggest_eps_percentile
        self.tune_mode = tune_mode
        self.min_samples_range = min_samples_range
        self.eps_percentiles = eps_percentiles

        # Tuning logic
        if tune_mode == 'grid':
            self.grid_search_results = self.grid_search_dbscan(
                min_samples_values=min_samples_range or list(range(3, 15)),
                eps_percentiles=eps_percentiles or list(range(80, 100, 2))
            )
            best_row = self.grid_search_results[self.grid_search_results["clusters"] > 1] \
                .sort_values(by="noise_percentage") \
                .head(1)
            if not best_row.empty:
                self.min_samples = int(best_row["min_samples"].values[0])
                self.eps = float(best_row["eps"].values[0])

        elif tune_mode == 'eps':
            self.eps = self.suggest_eps(
                k=self.min_samples, percentile=suggest_eps_percentile)

        elif tune_mode == 'tune_min_samples':
            results = self.tune_min_samples(
                min_samples_range=min_samples_range or list(range(3, 15)),
                eps_percentile=suggest_eps_percentile
            )
            # Pick best based on silhouette score or cluster count
            valid = results[results['n_clusters']
                            > 1].dropna(subset=['silhouette'])
            if not valid.empty:
                best = valid.sort_values(
                    by='silhouette', ascending=False).iloc[0]
                self.min_samples = int(best['min_samples'])
                self.eps = float(best['eps'])

        elif self.eps == 'auto':
            self.eps = self.suggest_eps(
                k=self.min_samples, percentile=suggest_eps_percentile)

    def fit(self):
        """
        Fit the DBSCAN model to the data and store labels and PCA projection.
        """
        X = self.df[self.features]

        if self.scale:
            X_scaled = self.scaler.fit_transform(X)
            self.scaling_applied = True
        else:
            X_scaled = X

        self.model = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        self.labels_ = self.model.fit_predict(X_scaled)
        self.pca_ = PCA(n_components=2).fit(X_scaled)
        return self.labels_

    def transform(self):
        """
        DBSCAN does not support transform after fitting, but we return the existing labels.

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
        centroids = df_temp[df_temp['Cluster'] != -
                            1].groupby('Cluster')[['PC1', 'PC2']].mean().values
        return centroids

    def compute_k_distances(self, k=None):
        """
        Compute sorted k-distances for DBSCAN eps tuning.

        Args:
            k (int): Number of neighbors to use. Defaults to min_samples.

        Returns:
            np.ndarray: Sorted k-distances (1D array).
        """

        k = k or self.min_samples

        X = self.df[self.features]

        if self.scale:
            if not hasattr(self.scaler, 'columns') or self.scaler.columns is None:
                self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        neighbors = NearestNeighbors(n_neighbors=k)
        distances, _ = neighbors.fit(X_scaled).kneighbors(X_scaled)

        k_distances = np.sort(distances[:, k - 1])
        return k_distances

    def suggest_eps(self, k=None, percentile=90):
        """
        Suggest an `eps` value using a percentile of the k-distance curve.

        Args:
            k (int): Number of neighbors to use. Defaults to min_samples.
            percentile (float): Percentile of the k-distance distribution.

        Returns:
            float: Suggested eps value.
        """
        k_distances = self.compute_k_distances(k)
        return np.percentile(k_distances, percentile)

    def tune_min_samples(self, min_samples_range=None, eps_percentile=None, verbose=True):
        """
        Tune min_samples by evaluating cluster counts, noise ratio, and silhouette score.

        Args:
            min_samples_range (list): List of min_samples values to test.
            eps_percentile (float): Percentile for auto-tuning eps. Defaults to self.suggest_eps_percentile.
            verbose (bool): Whether to print the results.

        Returns:
            pd.DataFrame: DataFrame with tuning results.
        """
        if min_samples_range is None:
            min_samples_range = list(range(4, 21))

        eps_percentile = eps_percentile or self.suggest_eps_percentile
        X = self.df[self.features]
        X_scaled = self.scaler.transform(X) if self.scale else X

        results = []

        for ms in min_samples_range:
            eps_val = self.suggest_eps(k=ms, percentile=eps_percentile)
            model = DBSCAN(eps=eps_val, min_samples=ms, metric=self.metric)
            labels = model.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = np.mean(labels == -1) * 100

            if n_clusters > 1:
                sil_score = silhouette_score(X_scaled, labels)
            else:
                sil_score = None

            results.append({
                'min_samples': ms,
                'eps': round(eps_val, 4),
                'n_clusters': n_clusters,
                'silhouette': round(sil_score, 4) if sil_score is not None else None,
                'noise_pct': round(noise_pct, 2)
            })

        df_results = pd.DataFrame(results)

        if verbose:
            print(df_results)

        return df_results

    def tune_eps(self, min_samples=None, percentiles=None):
        """
        Tune `eps` using a range of percentiles from the k-distance distribution.

        Args:
            min_samples (int): Value to use for k-distance calculation. Defaults to self.min_samples.
            percentiles (list): List of percentiles to try for eps selection.

        Returns:
            pd.DataFrame: A DataFrame with percentiles, eps values, and number of clusters.
        """
        if percentiles is None:
            percentiles = list(range(80, 100, 2))  # Try 80%, 82%, ..., 98%

        min_samples = min_samples or self.min_samples
        results = []

        for p in percentiles:
            eps_val = self.suggest_eps(k=min_samples, percentile=p)
            self.eps = eps_val
            self.min_samples = min_samples
            labels = self.fit_predict()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            results.append({
                "Percentile": p,
                "Eps": round(eps_val, 4),
                "Clusters": n_clusters
            })

        return pd.DataFrame(results)

    def grid_search_dbscan(self, min_samples_values=None, eps_percentiles=None):
        """
        Perform a grid search over `min_samples` and `eps` (via percentiles)
        to find a good combination for DBSCAN clustering.

        Args:
            min_samples_values (list): List of min_samples values to try.
            eps_percentiles (list): List of percentiles for eps suggestion.

        Returns:
            pd.DataFrame: A DataFrame with each combination and number of clusters.
        """
        if min_samples_values is None:
            min_samples_values = list(range(3, 15))  # You can adjust the range

        if eps_percentiles is None:
            eps_percentiles = list(range(80, 100, 2))

        results = []

        for min_samples in min_samples_values:
            for p in eps_percentiles:
                eps_val = self.suggest_eps(k=min_samples, percentile=p)
                self.eps = eps_val
                self.min_samples = min_samples
                labels = self.fit_predict()
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                noise_pct = np.mean(labels == -1) * 100

                results.append({
                    "min_samples": min_samples,
                    "eps_percentile": p,
                    "eps": round(eps_val, 4),
                    "clusters": n_clusters,
                    "noise_percentage": round(noise_pct, 2)
                })

        return pd.DataFrame(results)

    def add_cluster_column(self, base_col_name="dbscan_cluster"):
        """
        Adds cluster labels as a new column in the original DataFrame.
        If a column with the desired name already exists, appends a numeric suffix.

        Args:
            base_col_name (str): Desired base name of the cluster column.

        Returns:
            pd.DataFrame: DataFrame with the new cluster column added.
        """
        df_copy = self.df.copy()
        col_name = base_col_name

        # Check for name conflicts and add suffix if needed
        suffix = 1
        while col_name in df_copy.columns:
            col_name = f"{base_col_name}_{suffix}"
            suffix += 1

        df_copy[col_name] = self.labels_
        return df_copy
