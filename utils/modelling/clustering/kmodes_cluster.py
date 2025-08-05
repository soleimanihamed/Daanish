# daanish/utils/modelling/clustering/kmodes_cluster.py

from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


class KModesClustering:
    """
    A class to perform K-Modes clustering on categorical data, 
    with optional tuning of the number of clusters and initialization method.

    Attributes:
        df (pd.DataFrame): Full cleaned dataset.
        features (list): List of categorical features to use for clustering.
        n_clusters (int): Number of clusters to form.
        init (str): Initialization method ('Huang' or 'Cao').
        n_init (int): Number of time the k-modes algorithm will be run with different centroid seeds.
        verbose (int): Verbosity level.
        random_state (int or None): Random seed for reproducibility in KModes initialization.
        tune_mode (str or None): Tuning mode. One of:
            - None: No tuning.
            - 'grid': Perform grid search over cluster counts and init methods.
        cluster_range (list or None): Range of cluster numbers to try in tuning.
        init_methods (list or None): List of init methods to try in tuning ('Huang', 'Cao').
        model (KModes): Fitted KModes model.
        labels_ (np.ndarray): Cluster labels assigned to each sample.
        cost_ (float): Final cost of clustering.
    """

    def __init__(self, df, features, n_clusters=5, init='Huang', n_init=10,
                 verbose=0, tune_mode=None, cluster_range=None,
                 init_methods=None, random_state=42):
        self.df = df.copy()
        self.features = features
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.tune_mode = tune_mode
        self.cluster_range = cluster_range
        self.init_methods = init_methods
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.cost_ = None

        if self.tune_mode == 'grid':
            self.grid_search_results = self.grid_search_kmodes(
                cluster_range=cluster_range or list(range(2, 11)),
                init_methods=init_methods or ['Huang', 'Cao'],
                random_state=self.random_state
            )
            best_row = self.grid_search_results.sort_values(by="Cost").head(1)
            self.n_clusters = int(best_row["n_clusters"].values[0])
            self.init = best_row["init"].values[0]

        self.fit()
        self._validate_features()

    def fit(self):
        """Fit the K-Modes model on categorical features."""
        X = self.df[self.features].astype(str)
        self.model = KModes(n_clusters=self.n_clusters,
                            init=self.init,
                            n_init=self.n_init,
                            verbose=self.verbose,
                            random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)
        self.cost_ = self.model.cost_
        return self

    def grid_search_kmodes(self, cluster_range=None, init_methods=None, random_state=None):
        """
        Perform grid search over number of clusters and initialization methods for KModes.

        Args:
            cluster_range (list): List of integers representing cluster counts to try.
            init_methods (list): List of init methods to try, e.g., ['Huang', 'Cao'].
            random_state (int or None): Random seed for reproducibility in KModes initialization.

        Returns:
            pd.DataFrame: Grid search results with n_clusters, init method, cost, and inertia (if applicable).
        """
        if cluster_range is None:
            cluster_range = list(range(2, 11))

        if init_methods is None:
            init_methods = ['Huang', 'Cao']

        X = self.df[self.features].astype(str)
        results = []

        for k in cluster_range:
            for init_method in init_methods:
                model = KModes(n_clusters=k, init=init_method,
                               n_init=self.n_init, verbose=self.verbose, random_state=random_state)
                labels = model.fit_predict(X)
                cost = model.cost_

                results.append({
                    'n_clusters': k,
                    'init': init_method,
                    'Cost': cost
                })

        return pd.DataFrame(results)

    def transform(self):
        """Return the cluster labels for the original data."""
        X = self.df[self.features].astype(str)
        return self.model.predict(X)

    def fit_predict(self):
        """Fit and return the cluster labels."""
        self.fit()
        return self.labels_

    def get_centroids(self):
        """Return the categorical modes (centroids) for each cluster."""
        return pd.DataFrame(self.model.cluster_centroids_, columns=self.features)

    def profile_clusters(self, include_counts=True):
        """
        Return descriptive statistics for each cluster (category frequencies).

        Args:
            include_counts (bool): Whether to include count of members per cluster.

        Returns:
            pd.DataFrame: Cluster profiling with mode counts.
        """
        df_labeled = self.df.copy()
        df_labeled['Cluster'] = self.labels_

        profile = {}
        for feature in self.features:
            profile[feature] = df_labeled.groupby('Cluster')[feature].agg(
                lambda x: x.value_counts().index[0])

        profile_df = pd.DataFrame(profile)

        if include_counts:
            profile_df['Count'] = df_labeled.groupby('Cluster').size()

        return profile_df.reset_index()

    def _validate_features(self):
        """
        Ensure all selected features exist and convert them to string type.
        Also drops rows with missing values in those features.
        """
        for col in self.features:
            if col not in self.df.columns:
                raise ValueError(f"Feature '{col}' not found in DataFrame.")

        # Drop rows with missing values
        self.df = self.df.dropna(subset=self.features).reset_index(drop=True)

        # Convert features to string
        self.df[self.features] = self.df[self.features].astype(str)

    def project_mca(self, n_components=2):
        """
        Project categorical data into 2D space using MCA (via SVD on one-hot encoded data).

        Returns:
            np.ndarray: 2D coordinates of each observation.
        """
        X = self.df[self.features].astype(str)
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(X)

        mca = TruncatedSVD(n_components=n_components,
                           random_state=self.random_state)
        mca_components = mca.fit_transform(X_encoded)

        self.mca_ = mca
        self.encoder_ = encoder
        return mca_components

    def project_centroids_to_mca(self, n_components=2):
        """
        Project the cluster centroids (categorical modes) into MCA space for visualization.

        This method:
        - One-hot encodes the original dataset and cluster centroids.
        - Applies Truncated SVD (MCA) to reduce dimensionality.
        - Projects the centroids into the same 2D space as the clustered data.

        Parameters:
            n_components (int): Number of MCA components to project into (default is 2).

        Returns:
            np.ndarray: 2D array of MCA-projected centroids, shape (n_clusters, n_components)
        """

        # One-hot encode the full dataset
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_encoded = encoder.fit_transform(self.df[self.features].astype(str))

        # Apply MCA (via TruncatedSVD)
        svd = TruncatedSVD(n_components=n_components,
                           random_state=self.random_state)
        svd.fit(X_encoded)

        # Get categorical centroids and encode them
        centroids_df = self.get_centroids().astype(str)
        centroids_encoded = encoder.transform(centroids_df)

        # Project to MCA space
        centroids_mca = svd.transform(centroids_encoded)
        return centroids_mca

    def add_cluster_column(self, base_col_name="kmodes_cluster"):
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
