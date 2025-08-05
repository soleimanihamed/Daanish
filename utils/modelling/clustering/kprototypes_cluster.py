# daanish/utils/modelling/clustering/kprototypes_cluster.py

import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
from utils.preprocessing.scaler import Scaler
from prince import FAMD


class KPrototypesClustering:
    """
    A class to perform K-Prototypes clustering on datasets containing both numerical and categorical features.
    This hybrid clustering approach allows for mixed data types using the `kmodes` package.

    Attributes:
        df (pd.DataFrame): The cleaned input dataset.
        categorical_features (List[str]): List of categorical column names.
        numerical_features (List[str]): List of numerical column names.
        n_clusters (int): Number of clusters to form (used if tune_mode is None).
        init (str): Initialization method for K-Prototypes ('Huang' or 'Cao').
        n_init (int): Number of initializations to perform.
        verbose (int): Verbosity level.
        random_state (int): Random seed for reproducibility.
        tune_mode (str or None): If 'grid', performs tuning over clusters and init methods.
        cluster_range (List[int]): Range of cluster numbers to test in grid search.
        init_methods (List[str]): Initialization methods to test in tuning.
        scaling_method (str): Method for scaling numerical features ('zscore' or 'minmax').
        scale (bool): Whether to apply scaling to numerical features.
        handle_skew (bool): Whether to apply skewness correction before scaling.
        skew_method (str): Skewness correction method ('log' or 'sqrt').
        skew_threshold (float): Threshold to identify skewed features.
        model (KPrototypes): The fitted clustering model.
        labels_ (np.ndarray): Cluster labels for each data point.
        centroids_ (tuple): The cluster centroids (numerical and categorical).
        scaling_applied (bool): Whether scaling was applied.
        grid_search_results (pd.DataFrame or None): Tuning results if `tune_mode='grid'`.
    """

    def __init__(self, df, categorical_features, numerical_features,
                 n_clusters=3, init='Huang', n_init=10, verbose=0,
                 random_state=42, tune_mode=None, cluster_range=None,
                 init_methods=['Huang', 'Cao'],
                 scale=True, scaling_method='zscore',
                 handle_skew=False, skew_method='log', skew_threshold=1.0):

        # Input and configuration
        self.df = df.copy()
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.tune_mode = tune_mode
        self.cluster_range = cluster_range or list(range(2, 11))
        self.init_methods = init_methods
        self.scale = scale
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold

        # Outputs and internals
        self.labels_ = None
        self.model = None
        self.centroids_ = None
        self.grid_search_results = None
        self.scaling_applied = False

        # Validator
        self._validate_features()

        # Scaling setup
        if self.scale:
            self.scaler = Scaler(method=self.scaling_method,
                                 handle_skew=self.handle_skew,
                                 skew_method=self.skew_method,
                                 skew_threshold=self.skew_threshold)
            self.df[self.numerical_features] = self.scaler.fit_transform(
                self.df[self.numerical_features])
            self.scaling_applied = True

        # Grid search tuning
        if self.tune_mode == 'grid':
            self.grid_search_results = self.grid_search_kprototypes()
            best = self.grid_search_results.sort_values('Cost').head(1).iloc[0]
            self.n_clusters = int(best['n_clusters'])
            self.init = best['init']

        # Fit final model
        self.fit()

    def _validate_features(self):
        """
        Validates that all specified numerical and categorical features exist in the DataFrame,
        are of appropriate data types, and contain no missing values.

        - Ensures all provided columns are present.
        - Checks that numerical features are numeric and categorical are not.
        - Fills NA in categorical features with 'Missing' and in numerical with mean values.
        """

        # Check presence
        for col in self.numerical_features + self.categorical_features:
            if col not in self.df.columns:
                raise ValueError(
                    f"Feature '{col}' not found in the DataFrame.")

        # Validate data types
        for col in self.numerical_features:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise TypeError(f"Numerical feature '{col}' is not numeric.")

        for col in self.categorical_features:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                raise TypeError(
                    f"Categorical feature '{col}' appears to be numeric.")

        # Handle missing values
        self.df[self.numerical_features] = self.df[self.numerical_features].fillna(
            self.df[self.numerical_features].mean())
        self.df[self.categorical_features] = self.df[self.categorical_features].fillna(
            "Missing")

    def fit(self):
        """
        Fit the K-Prototypes clustering model on the dataset.

        This method initializes and trains the K-Prototypes model using the preprocessed
        numerical and categorical features stored in `self.X`. It assigns cluster labels
        to the data and extracts the final cluster centroids.
        """

        # Prepare feature matrix from preprocessed df
        X_num = self.df[self.numerical_features].values
        X_cat = self.df[self.categorical_features].values
        self.X = np.concatenate([X_num, X_cat], axis=1)

        # Identify column indices for categorical features in X
        self.cat_idx = list(range(len(self.numerical_features),
                                  len(self.numerical_features) + len(self.categorical_features)))

        # Fit K-Prototypes model
        self.model = KPrototypes(n_clusters=self.n_clusters, init=self.init,
                                 n_init=self.n_init, verbose=self.verbose,
                                 random_state=self.random_state)

        self.labels_ = self.model.fit_predict(self.X, categorical=self.cat_idx)

        # Get raw centroids from the model
        centroids = self.model.cluster_centroids_

        # Split numerical and categorical centroid parts
        num_centroids_scaled = centroids[:, :len(
            self.numerical_features)].astype(float)
        cat_centroids = centroids[:, len(self.numerical_features):]

        # Inverse transform numerical centroids if scaling was applied
        if self.scale:
            num_centroids = self.scaler.inverse_transform(num_centroids_scaled)
        else:
            num_centroids = num_centroids_scaled

        # Store as separate tuple for future use (e.g., in cluster profiling)
        self.centroids_ = (num_centroids, cat_centroids)

    def fit_predict(self):
        """
        Fit the model and return the cluster labels.

        This is a convenience method that calls `fit()` internally and returns the
        cluster labels assigned to each sample in the dataset.

        Returns:
            np.ndarray: Array of cluster labels for each observation.
        """

        self.fit()
        return self.labels_

    def grid_search_kprototypes(self):
        """
        Perform grid search over specified cluster numbers and initialization methods
        to identify the combination with the lowest clustering cost.

        Returns:
            pd.DataFrame: A DataFrame containing each (n_clusters, init) combination and its associated cost.
        """

        results = []

        # Prepare input matrix and categorical indices
        X = self.df[self.numerical_features +
                    self.categorical_features].to_numpy()
        combined_cols = self.numerical_features + self.categorical_features
        cat_idx = [combined_cols.index(col)
                   for col in self.categorical_features]

        for k in self.cluster_range:
            for init_method in self.init_methods:
                model = KPrototypes(n_clusters=k, init=init_method,
                                    n_init=self.n_init, verbose=0,
                                    random_state=self.random_state)
                labels = model.fit_predict(X, categorical=cat_idx)
                cost = model.cost_
                results.append({
                    'n_clusters': k,
                    'init': init_method,
                    'Cost': cost
                })

        return pd.DataFrame(results)

    def profile_clusters(self):
        """
        Generate a cluster profile summary, including the most frequent categorical values
        and the mean and standard deviation of numerical features per cluster.

        Returns:
            pd.DataFrame: A combined summary DataFrame indexed by cluster label, 
                        showing modal values for categorical features and 
                        mean/std for numerical features.
        """

        df_labeled = self.df.copy()
        df_labeled['Cluster'] = self.labels_

        # Inverse scale numerical features if scaling was applied
        if self.scaling_applied:
            df_labeled[self.numerical_features] = self.scaler.inverse_transform(
                df_labeled[self.numerical_features])

        # Categorical: mode (most frequent)
        cat_summary = df_labeled.groupby('Cluster')[self.categorical_features].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )

        # Numerical: mean and standard deviation
        num_summary = df_labeled.groupby(
            'Cluster')[self.numerical_features].agg(['mean', 'std'])

        # Combine summaries
        cluster_profile = pd.concat([cat_summary, num_summary], axis=1)

        return cluster_profile

    def project_clusters_famd(self, n_components=2):
        """
        Project the full dataset (numerical + categorical) onto a lower-dimensional space
        using Factor Analysis of Mixed Data (FAMD), suitable for visualising mixed-type data.

        This method allows visualisation of clustering structure in 2D while preserving the
        influence of both numerical and categorical variables.

        Args:
            n_components (int): Number of FAMD components to project onto. Default is 2.

        Returns:
            pd.DataFrame: A DataFrame containing the projected components and the assigned cluster labels.
        """
        # Ensure dependencies and input integrity
        if self.labels_ is None:
            raise ValueError(
                "Model must be fitted before projecting. Call `fit()` first.")

        # Prepare combined data
        df_with_labels = self.df.copy()
        df_with_labels['Cluster'] = self.labels_

        # Initialize and fit FAMD
        famd = FAMD(n_components=n_components, random_state=self.random_state)
        famd_projection = famd.fit_transform(
            df_with_labels[self.numerical_features + self.categorical_features])

        # Add cluster labels for plotting
        famd_projection['Cluster'] = self.labels_

        return famd_projection

    def get_cluster_means_in_2d_space(self, famd_components):
        """
        Returns the mean 2D position of each cluster in FAMD space.

        Parameters:
            famd_components (pd.DataFrame): 2D FAMD-transformed components (same order as self.df)

        Returns:
            pd.DataFrame: A dataframe with the centroids' coordinates in the FAMD space
        """
        famd_df = famd_components.copy()
        famd_df['Cluster'] = self.labels_

        centroids = famd_df.groupby(
            'Cluster')[[famd_df.columns[0], famd_df.columns[1]]].mean()
        centroids.reset_index(drop=True, inplace=True)
        return centroids

    def add_cluster_column(self, base_col_name="kproto_cluster"):
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
