# daanish/utils/modelling/clustering/hierarchical_gower_cluster.py

import pandas as pd
import numpy as np
import gower
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
from sklearn.metrics import silhouette_score
from utils.preprocessing.scaler import Scaler
from prince import FAMD


class HierarchicalGowerClustering:
    """
    A class to perform Hierarchical Agglomerative Clustering on datasets 
    with mixed numerical and categorical features using Gower distance.
    """

    def __init__(self, df, categorical_features, numerical_features,
                 n_clusters=3, linkage_method='average', verbose=0, random_state=42,
                 tune_mode=None, cluster_range=None, linkage_methods_to_tune=None,
                 scale=True, scaling_method='zscore',
                 handle_skew=False, skew_method='log', skew_threshold=1.0):
        """
        Initializes the HierarchicalGowerClustering class.

        This constructor sets up the configuration for hierarchical clustering
        using Gower distance. It handles data preprocessing steps like scaling
        and skewness correction for numerical features, validates input features,
        and can optionally perform hyperparameter tuning to find the optimal
        number of clusters and linkage method based on silhouette scores.
        Finally, it fits the clustering model.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be clustered.
        categorical_features : List[str]
            A list of column names in `df` to be treated as categorical features.
        numerical_features : List[str]
            A list of column names in `df` to be treated as numerical features.
        n_clusters : int, optional
            The number of clusters to form. This is used if `tune_mode` is None
            or if tuning does not yield a better result. Defaults to 3.
        linkage_method : str, optional
            The linkage method to use for hierarchical clustering (e.g., 'average',
            'complete', 'ward'). Defaults to 'average'.
        verbose : int, optional
            Verbosity level. Higher values print more information during
            processing. Defaults to 0.
        random_state : int, optional
            Seed for random number generation, primarily used for reproducibility
            in sub-processes like FAMD. Defaults to 42.
        tune_mode : str or None, optional
            If 'silhouette', performs hyperparameter tuning for `n_clusters` and
            `linkage_method` using silhouette scores. If None, uses the provided
            `n_clusters` and `linkage_method`. Defaults to None.
        cluster_range : List[int] or None, optional
            A list of integers representing the range of cluster numbers to
            evaluate during tuning. If None, defaults to `list(range(2, 8))`.
        linkage_methods_to_tune : List[str] or None, optional
            A list of linkage methods to evaluate during tuning. If None,
            defaults to `['average', 'complete', 'ward']`.
        scale : bool, optional
            Whether to apply scaling to numerical features. Defaults to True.
        scaling_method : str, optional
            The method to use for scaling numerical features ('zscore' or 'minmax').
            Defaults to 'zscore'.
        handle_skew : bool, optional
            Whether to apply skewness correction to numerical features before scaling.
            Defaults to False.
        skew_method : str, optional
            The method for skewness correction ('log' or 'sqrt'). Defaults to 'log'.
        skew_threshold : float, optional
            The absolute skewness value above which a numerical feature is
            considered skewed and transformed. Defaults to 1.0.

        Attributes
        ----------
        labels_ : np.ndarray or None
            Cluster labels assigned to each data point after fitting.
        linkage_matrix_ : np.ndarray or None
            The hierarchical clustering linkage matrix.
        distance_matrix_ : np.ndarray or None
            The computed Gower distance matrix.
        centroids_ : tuple or None
            A tuple containing two DataFrames: (numerical_centroids, categorical_centroids).
            Numerical centroids represent the mean of numerical features per cluster.
            Categorical centroids represent the mode of categorical features per cluster.
        tuning_results_ : pd.DataFrame or None
            A DataFrame containing the results of the hyperparameter tuning process,
            if `tune_mode` was enabled. Includes parameters and silhouette scores.
        scaler : Scaler or None
            The scaler object used for numerical feature scaling, if applied.
        scaling_applied : bool
            Indicates whether scaling was applied to numerical features.
        """
        self.df = df.copy()
        self.categorical_features = list(categorical_features)
        self.numerical_features = list(numerical_features)
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.verbose = verbose
        # Used for FAMD, less so for deterministic hierarchical
        self.random_state = random_state
        self.tune_mode = tune_mode
        # Default shorter range for hierarchical
        self.cluster_range = cluster_range or list(range(2, 8))
        self.linkage_methods_to_tune = linkage_methods_to_tune or [
            'average', 'complete', 'ward']

        self.scale = scale
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold

        self.labels_ = None
        self.linkage_matrix_ = None
        self.distance_matrix_ = None
        self.centroids_ = None  # (num_centroids_df, cat_centroids_df)
        self.tuning_results_ = None
        self.scaler = None
        self.scaling_applied = False

        self._validate_features()

        if self.scale and self.numerical_features:
            self.scaler = Scaler(method=self.scaling_method,
                                 handle_skew=self.handle_skew,
                                 skew_method=self.skew_method,
                                 skew_threshold=self.skew_threshold)
            # Ensure only non-empty numerical features are scaled
            if not self.df[self.numerical_features].empty:
                self.df[self.numerical_features] = self.scaler.fit_transform(
                    self.df[self.numerical_features])
            self.scaling_applied = True

        if self.tune_mode == 'silhouette':
            if self.verbose > 0:
                print("Starting cluster tuning...")
            self.tuning_results_ = self.tune_clusters_hierarchical()
            if not self.tuning_results_.empty:
                best_params = self.tuning_results_.sort_values(
                    'SilhouetteScore', ascending=False).iloc[0]
                self.n_clusters = int(best_params['n_clusters'])
                self.linkage_method = best_params['linkage_method']
                if self.verbose > 0:
                    print(f"Tuning complete. Best n_clusters: {self.n_clusters}, Best linkage: {self.linkage_method} "
                          f"with Silhouette Score: {best_params['SilhouetteScore']:.4f}")
            else:
                if self.verbose > 0:
                    print(
                        "Tuning yielded no results, using default n_clusters and linkage_method.")

        self.fit()

    def _validate_features(self):
        """
        Validates features and handles missing values.
        Numerical NA filled with mean, Categorical NA with 'Missing'.
        """
        all_features = self.numerical_features + self.categorical_features
        for col in all_features:
            if col not in self.df.columns:
                raise ValueError(
                    f"Feature '{col}' not found in the DataFrame.")

        for col in self.numerical_features:
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                try:  # Attempt to convert if possible, or raise error
                    self.df[col] = pd.to_numeric(self.df[col])
                    # Check again
                    if not pd.api.types.is_numeric_dtype(self.df[col]):
                        raise TypeError(
                            f"Numerical feature '{col}' could not be converted to numeric.")
                except (ValueError, TypeError):
                    raise TypeError(
                        f"Numerical feature '{col}' is not numeric and could not be converted.")
            # Impute numerical NAs with mean
            if self.df[col].isnull().any():
                self.df[col].fillna(self.df[col].mean(), inplace=True)

        for col in self.categorical_features:
            if pd.api.types.is_numeric_dtype(self.df[col]) and not pd.api.types.is_string_dtype(self.df[col]):
                # Allow numerical-looking categoricals if they are not purely float/int without distinct values
                # For Gower, it's better to ensure they are treated as objects/strings if truly categorical
                # Convert to string if numeric but intended as categorical
                self.df[col] = self.df[col].astype(str)
            # Impute categorical NAs with 'Missing'
            if self.df[col].isnull().any():
                self.df[col].fillna("Missing", inplace=True)
            # Ensure categorical type for gower
            self.df[col] = self.df[col].astype('object')

    def _calculate_gower_matrix(self):
        """Computes the Gower distance matrix."""
        if self.df.empty:
            raise ValueError(
                "DataFrame is empty, cannot compute Gower matrix.")

        df_for_gower = self.df[self.numerical_features +
                               self.categorical_features]

        # Create a boolean mask for categorical features
        # True if a column is categorical, False otherwise
        categorical_feature_mask = [
            col in self.categorical_features for col in df_for_gower.columns]

        if self.verbose > 0:
            print("Calculating Gower distance matrix...")
        # gower_matrix function expects a DataFrame.
        # It handles mixed types based on dtype or the cat_features mask.
        # Ensure dtypes are appropriate: numerical features as float/int, categorical as object/category.
        # Our _validate_features and scaling should have prepared dtypes.
        try:
            self.distance_matrix_ = gower.gower_matrix(
                df_for_gower, cat_features=categorical_feature_mask)
        except Exception as e:
            print(f"Error during Gower matrix calculation: {e}")
            # Fallback: Try without explicit mask if it causes issues with specific gower lib versions
            # print("Attempting Gower calculation by dtype inference.")
            # self.distance_matrix_ = gower.gower_matrix(df_for_gower)
            raise e  # Re-raise the exception if primary method fails

        if self.verbose > 0:
            print("Gower distance matrix calculated.")
        # The output of gower_matrix is a square distance matrix.
        # For scipy.cluster.hierarchy.linkage, a condensed distance matrix (upper triangle) is often preferred.
        # However, linkage can also accept a square matrix if it's a distance matrix.
        # Let's ensure it's in a format linkage can use, or we can convert it.
        # For now, we'll pass the square matrix; linkage should handle it.
        # If using sklearn's AgglomerativeClustering(affinity='precomputed'), it expects the square matrix.

    def _calculate_centroids(self):
        """Calculates centroids (mean for numerical, mode for categorical) for each cluster."""
        if self.labels_ is None:
            if self.verbose > 0:
                print("Labels not available, cannot calculate centroids.")
            return

        # Use the (potentially scaled) self.df for finding modes/means
        df_labeled = self.df.copy()
        df_labeled['Cluster'] = self.labels_

        # Inverse transform numerical features for interpretable centroids
        # This df_for_centroids will have numerical features in their original scale
        df_for_centroids = self.df.copy()
        if self.scaling_applied and self.scaler and self.numerical_features:
            if not df_for_centroids[self.numerical_features].empty:
                df_for_centroids[self.numerical_features] = self.scaler.inverse_transform(
                    df_for_centroids[self.numerical_features]
                )
        df_for_centroids['Cluster'] = self.labels_

        num_centroids_list = []
        cat_centroids_list = []

        for k in sorted(np.unique(self.labels_)):
            cluster_data_orig_scale = df_for_centroids[df_for_centroids['Cluster'] == k]

            if self.numerical_features:
                num_mean = cluster_data_orig_scale[self.numerical_features].mean(
                ).values
                num_centroids_list.append(num_mean)

            if self.categorical_features:
                cat_mode = cluster_data_orig_scale[self.categorical_features].mode(
                )
                # Handle cases where mode() might return multiple modes or be empty for a cluster
                if not cat_mode.empty:
                    cat_centroids_list.append(cat_mode.iloc[0].values)
                else:  # Append NaNs or placeholders if no mode found for a feature in a cluster
                    cat_centroids_list.append(
                        [np.nan] * len(self.categorical_features))

        num_centroids_df = pd.DataFrame(
            num_centroids_list, columns=self.numerical_features) if self.numerical_features else pd.DataFrame()
        cat_centroids_df = pd.DataFrame(
            cat_centroids_list, columns=self.categorical_features) if self.categorical_features else pd.DataFrame()

        self.centroids_ = (num_centroids_df, cat_centroids_df)

    def fit(self):
        """
        Fits the Hierarchical Gower Clustering model.
        Computes Gower distance, performs linkage, and assigns cluster labels.
        """
        if self.distance_matrix_ is None:
            self._calculate_gower_matrix()

        if self.distance_matrix_ is None:  # Check again if calculation failed
            raise SystemError(
                "Gower distance matrix could not be computed or is missing.")

        if self.verbose > 0:
            print(
                f"Performing hierarchical clustering with linkage: '{self.linkage_method}'...")
        # Convert square distance matrix to condensed form for scipy.linkage
        # (pdist format: 1D array representing the upper triangle)
        condensed_dist_matrix = self.distance_matrix_[
            np.triu_indices(self.distance_matrix_.shape[0], k=1)]

        try:
            self.linkage_matrix_ = scipy_linkage(
                condensed_dist_matrix, method=self.linkage_method)
        except ValueError as e:
            if 'valid linkage method' in str(e).lower() and self.linkage_method == 'ward':
                print("Warning: 'ward' linkage usually requires Euclidean distances. "
                      "Results with Gower distance might be suboptimal or error-prone. "
                      "Consider 'average' or 'complete' linkage.")
            raise e

        if self.verbose > 0:
            print("Assigning cluster labels...")
        # Cut the tree to get n_clusters
        self.labels_ = fcluster(self.linkage_matrix_,
                                self.n_clusters, criterion='maxclust')
        # Adjust labels to be 0-indexed if they are 1-indexed from fcluster
        if np.min(self.labels_) == 1:
            self.labels_ = self.labels_ - 1

        self._calculate_centroids()
        if self.verbose > 0:
            print("Hierarchical Gower clustering fit complete.")
        return self

    def fit_predict(self):
        """Fits the model and returns cluster labels."""
        self.fit()
        return self.labels_

    def tune_clusters_hierarchical(self):
        """
        Performs tuning over cluster numbers and linkage methods using silhouette score.
        """
        if self.distance_matrix_ is None:
            self._calculate_gower_matrix()

        if self.distance_matrix_ is None:
            print("Error: Gower distance matrix is missing, cannot perform tuning.")
            return pd.DataFrame()

        results = []
        if self.verbose > 0:
            print("Starting tuning for n_clusters and linkage_method...")

        for linkage_m in self.linkage_methods_to_tune:
            if self.verbose > 0:
                print(f"  Tuning with linkage: {linkage_m}")
            try:
                condensed_dist_matrix = self.distance_matrix_[
                    np.triu_indices(self.distance_matrix_.shape[0], k=1)]
                current_linkage_matrix = scipy_linkage(
                    condensed_dist_matrix, method=linkage_m)
            except Exception as e:
                if self.verbose > 0:
                    print(
                        f"    Could not compute linkage for {linkage_m}: {e}")
                continue

            for k_clusters in self.cluster_range:
                # Silhouette needs at least 2 clusters and < n_samples clusters
                if k_clusters <= 1 or k_clusters >= self.distance_matrix_.shape[0]:
                    continue
                try:
                    labels = fcluster(current_linkage_matrix,
                                      k_clusters, criterion='maxclust')
                    # Silhouette score requires at least 2 distinct labels
                    if len(np.unique(labels)) < 2:
                        if self.verbose > 0:
                            print(
                                f"    Skipping k={k_clusters} for linkage {linkage_m} - not enough distinct clusters formed.")
                        continue

                    score = silhouette_score(
                        self.distance_matrix_, labels, metric='precomputed')
                    results.append({
                        'linkage_method': linkage_m,
                        'n_clusters': k_clusters,
                        'SilhouetteScore': score
                    })
                    if self.verbose > 1:
                        print(
                            f"    k={k_clusters}, Silhouette: {score:.4f} for linkage {linkage_m}")
                except Exception as e:
                    if self.verbose > 0:
                        print(
                            f"    Error evaluating k={k_clusters} for linkage {linkage_m}: {e}")

        return pd.DataFrame(results)

    def profile_clusters(self):
        """
        Generates a cluster profile: modes for categorical, mean/std for numerical features.
        Numerical features are inverse-transformed to their original scale for profiling.
        """
        if self.labels_ is None:
            raise ValueError(
                "Model not fitted yet. Call fit() or fit_predict() first.")

        # Use a copy of the original df for profiling, then inverse scale numerical features
        df_profile = self.df.copy()  # self.df has scaled numerical features

        if self.scaling_applied and self.scaler and self.numerical_features:
            if not df_profile[self.numerical_features].empty:
                df_profile[self.numerical_features] = self.scaler.inverse_transform(
                    df_profile[self.numerical_features]
                )

        df_profile['Cluster'] = self.labels_

        # Initialize cat_summary and num_summary
        cat_summary = pd.DataFrame(
            index=sorted(df_profile['Cluster'].unique()))
        num_summary = pd.DataFrame(
            index=sorted(df_profile['Cluster'].unique()))

        if self.categorical_features:
            cat_summary = df_profile.groupby('Cluster')[self.categorical_features].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )

        if self.numerical_features:
            num_means = df_profile.groupby(
                'Cluster')[self.numerical_features].mean()
            num_stds = df_profile.groupby(
                'Cluster')[self.numerical_features].std()

            # Create multi-index columns for numerical summary: (feature, 'mean') and (feature, 'std')
            num_summary_multi_col = pd.DataFrame(index=num_means.index)
            for col in self.numerical_features:
                num_summary_multi_col[(col, 'mean')] = num_means[col]
                num_summary_multi_col[(col, 'std')] = num_stds[col]
            num_summary = num_summary_multi_col

         # Only try to concatenate if both have been populated (or if one is guaranteed to exist based on features)
        if self.categorical_features and self.numerical_features:
            # Ensure indices align if one group of features resulted in an empty aggregation for some reason
            cluster_profile = pd.concat([cat_summary, num_summary], axis=1)
        elif self.categorical_features:  # Only categorical features were present or processed
            cluster_profile = cat_summary
        elif self.numerical_features:  # Only numerical features were present or processed
            cluster_profile = num_summary
        else:  # No features to profile
            cluster_profile = pd.DataFrame(
                index=sorted(df_profile['Cluster'].unique()))
            if self.verbose > 0:
                print("Warning: No categorical or numerical features to profile.")

        return cluster_profile

    def project_clusters_famd(self, n_components: int = 2) -> pd.DataFrame:
        """
        Projects data onto lower dimensions using FAMD for visualization.
        Uses self.df which has scaled numerical features.
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted. Call `fit()` first.")

        # Data for FAMD: scaled numerical + original categorical from self.df
        df_for_famd = self.df[self.numerical_features +
                              self.categorical_features].copy()
        df_for_famd['Cluster'] = self.labels_

        X_famd = df_for_famd[self.numerical_features +
                             self.categorical_features]

        X_famd = X_famd.copy()
        # Clean and normalize categorical values
        for col in self.categorical_features:
            X_famd[col] = (
                X_famd[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .replace({'': 'UNKNOWN', 'NAN': 'UNKNOWN'})
            )
        # Convert to category dtype
        for col in self.categorical_features:
            X_famd[col] = X_famd[col].astype('category')

        famd_model = FAMD(n_components=n_components,
                          random_state=self.random_state)

        famd_projection_values = famd_model.fit_transform(X_famd)

        famd_projection_df = famd_projection_values.reset_index(drop=True)
        famd_projection_df.columns = [
            f'FAMD_Comp_{i+1}' for i in range(famd_projection_df.shape[1])]

        famd_projection_df['Cluster'] = df_for_famd['Cluster'].reset_index(
            drop=True)

        return famd_projection_df

    def get_cluster_means_in_2d_space(self, famd_components_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mean 2D position of each cluster in FAMD space.
        famd_components_df is the DataFrame output from project_clusters_famd.
        """
        if 'Cluster' not in famd_components_df.columns:
            famd_components_df_copy = famd_components_df.copy()
            # Ensure Cluster column is present
            famd_components_df_copy['Cluster'] = self.labels_
        else:
            famd_components_df_copy = famd_components_df

        # Assuming the first n_components columns are the FAMD components
        # If famd_components_df was created by project_clusters_famd, it has named columns
        component_cols = [
            col for col in famd_components_df_copy.columns if col.startswith('FAMD_Comp_')]
        if not component_cols:  # Fallback if columns are not named as expected
            # Exclude 'Cluster'
            component_cols = famd_components_df_copy.columns[:
                                                             famd_components_df_copy.shape[1]-1].tolist()

        # Ensure we only try to take mean of component columns
        if not component_cols:
            print("Warning: No component columns found for calculating 2D means.")
            return pd.DataFrame()

        centroids_2d = famd_components_df_copy.groupby(
            'Cluster')[component_cols].mean()
        return centroids_2d  # Index will be 'Cluster'

    def add_cluster_column(self, base_col_name="hierarchical_gower_cluster"):
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
