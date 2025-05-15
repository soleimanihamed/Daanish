# utils/eda/pca_analyzer.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from utils.preprocessing.scaler import Scaler
import seaborn as sns


class PCAAnalyzer:
    """
    A class for performing PCA analysis as part of exploratory data analysis (EDA).
    It includes scaling, dimensionality reduction, and visualization of explained variance and loadings.
    """

    def __init__(self, n_components=None, scaler_params=None):
        """
        Initialize the PCAAnalyzer.

        Parameters:
        n_components (int or None): Number of principal components to keep. 
                                    If None, all components are kept.
        scaler_params (dict or None): Parameters to initialize the custom Scaler. 
                                If None, defaults to {'method': 'zscore'}.
        """
        self.n_components = n_components
        self.scaler_params = scaler_params if scaler_params is not None else {
            'method': 'zscore'}
        self.scaler = Scaler(**self.scaler_params)
        self.pca = None
        self.loadings = None
        self.explained_variance = None
        self.pc_scores = None
        self.feature_names = None
        self.numeric_features_used = None

    def fit(self, X: pd.DataFrame):
        """
        Fit PCA on the numeric features of the input DataFrame.

        Parameters:
        X (pd.DataFrame): Input DataFrame, may include non-numeric columns.

        Returns:
        self: Fitted PCAAnalyzer instance.
        """
        # Select only numeric columns
        X_numeric = X.select_dtypes(include=[np.number])
        self.numeric_features_used = X_numeric.columns.tolist()
        self.feature_names = X_numeric.columns

        # Standardize numeric data
        X_scaled_df = self.scaler.fit_transform(
            X_numeric, columns=self.numeric_features_used)

        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pc_scores = self.pca.fit_transform(X_scaled_df)

        # Store explained variance and loadings
        self.explained_variance = self.pca.explained_variance_ratio_
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )

        if X.shape[1] != X_numeric.shape[1]:
            print(f"[INFO] PCA was applied to {X_numeric.shape[1]} numeric features only. "
                  f"{X.shape[1] - X_numeric.shape[1]} non-numeric features were excluded.")

        return self

    def get_scores_df(self):
        """
        Get the DataFrame of principal component scores.

        Returns:
        pd.DataFrame: Transformed data in principal component space.
        """
        return pd.DataFrame(
            self.pc_scores,
            columns=[f'PC{i+1}' for i in range(self.pc_scores.shape[1])]
        )
