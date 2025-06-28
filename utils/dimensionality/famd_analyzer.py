# utils/dimensionality/famd_analyzer.py

import pandas as pd
import prince
from utils.preprocessing.scaler import Scaler


class FAMDAnalyzer:
    """
    A class for performing FAMD (Factor Analysis of Mixed Data) on datasets with both categorical and numerical features.
    Includes built-in support for numerical feature scaling.
    """

    def __init__(self, n_components=None, scale_numerical=True, scaling_method='zscore',
                 handle_skew=False, skew_method='log', skew_threshold=1.0):
        """
        Initialize the FAMDAnalyzer.

        Parameters:
            n_components (int): Number of dimensions to retain.
            scale_numerical (bool): If True, numerical features will be scaled before analysis.
            scaling_method (str): Scaling method ('zscore' or 'minmax').
            handle_skew (bool): Whether to automatically transform skewed variables.
            skew_method (str): Method for skewness correction ('log' or 'sqrt').
            skew_threshold (float): Skewness value to trigger transformation.
        """
        self.n_components = n_components
        self.famd = None
        self.row_coordinates = None
        self.column_coordinates = None
        self.explained_inertia_proportions = None
        self.raw_eigenvalues = None
        self.total_inertia = None
        self.contributions_df = None
        self.num_features = []
        self.cat_features = []

        # Initialize the scaler if scaling is enabled
        self.scale_numerical = scale_numerical
        if self.scale_numerical:
            self.scaler = Scaler(
                method=scaling_method,
                handle_skew=handle_skew,
                skew_method=skew_method,
                skew_threshold=skew_threshold
            )
        else:
            self.scaler = None

    def fit(self, X: pd.DataFrame):
        """
        Fit FAMD on the input DataFrame.

        Parameters:
        X (pd.DataFrame): Input DataFrame with mixed data types.

        Returns:
        self: Fitted FAMDAnalyzer instance.
        """

        # Identify feature types
        self.num_features = X.select_dtypes(
            include=["number"]).columns.tolist()
        self.cat_features = X.select_dtypes(
            include=["object", "category"]).columns.tolist()

        if not self.cat_features and not self.num_features:
            raise ValueError("No usable features found for FAMD.")

        # Prepare the DataFrame for FAMD
        X_for_famd = X.copy()

        # If scaling is enabled and there are numerical features, scale them
        if self.scale_numerical and self.num_features:
            print("Applying scaling to numerical features...")
            # Use the internal Scaler to fit and transform the numerical columns
            scaled_numerical_df = self.scaler.fit_transform(
                X, columns=self.num_features)

            # Combine the newly scaled numerical data with the original categorical data
            X_for_famd = pd.concat(
                [scaled_numerical_df, X[self.cat_features]], axis=1)
            print("Scaling complete. Final DataFrame prepared for FAMD.")

        # Determine number of components if not specified
        if self.n_components is None:
            num_cats = sum(X_for_famd[cat].nunique()
                           for cat in self.cat_features)
            num_vars = len(self.cat_features) + len(self.num_features)
            self.n_components = num_cats + len(self.num_features) - num_vars
            print(
                f"[INFO] n_components not specified. Fitting with maximum possible components: {self.n_components}")

        # Fit the FAMD model on the (potentially scaled) data
        self.famd = prince.FAMD(
            n_components=self.n_components, random_state=42)
        self.famd = self.famd.fit(X_for_famd)

        # Extract results
        self.row_coordinates = self.famd.row_coordinates(X_for_famd)
        self.column_coordinates = self.famd.column_coordinates_

        # Eigenvalues and total inertia
        if hasattr(self.famd, 'eigenvalues_') and hasattr(self.famd, 'total_inertia_'):
            self.raw_eigenvalues = self.famd.eigenvalues_
            self.total_inertia = self.famd.total_inertia_

            if self.total_inertia > 0:
                self.explained_inertia_proportions = self.raw_eigenvalues / self.total_inertia
            else:
                self.explained_inertia_proportions = pd.Series(
                    [0.0] * len(self.raw_eigenvalues),
                    index=self.raw_eigenvalues.index if hasattr(
                        self.raw_eigenvalues, 'index') else None,
                )
        else:
            raise AttributeError(
                "FAMD object is missing required attributes: 'eigenvalues_' or 'total_inertia_'.")

        # Contributions
        self.contributions_df = self.famd.column_contributions_

        return self

    def get_row_coordinates_df(self):
        """
        Get the DataFrame of row coordinates (FAMD scores for each record), with readable dimension names.

        Returns:
        pd.DataFrame: FAMD-transformed data with renamed dimensions.
        """
        if self.row_coordinates is None:
            raise ValueError("FAMD model has not been fitted yet.")

        renamed_coords = self.row_coordinates.copy()
        renamed_coords.columns = [
            f"Dim{i+1}" for i in range(renamed_coords.shape[1])]
        return renamed_coords

    def get_column_coordinates_df(self):
        """
        Return the coordinates of the original variables in the FAMD space.
        """
        return self.column_coordinates

    def get_explained_inertia(self):
        """
        Return the proportion of inertia (variance) explained by each FAMD component.
        """
        if self.explained_inertia_proportions is None:
            raise ValueError(
                "FAMD has not been fitted or explained inertia could not be determined.")
        return self.explained_inertia_proportions

    def get_contributions(self):
        """
        Return contributions of original variables to the FAMD dimensions.

        Returns:
        pd.DataFrame: Contribution matrix.
        """
        if self.contributions_df is None:
            raise ValueError(
                "Contributions not available. Fit the model first.")
        return self.contributions_df
