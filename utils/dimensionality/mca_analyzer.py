# utils/eda/mca_analyzer.py

import pandas as pd
import prince


class MCAAnalyzer:
    """
    A class for performing MCA (Multiple Correspondence Analysis) on categorical data.
    """

    def __init__(self, n_components=2):
        """
        Initialize the MCAAnalyzer.

        Parameters:
        n_components (int): Number of dimensions to retain.
        """
        self.n_components = n_components
        self.mca = None
        self.explained_inertia_proportions = None
        self.column_coordinates = None
        self.row_coordinates = None
        self.categorical_features_used = None
        self.raw_eigenvalues = None
        self.total_inertia = None

    def fit(self, X: pd.DataFrame):
        """
        Fit MCA on the categorical features of the input DataFrame.

        Parameters:
        X (pd.DataFrame): Input DataFrame, may include non-categorical columns.

        Returns:
        self: Fitted MCAAnalyzer instance.
        """

        X_cat = X.select_dtypes(include=['object', 'category'])
        self.categorical_features_used = X_cat.columns.tolist()

        if X_cat.empty:
            raise ValueError("No categorical features found for MCA.")

        self.mca = prince.MCA(n_components=self.n_components, random_state=42)
        self.mca = self.mca.fit(X_cat)
        self.row_coordinates = self.mca.transform(X_cat)
        self.column_coordinates = self.mca.column_coordinates(X_cat)

        # Calculate explained inertia proportions using eigenvalues_ and total_inertia_
        if hasattr(self.mca, 'eigenvalues_') and hasattr(self.mca, 'total_inertia_'):
            # These are the eigenvalues for each component
            self.raw_eigenvalues = self.mca.eigenvalues_
            # This is the single total inertia value
            self.total_inertia = self.mca.total_inertia_

            if self.total_inertia > 0:
                # self.mca.eigenvalues_ is typically a pandas Series or numpy array
                self.explained_inertia_proportions = self.raw_eigenvalues / self.total_inertia
            # Handle cases with eigenvalues but zero total inertia (unlikely but safe)
            elif len(self.raw_eigenvalues) > 0:
                self.explained_inertia_proportions = pd.Series(
                    [0.0] * len(self.raw_eigenvalues), index=self.raw_eigenvalues.index if hasattr(self.raw_eigenvalues, 'index') else None)
            else:
                self.explained_inertia_proportions = pd.Series([])

            # Ensure it's a pandas Series for consistency if your plotting function expects it
            if not isinstance(self.explained_inertia_proportions, pd.Series) and hasattr(self.explained_inertia_proportions, 'tolist'):
                self.explained_inertia_proportions = pd.Series(
                    self.explained_inertia_proportions)

        else:
            missing_attrs = []
            if not hasattr(self.mca, 'eigenvalues_'):
                missing_attrs.append("'eigenvalues_'")
            if not hasattr(self.mca, 'total_inertia_'):
                missing_attrs.append("'total_inertia_'")
            raise AttributeError(f"MCA object is missing required attributes: {', '.join(missing_attrs)}. "
                                 "Please verify the prince library version and API.")

        if X.shape[1] != X_cat.shape[1]:
            print(f"[INFO] MCA applied to {X_cat.shape[1]} categorical features. "
                  f"{X.shape[1] - X_cat.shape[1]} non-categorical features were excluded.")

        return self

    def get_row_coordinates_df(self):
        """
        Get the DataFrame of row coordinates (MCA scores for each record), with readable dimension names.

        Returns:
        pd.DataFrame: MCA-transformed data with renamed dimensions.
        """
        if self.row_coordinates is None:
            raise ValueError("MCA model has not been fitted yet.")

        renamed_coords = self.row_coordinates.copy()
        renamed_coords.columns = [
            f"Dim{i+1}" for i in range(renamed_coords.shape[1])]
        return renamed_coords

    def get_column_coordinates_df(self):
        """
        Return the coordinates of the original categories in the MCA space.
        """
        return self.column_coordinates

    def get_explained_inertia(self):
        """
        Return the proportion of inertia (variance) explained by each MCA component.
        """
        if self.explained_inertia_proportions is None:
            raise ValueError(
                "MCA has not been fitted or explained inertia could not be determined.")
        return self.explained_inertia_proportions

    def get_column_contributions_df(self):
        """
        Get the contributions of each category (column) to each MCA dimension (% contribution to inertia).

        Returns:
        pd.DataFrame: Contribution of each category to each dimension.
        """
        if self.mca is None:
            raise ValueError("MCA model has not been fitted yet.")
        return self.mca.column_contributions_
