# daanish/utils/simulation/covariance_matrix.py

import numpy as np
import pandas as pd


class CovarianceMatrix:
    """
    Computes and stores the covariance matrix for any numerical data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        :param data: A DataFrame containing numerical values.
        """
        self.data = data.select_dtypes(
            include=[np.number])  # Filter only numeric columns
        self.cov_matrix = self._compute_covariance()

    def _compute_covariance(self) -> pd.DataFrame:
        cov = self.data.cov()
        return cov

    def get_matrix(self) -> pd.DataFrame:
        return self.cov_matrix
