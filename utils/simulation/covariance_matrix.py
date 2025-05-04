# daanish/utils/simulation/covariance_matrix.py

import numpy as np
import pandas as pd


class CovarianceMatrix:
    """
    Computes and stores the covariance matrix for any numerical data.
    """

    def __init__(self, data: pd.DataFrame, annualize: bool = False, periods_per_year: int = 252):
        """
        :param data: A DataFrame containing numerical values.
        :param annualize: Whether to scale the covariance matrix by periods_per_year.
        :param periods_per_year: Number of periods in a year (e.g., 252 for daily financial data).
        """
        self.data = data.select_dtypes(
            include=[np.number])  # Filter only numeric columns
        self.annualize = annualize
        self.periods_per_year = periods_per_year
        self.cov_matrix = self._compute_covariance()

    def _compute_covariance(self) -> pd.DataFrame:
        cov = self.data.cov()
        if self.annualize:
            cov *= self.periods_per_year
        return cov

    def get_matrix(self) -> pd.DataFrame:
        return self.cov_matrix
