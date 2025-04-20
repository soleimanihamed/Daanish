# utils/simulation/sim_utils.py

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class RandomSimulator:
    """
    A utility class for generating random simulations using various probability distributions.
    It supports standard Poisson simulations as well as normally-distributed random variables
    with controlled skewness and kurtosis, while ensuring variables remain uncorrelated.

    Parameters
    ----------
    parameters : array-like
        Array of input parameters (e.g., means or rates) for each variable.
    num_simulations : int, optional
        Number of simulations to generate (default is 10,000).
    """

    def __init__(self, parameters, num_simulations=10000):
        self.parameters = np.array(parameters)
        self.num_simulations = num_simulations
        self.n = len(self.parameters)

    def simulate_poisson(self):
        """
        Simulates random draws from the Poisson distribution for each parameter.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing Poisson-distributed simulations for each variable.
        """
        A = np.ones((self.num_simulations, self.n)) * self.parameters
        poisson_draws = np.random.poisson(A)
        df_poisson = pd.DataFrame(poisson_draws, columns=[
                                  f"Variable_{i+1}" for i in range(self.n)])
        return df_poisson

    def simulate_normal(self, target_skew=0, target_kurt=3):
        """
        Simulates uncorrelated normal distributions with optional adjustment for skewness and kurtosis.

        Parameters
        ----------
        target_skew : float, optional
            Desired skewness for each variable (default is 0, meaning symmetric).
        target_kurt : float, optional
            Desired kurtosis for each variable (default is 3, i.e., normal kurtosis).

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing simulated values adjusted for skewness and kurtosis.
        """
        random_matrix = self._generate_uncorrelated_random_with_skew_kurt(
            self.num_simulations, self.n, target_skew, target_kurt
        )

        # Scale by parameters (e.g., standard deviations, volatilities, etc.)
        adjusted_matrix = random_matrix * self.parameters
        df_normal = pd.DataFrame(adjusted_matrix, columns=[
                                 f"Variable_{i+1}" for i in range(self.n)])
        return df_normal

    def _generate_uncorrelated_random_with_skew_kurt(self, num_samples, num_variables, target_skew, target_kurt):
        """
        Internal method for generating uncorrelated standard normal variables
        with desired skewness and kurtosis.

        Returns
        -------
        np.ndarray
            A matrix of shape (num_samples, num_variables) containing uncorrelated,
            standardized random numbers.
        """
        random_numbers = np.random.normal(
            loc=0, scale=1, size=(num_samples, num_variables))

        for i in range(num_variables):
            col = random_numbers[:, i]
            col = (col - np.mean(col)) / np.std(col)

            if target_skew != 0:
                col += target_skew * (col ** 3 - col)

            if target_kurt != 3:
                col += (target_kurt - 3) * (col ** 4 - 3 * col ** 2 + 1)

            col = (col - np.mean(col)) / np.std(col)
            random_numbers[:, i] = col

        # Decorrelate
        cov_matrix = np.cov(random_numbers, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        decorrelated = random_numbers @ eigenvectors @ np.diag(
            1 / np.sqrt(eigenvalues)) @ eigenvectors.T

        for i in range(num_variables):
            np.random.shuffle(decorrelated[:, i])

        return decorrelated
