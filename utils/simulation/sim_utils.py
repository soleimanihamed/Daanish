# daanish/utils/simulation/sim_utils.py

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

    def __init__(self, parameters=None, num_simulations=10000):
        self.parameters = np.array(
            parameters) if parameters is not None else None
        self.num_simulations = num_simulations
        self.n = len(self.parameters) if self.parameters is not None else None

    def simulate_poisson(self):
        """
        Simulates random draws from the Poisson distribution for each parameter.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing Poisson-distributed simulations for each variable.
        """
        if self.parameters is None:
            raise ValueError(
                "Poisson simulation requires parameters (e.g., lambda values).")

        A = np.ones((self.num_simulations, self.n)) * self.parameters
        poisson_draws = np.random.poisson(A)
        df_poisson = pd.DataFrame(poisson_draws, columns=[
                                  f"Variable_{i+1}" for i in range(self.n)])
        return df_poisson

    def simulate_normal(self, target_skew=0, target_kurt=3, num_variables=None):
        """
        Simulates uncorrelated normal distributions with optional adjustment for skewness and kurtosis.

        Parameters
        ----------
        target_skew : float, optional
            Desired skewness for each variable (default is 0, meaning symmetric).
        target_kurt : float, optional
            Desired kurtosis for each variable (default is 3, i.e., normal kurtosis).
        num_variables : int, optional
            Number of variables to simulate. Required if `parameters` are not provided.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing uncorrelated simulated values adjusted for skewness and kurtosis.
        """

        if self.parameters is not None:
            n_vars = self.n
        elif num_variables is not None:
            n_vars = num_variables
        else:
            raise ValueError(
                "Provide either `parameters` or `num_variables` for normal simulation.")

        random_matrix = self._generate_uncorrelated_random(
            self.num_simulations, n_vars, target_skew, target_kurt
        )

        # If parameters are provided, scale the results by parameters (e.g., standard deviations, volatilities, etc.)
        if self.parameters is not None:
            random_matrix = random_matrix * self.parameters

        df_normal = pd.DataFrame(random_matrix, columns=[
                                 f"Variable_{i+1}" for i in range(n_vars)])
        return df_normal

    def _generate_uncorrelated_random(self, num_samples, num_variables, target_skew, target_kurt):
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

        # Skip decorrelation if only one variable
        if num_variables == 1:
            return random_numbers

        # Decorrelate
        cov_matrix = np.cov(random_numbers, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        decorrelated = random_numbers @ eigenvectors @ np.diag(
            1 / np.sqrt(eigenvalues)) @ eigenvectors.T

        for i in range(num_variables):
            np.random.shuffle(decorrelated[:, i])

        return decorrelated
