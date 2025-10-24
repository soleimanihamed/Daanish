# daanish/utils/simulation/sim_utils.py

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


class RandomSimulator:
    """
    A utility class for generating random simulations using various probability distributions.
    ...

    Parameters
    ----------
    parameters : array-like
        Array of input parameters (e.g., means or rates) for each variable.
    num_simulations : int, optional
        Number of simulations to generate (default is 10,000).
    column_names : list of str, optional
        Optional names for the simulated columns. If not provided, generic names are used.
    """

    def __init__(self, parameters=None, num_simulations=10000, column_names=None, decorrelate=True):
        self.parameters = np.array(
            parameters) if parameters is not None else None
        self.num_simulations = num_simulations
        self.n = len(self.parameters) if self.parameters is not None else None
        self.column_names = (
            column_names if column_names is not None
            else [f"Variable_{i+1}" for i in range(self.n)] if self.n is not None
            else None
        )
        self.decorrelate = decorrelate

    def simulate_poisson(self, decorrelate=None):
        """
        Simulates uncorrelated Poisson-distributed values for each parameter.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing uncorrelated Poisson-distributed simulations
            for each variable.
        """
        if self.parameters is None:
            raise ValueError(
                "Poisson simulation requires parameters (e.g., lambda values).")

        # Generate raw Poisson samples
        A = np.ones((self.num_simulations, self.n)) * self.parameters
        poisson_draws = np.random.poisson(A)

        # Standardize
        standardized = (poisson_draws - np.mean(poisson_draws,
                        axis=0)) / np.std(poisson_draws, axis=0)

        # Decorrelate
        if decorrelate and self.n > 1:
            cov_matrix = np.cov(standardized, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            decorrelated = standardized @ eigenvectors @ np.diag(
                1 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Shuffle each column to break residual patterns
            for i in range(self.n):
                np.random.shuffle(decorrelated[:, i])

            # Reintroduce Poisson-like structure by inverse-scaling
            decorrelated_scaled = decorrelated * \
                np.std(poisson_draws, axis=0) + np.mean(poisson_draws, axis=0)

            # Round to nearest integer
            decorrelated_scaled = np.round(decorrelated_scaled)

            # Replace negative values with zero
            decorrelated_scaled[decorrelated_scaled < 0] = 0

            df_poisson = pd.DataFrame(
                decorrelated_scaled, columns=self.column_names)

        else:
            df_poisson = pd.DataFrame(poisson_draws, columns=self.column_names)

        return df_poisson

    def simulate_normal(self, target_skew=0, target_kurt=3, num_variables=None, decorrelate=None, loc=0, scale=1):
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
        loc : float, optional
            Mean (location parameter) of the normal distribution. Default is 0.
        scale : float, optional
            Standard deviation (scale parameter) of the normal distribution. Default is 1.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing uncorrelated simulated values adjusted for skewness and kurtosis.
        """
        self.loc = loc
        self.scale = scale

        if self.parameters is not None:
            n_vars = self.n
        elif num_variables is not None:
            n_vars = num_variables
            self.parameters = np.ones(n_vars)
            self.n = n_vars
        else:
            raise ValueError(
                "Provide either `parameters` or `num_variables` for normal simulation.")

        # Validate column_names
        if self.column_names is not None:
            if len(self.column_names) != n_vars:
                raise ValueError(
                    f"Length of column_names ({len(self.column_names)}) does not match number of variables ({n_vars}).")
        else:
            self.column_names = [f"Variable_{i+1}" for i in range(n_vars)]

        random_matrix = self._generate_uncorrelated_random(
            self.num_simulations, n_vars, target_skew, target_kurt, decorrelate
        )

        # If parameters are provided, scale the results by parameters (e.g., standard deviations, volatilities, etc.)
        if self.parameters is not None:
            if self.parameters.ndim == 1:  # Only std provided
                random_matrix = random_matrix * self.parameters
            elif self.parameters.shape[1] == 2:  # Mean and std provided
                means = self.parameters[:, 0]
                stds = self.parameters[:, 1]
                random_matrix = random_matrix * stds + means
            else:
                raise ValueError(
                    "Normal simulation parameters must be 1D (std) or 2D (mean, std).")

        df_normal = pd.DataFrame(random_matrix, columns=self.column_names)

        return df_normal

    def _generate_uncorrelated_random(self, num_samples, num_variables, target_skew, target_kurt, decorrelate=True):
        """
        Internal method for generating standard normal variables with optional
        skewness, kurtosis, and decorrelation adjustment.

        Parameters
        ----------
        num_samples : int
            Number of random samples to generate.
        num_variables : int
            Number of independent variables (columns) to simulate.
        target_skew : float
            Desired skewness (0 for symmetric).
        target_kurt : float
            Desired kurtosis (3 for normal).
        decorrelate : bool, optional
            Whether to decorrelate the simulated variables (default is True).

        Returns
        -------
        np.ndarray
            A matrix of shape (num_samples, num_variables) containing
            simulated, standardized random numbers.
        """
        # Step 1: Generate base normal random numbers
        random_numbers = np.random.normal(
            loc=self.loc, scale=self.scale, size=(num_samples, num_variables))

        # Step 2: Adjust each variable for skewness and kurtosis
        for i in range(num_variables):
            col = random_numbers[:, i]
            col = (col - np.mean(col)) / np.std(col)

            if target_skew != 0:
                col += target_skew * (col ** 3 - col)

            if target_kurt != 3:
                col += (target_kurt - 3) * (col ** 4 - 3 * col ** 2 + 1)

            # Re-standardize after transformation
            col = (col - np.mean(col)) / np.std(col)
            random_numbers[:, i] = col

        # Step 3: Optionally decorrelate
        if decorrelate and num_variables > 1:
            cov_matrix = np.cov(random_numbers, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Apply whitening transformation
            decorrelated = random_numbers @ eigenvectors @ np.diag(
                1 / np.sqrt(eigenvalues)
            ) @ eigenvectors.T

            # Shuffle each column to break residual structure
            for i in range(num_variables):
                np.random.shuffle(decorrelated[:, i])

            return decorrelated

        # If decorrelation is off or only one variable
        return random_numbers

    def simulate_beta(self, decorrelate=None):
        """
        Simulates uncorrelated Beta-distributed values for each parameter set (α, β, loc, scale).

        Parameters shape:
        - If only (α, β) provided → shape: (n, 2), defaults loc=0, scale=1
        - If (α, β, loc, scale) provided → shape: (n, 4)

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing uncorrelated Beta-distributed simulations
            for each variable.
        """

        if self.parameters is None:
            raise ValueError(
                "Beta simulation requires parameter sets (alpha, beta) or (alpha, beta, loc, scale)."
            )

        # Ensure parameters are a NumPy array
        if not isinstance(self.parameters, np.ndarray):
            self.parameters = np.array(self.parameters, dtype=float)

        if self.parameters.shape[1] not in (2, 4):
            raise ValueError(
                f"Beta simulation requires parameters of shape ({self.n}, 2) or ({self.n}, 4)."
            )

        # Split parameters
        if self.parameters.shape[1] == 2:
            alpha = self.parameters[:, 0]
            beta = self.parameters[:, 1]
            loc = np.zeros(self.n)
            scale = np.ones(self.n)
        else:
            alpha = self.parameters[:, 0]
            beta = self.parameters[:, 1]
            loc = self.parameters[:, 2]
            scale = self.parameters[:, 3]

        # Generate raw beta samples with loc & scale
        beta_draws = np.array([
            loc[i] + scale[i] *
            np.random.beta(alpha[i], beta[i], size=self.num_simulations)
            for i in range(self.n)
        ]).T  # Shape: (num_simulations, n)

        # Standardize
        standardized = (beta_draws - np.mean(beta_draws, axis=0)
                        ) / np.std(beta_draws, axis=0)

        # Decorrelate
        if decorrelate and self.n > 1:
            cov_matrix = np.cov(standardized, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            decorrelated = standardized @ eigenvectors @ np.diag(
                1 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Shuffle each column to break residual patterns
            for i in range(self.n):
                np.random.shuffle(decorrelated[:, i])

            # Reintroduce Beta-like structure (with loc & scale)
            decorrelated_scaled = decorrelated * \
                np.std(beta_draws, axis=0) + np.mean(beta_draws, axis=0)
            df_beta = pd.DataFrame(decorrelated_scaled,
                                   columns=self.column_names)
        else:
            df_beta = pd.DataFrame(beta_draws, columns=self.column_names)

        return df_beta

    def simulate_lognormal(self, decorrelate=None):
        """
        Simulates uncorrelated Lognormal-distributed values for each parameter set (s, loc, scale).
        Parameters shape:
        - If (s, loc, scale) provided → shape: (n, 3)
        Returns
        -------
        pandas.DataFrame
            A DataFrame containing uncorrelated Lognormal-distributed simulations
            for each variable.
        """
        if self.parameters is None:
            raise ValueError(
                "Lognormal simulation requires parameter sets (s, loc, scale)."
            )

        if not isinstance(self.parameters, np.ndarray):
            self.parameters = np.array(self.parameters, dtype=float)

        if self.parameters.shape[1] != 3:
            raise ValueError(
                f"Lognormal simulation requires parameters of shape ({self.n}, 3)."
            )

        s = self.parameters[:, 0]
        loc = self.parameters[:, 1]
        scale = self.parameters[:, 2]

        meanlog = np.log(scale)  # convert scale to meanlog

        # Generate samples
        lognorm_draws = np.array([
            loc[i] + np.random.lognormal(meanlog[i],
                                         s[i], size=self.num_simulations)
            for i in range(self.n)
        ]).T

        # Standardize
        standardized = (lognorm_draws - np.mean(lognorm_draws,
                        axis=0)) / np.std(lognorm_draws, axis=0)

        # Decorrelate
        if decorrelate and self.n > 1:
            cov_matrix = np.cov(standardized, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            decorrelated = standardized @ eigenvectors @ np.diag(
                1 / np.sqrt(eigenvalues)) @ eigenvectors.T

            for i in range(self.n):
                np.random.shuffle(decorrelated[:, i])

            decorrelated_scaled = decorrelated * \
                np.std(lognorm_draws, axis=0) + np.mean(lognorm_draws, axis=0)
            df_lognorm = pd.DataFrame(
                decorrelated_scaled, columns=self.column_names)
        else:
            df_lognorm = pd.DataFrame(lognorm_draws, columns=self.column_names)

        return df_lognorm

    def simulate_gamma(self, decorrelate=None):
        """
        Simulates uncorrelated Gamma-distributed values for each parameter set (shape, loc, scale).
        Parameters shape:
        - shape (a), loc, scale → shape (n, 3)
        Returns
        -------
        pandas.DataFrame
        """
        if self.parameters is None:
            raise ValueError(
                "Gamma simulation requires parameter sets (shape, loc, scale).")

        if not isinstance(self.parameters, np.ndarray):
            self.parameters = np.array(self.parameters, dtype=float)

        if self.parameters.shape[1] != 3:
            raise ValueError(
                f"Gamma simulation requires parameters of shape ({self.n}, 3).")

        shape = self.parameters[:, 0]
        loc = self.parameters[:, 1]
        scale = self.parameters[:, 2]

        # Generate samples
        gamma_draws = np.array([
            loc[i] + np.random.gamma(shape[i], scale[i],
                                     size=self.num_simulations)
            for i in range(self.n)
        ]).T

        # Standardize
        standardized = (gamma_draws - np.mean(gamma_draws, axis=0)
                        ) / np.std(gamma_draws, axis=0)

        # Decorrelate
        if decorrelate and self.n > 1:
            cov_matrix = np.cov(standardized, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            decorrelated = standardized @ eigenvectors @ np.diag(
                1 / np.sqrt(eigenvalues)) @ eigenvectors.T

            for i in range(self.n):
                np.random.shuffle(decorrelated[:, i])

            decorrelated_scaled = decorrelated * \
                np.std(gamma_draws, axis=0) + np.mean(gamma_draws, axis=0)
            df_gamma = pd.DataFrame(
                decorrelated_scaled, columns=self.column_names)
        else:
            df_gamma = pd.DataFrame(gamma_draws, columns=self.column_names)

        return df_gamma
