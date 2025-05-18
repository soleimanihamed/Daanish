# daanish/utils/simulation/monte-carlo-simulator.py

import numpy as np
import pandas as pd

from utils.simulation.covariance_matrix import CovarianceMatrix
from utils.simulation.decomposition import Decomposer
from utils.simulation.sim_utils import RandomSimulator


class MonteCarloSimulator:
    """
    Monte Carlo Simulator using Rubinstein's approach.

    Monte Carlo Simulator for generating correlated variables simulations.

    This class computes correlated normal simulations from data using:
    - Covariance matrix estimation
    - Cholesky decomposition
    - Multivariate standard normal generation

    Attributes:
        data (pd.DataFrame): A DataFrame containing numerical values.
        num_simulations (int): Number of Monte Carlo simulations to perform.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.
        cholesky_matrix (pd.DataFrame): Lower-triangular Cholesky decomposition of covariance matrix.
        simulated_raw (pd.DataFrame): Simulated uncorrelated returns.
        simulated_final (pd.DataFrame): Final simulated correlated returns.
    """

    def __init__(self, data: pd.DataFrame, num_simulations: int = 10000):
        """
        Initialize the Monte Carlo simulator with variables data and simulation count.

        Args:
            data (pd.DataFrame): A DataFrame containing numerical values.
            num_simulations (int): Number of simulations to run (default: 10,000).
        """
        self.data = data.select_dtypes(include=[np.number])
        self.variable_names = self.data.columns.tolist()
        self.num_variables = len(self.variable_names)
        self.num_simulations = num_simulations

        self.cov_matrix = None
        self.cholesky_matrix = None
        self.simulated_raw = None
        self.simulated_final = None

    def run_simulation(self, skew: float = 0, kurt: float = 3):
        """
        Run the full Monte Carlo simulation pipeline:
            For multiple variables:
                1. Covariance matrix calculation
                2. Cholesky decomposition
                3. Raw normal simulation
                4. Apply Cholesky to induce correlation
                5. Add means
            For a single variable:
                1. Raw normal simulation
                2. Scale by std and shift by mean
        Args:
            skew (float): Desired skewness of simulated returns (default: 0).
            kurt (float): Desired kurtosis of simulated returns (default: 3).
        """

        simulator = RandomSimulator(num_simulations=self.num_simulations)

        if self.num_variables == 1:
            # Univariate simulation
            variable = self.variable_names[0]
            std = self.data[variable].std()
            mean = self.data[variable].mean()

            self.simulated_raw = simulator.simulate_normal(
                target_skew=skew, target_kurt=kurt, num_variables=1
            )

            self.simulated_raw.columns = [variable]

            self.simulated_final = self.simulated_raw * std + mean

            # No covariance or cholesky matrix
            self.cov_matrix = None
            self.cholesky_matrix = None

        else:
            # Multivariate simulation
            # Compute covariance matrix
            cov_calculator = CovarianceMatrix(self.data)
            self.cov_matrix = cov_calculator.get_matrix()

            # Cholesky decomposition
            self.cholesky_matrix = Decomposer.cholesky_decomposition(
                self.cov_matrix)

            # Generate uncorrelated standard normal values
            simulator = RandomSimulator(num_simulations=self.num_simulations)
            self.simulated_raw = simulator.simulate_normal(
                target_skew=skew, target_kurt=kurt, num_variables=self.num_variables)
            self.simulated_raw.columns = self.variable_names

            # Apply Cholesky to correlate
            correlated = self.simulated_raw @ self.cholesky_matrix.T

            # Add means
            means = self.data.mean().values  # shape: (num_variables,)
            self.simulated_final = correlated + means

    def get_final_simulated_values(self) -> pd.DataFrame:
        """
        Get the correlated simulations.

        Returns:
            pd.DataFrame: Correlated simulated variables.

        Raises:
            RuntimeError: If the simulation has not yet been executed.
        """
        if self.simulated_final is None:
            raise RuntimeError("Simulation has not been run yet.")
        return self.simulated_final

    def get_cholesky_matrix(self) -> pd.DataFrame:
        """
        Get the Cholesky matrix used in simulation.

        Returns:
            pd.DataFrame: Lower-triangular Cholesky matrix.
        """
        if self.cholesky_matrix is None:
            raise RuntimeError("Simulation has not been run yet.")
        return self.cholesky_matrix

    def get_raw_simulations(self) -> pd.DataFrame:
        """
        Get the raw uncorrelated standard normal simulations.

        Returns:
            pd.DataFrame: Uncorrelated normal simulations.
        """
        if self.simulated_raw is None:
            raise RuntimeError("Simulation has not been run yet.")
        return self.simulated_raw

    def get_covariance_matrix(self) -> pd.DataFrame:
        """
        Get the covariance matrix computed from input data.

        Returns:
            pd.DataFrame: Covariance matrix of the original data.

        Raises:
            RuntimeError: If the simulation has not been run yet.
        """
        if self.cov_matrix is None:
            raise RuntimeError("Simulation has not been run yet.")
        return self.cov_matrix
