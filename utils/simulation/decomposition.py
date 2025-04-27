# utils/simulation/decomposition

import numpy as np
import pandas as pd


class Decomposer:
    @staticmethod
    def cholesky_decomposition(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Cholesky decomposition on a given correlation matrix.

        Args:
            correlation_matrix (pd.DataFrame): A symmetric, positive definite matrix.

        Returns:
            pd.DataFrame: Lower triangular Cholesky matrix.
        """
        try:
            # Perform Cholesky decomposition
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)

            # Convert back to DataFrame
            cholesky_df = pd.DataFrame(
                cholesky_matrix,
                columns=correlation_matrix.columns,
                index=correlation_matrix.index
            )

            return cholesky_df

        except np.linalg.LinAlgError as e:
            raise ValueError(f"Matrix is not positive definite: {e}")
