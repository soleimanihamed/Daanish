# daanish/utils/simulation/transformation.py

from scipy.stats import norm, distributions
import pandas as pd


class DistributionTransformer:
    """
    Handles transformations between empirical data, CDF space,
    standard normal space, and back to original distributions.
    """

    @staticmethod
    def to_standard_normal(cdf_df: pd.DataFrame) -> pd.DataFrame:
        """Φ⁻¹(u): map CDF probabilities to standard normal Z values."""
        return pd.DataFrame(norm.ppf(cdf_df), columns=cdf_df.columns)

    @staticmethod
    def from_standard_normal_to_real(z_df: pd.DataFrame, fitted_params: dict) -> pd.DataFrame:
        """
        F_i⁻¹(Φ(z)): Convert simulated standard normal data back to
        original distributions using fitted parameters.
        """
        result = {}
        for col in z_df.columns:
            if col not in fitted_params:
                print(f"⚠️ Missing fitted parameters for {col}, skipping.")
                continue

            info = fitted_params[col]
            dist_name = info['distribution']
            params = {k: v for k, v in info.items() if k != 'distribution'}

            dist = getattr(distributions, dist_name, None)
            if dist is None:
                print(f"⚠️ Invalid distribution name: {dist_name}")
                continue

            u = norm.cdf(z_df[col])
            result[col] = dist.ppf(u, **params)

        return pd.DataFrame(result)
