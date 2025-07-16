# utils/importance/shap_explainer.py

import shap
import numpy as np
import pandas as pd


class SHAPExplainer:
    """
    Compute SHAP values using TreeExplainer (for tree-based models).

    Parameters
    ----------
    model : fitted tree-based model
        Must be compatible with SHAP TreeExplainer (e.g., RandomForest, XGBoost, LightGBM).

    X_background : pd.DataFrame or np.ndarray, optional
        Background dataset used for SHAP initialization.

    Attributes
    ----------
    explainer_ : shap.Explainer
        Fitted SHAP explainer.

    shap_values_ : shap.Explanation
        SHAP values after calling `compute()`.
    """

    def __init__(self, model, X_background=None):
        self.model = model
        self.X_background = X_background
        self.explainer_ = None
        self.shap_values_ = None

    def compute(self, X_to_explain, max_display=30):
        """
        Compute SHAP values and return a summary DataFrame of mean absolute SHAP values.

        Parameters
        ----------
        X_to_explain : pd.DataFrame or np.ndarray
            Dataset for which to compute SHAP values.

        max_display : int, default=20
            Number of top features to return based on mean absolute SHAP values.

        Returns
        -------
        pd.DataFrame
            A DataFrame with:
            - 'Feature': Feature names
            - 'Mean_ABS_SHAP_Value': Average absolute SHAP value across all rows
        """

        self.explainer_ = shap.TreeExplainer(self.model)

        self.shap_values_ = self.explainer_(
            X_to_explain, check_additivity=False)

        # SHAP values: could be Explanation or ndarray
        shap_vals_abs = np.abs(self.shap_values_.values)

        # Case 1: 2D (n_samples, n_features) → average over samples
        if shap_vals_abs.ndim == 2:
            mean_abs_vals = shap_vals_abs.mean(axis=0)

        # Case 2: 3D (n_samples, n_features, n_outputs) → average over both
        elif shap_vals_abs.ndim == 3:
            mean_abs_vals = shap_vals_abs.mean(axis=(0, 2))

        else:
            raise ValueError(
                f"Unexpected SHAP value shape: {shap_vals_abs.shape}")

        # Validate length match
        feature_names = list(X_to_explain.columns)
        if len(mean_abs_vals) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: {len(feature_names)} features vs {len(mean_abs_vals)} SHAP values")

        # Create summary DataFrame
        summary_df = (
            pd.DataFrame({
                'Feature': feature_names,
                'Mean_ABS_SHAP_Value': mean_abs_vals
            })
            .sort_values(by='Mean_ABS_SHAP_Value', ascending=False)
            .head(max_display)
            .reset_index(drop=True)
        )

        return summary_df

    # def summary_plot(self, X_to_explain, plot_type='bar'):
    #     """
    #     Display a summary plot of SHAP values.

    #     Parameters
    #     ----------
    #     X_to_explain : pd.DataFrame or np.ndarray
    #         Same dataset used in `compute()`.

    #     plot_type : str, default='bar'
    #         Type of summary plot to show ('bar' for global importance, 'dot' for distribution).

    #     Returns
    #     -------
    #     None
    #     """
    #     if self.shap_values_ is None:
    #         self.compute(X_to_explain)
    #     shap.summary_plot(self.shap_values_.values, X_to_explain, plot_type=plot_type)
