# utils/importance/permutation_importance.py

from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
import pandas as pd


class PermutationImportance:
    """
    Compute feature importance using permutation method.

    Parameters
    ----------
    model : fitted sklearn-compatible model
        The trained model to explain.

    X : pd.DataFrame or np.ndarray
        Feature set used for computing importance.

    y : pd.Series or np.ndarray
        True target values.

    scoring : str, default='roc_auc'
        Scoring metric to evaluate the importance drop. Can be any valid sklearn scorer.

    n_repeats : int, default=10
        Number of times to permute a feature.

    random_state : int, default=42
        Seed for reproducibility.

    Attributes
    ----------
    result_ : sklearn.utils.Bunch
        Raw output from sklearn's permutation_importance.

    importance_df_ : pd.DataFrame
        A DataFrame summarizing feature importance, standard deviation, and rank.
    """

    def __init__(self, model, X, y, scoring='roc_auc', n_repeats=10, random_state=42):
        self.model = model
        self.X = X
        self.y = y
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.result_ = None
        self.importance_df_ = None

    def compute(self):
        """
        Computes permutation importance for each feature.

        Returns
        -------
        pd.DataFrame
            A sorted DataFrame with: 'Feature', 'Importance_Mean', 'Importance_Std', 'Rank'.
        """
        check_is_fitted(self.model)
        result = permutation_importance(
            self.model, self.X, self.y,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        self.result_ = result
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance_Mean': result.importances_mean,
            'Importance_Std': result.importances_std
        }).sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)
        importance_df['Rank'] = importance_df['Importance_Mean'].rank(
            method='min', ascending=False).astype(int)
        self.importance_df_ = importance_df
        return importance_df
