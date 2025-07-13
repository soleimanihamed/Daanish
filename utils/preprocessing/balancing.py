# daanish/utils/preprocessing/balancing.py

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from collections import Counter


class ImbalanceHandler:
    """
    A reusable utility for detecting class imbalance and building resampling pipelines.

    Supports:
    - Undersampling (RandomUnderSampler)
    - Oversampling (RandomUnderSampler with replacement)
    - SMOTE (Synthetic Minority Over-sampling Technique)

    Parameters
    ----------
    balance_method : str
        One of 'none', 'undersample', 'oversample', 'smote'

    random_state : int
        Random seed for reproducibility

    Methods
    -------
    is_imbalanced(y, threshold=0.1)
        Returns True if imbalance ratio exceeds given threshold.

    build_pipeline(estimator)
        Returns an imblearn pipeline with resampling and given model.
    """

    def __init__(self, balance_method='none', random_state=42):
        self.balance_method = balance_method
        self.random_state = random_state

    def is_imbalanced(self, y, threshold=0.1) -> bool:
        """
        Checks if the dataset target variable is imbalanced.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            Target variable.

        threshold : float, default=0.1
            If the minority class is less than (1 - threshold), it's considered imbalanced.

        Returns
        -------
        bool
            True if class imbalance exists.
        """
        if isinstance(y, pd.Series):
            y = y.values
        counts = Counter(y)
        total = sum(counts.values())
        ratios = {cls: count / total for cls, count in counts.items()}
        minority_ratio = min(ratios.values())

        return minority_ratio < (1 - threshold)

    def build_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        """
        Constructs a resampling pipeline with the given estimator.

        Parameters
        ----------
        estimator : sklearn BaseEstimator
            The model to place at the end of the pipeline.

        Returns
        -------
        imblearn.pipeline.Pipeline
            Pipeline with optional resampling and the classifier.
        """
        steps = []

        if self.balance_method == 'undersample':
            steps.append(('under', RandomUnderSampler(
                random_state=self.random_state)))
        elif self.balance_method == 'oversample':
            steps.append(('over', RandomUnderSampler(
                sampling_strategy='majority', replacement=True)))
        elif self.balance_method == 'smote':
            steps.append(('smote', SMOTE(random_state=self.random_state)))

        steps.append(('clf', estimator))
        return Pipeline(steps=steps)
