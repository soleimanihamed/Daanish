# daanish/utils/modelling/classification/decision_tree.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from utils.preprocessing.balancing import ImbalanceHandler
from utils.importance.shap_explainer import SHAPExplainer
from utils.modelling.classification.base_classification_model import BaseClassificationModel


class DecisionTreeModel(BaseClassificationModel):
    """
    Decision Tree (CART) classification model with support for balancing, 
    SHAP and permutation importance, and optional hyperparameter tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target column.

    features : list
        List of feature column names to be used in training.

    target : str
        Name of the binary target variable.

    id_column : str, optional
        Optional ID column to preserve traceability in outputs.

    test_size : float, default=0.2
        Proportion of the dataset to allocate as test set.

    eval_size : float, default=0.0
        Proportion of the dataset to reserve for internal evaluation.

    random_state : int, default=42
        Seed for reproducibility.

    balance_method : str, optional
        One of {'none', 'undersample', 'oversample', 'smote'}.

    tune_hyperparameters : bool, default=False
        If True, tunes hyperparameters with GridSearchCV.

    scoring : str, default='roc_auc'
        Metric for tuning and evaluation.

    model parameters:
    -----------------
    max_depth : int, default=None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    criterion : str, default='gini' ('entropy' also supported)
    """

    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 criterion='gini',
                 tune_hyperparameters=False,
                 **kwargs):

        super().__init__(**kwargs)

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self._validate_criterion(criterion)
        self.criterion = criterion

        self.tune_hyperparameters = tune_hyperparameters

    def _get_pipeline(self):
        """
        Returns a pipeline that includes resampling (if any) and the DecisionTree classifier.
        """
        clf = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=self.random_state
        )

        imbalance_handler = ImbalanceHandler(
            balance_method=self.balance_method, random_state=self.random_state)
        return imbalance_handler.build_pipeline(clf)

    def fit_model(self):
        """
        Trains the Decision Tree classifier.

        If `tune_hyperparameters` is True, performs a grid search to optimize hyperparameters.
        """
        pipeline = self._get_pipeline()

        if self.tune_hyperparameters:
            param_grid = {
                'clf__max_depth': [3, 5, 10, None],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 5],
                'clf__criterion': ['gini', 'entropy', 'log_loss']
            }

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring=self.scoring_name,
                n_jobs=-1
            )
            grid.fit(self.X_train, self.y_train)
            self.model = grid.best_estimator_
        else:
            pipeline.fit(self.X_train, self.y_train)
            self.model = pipeline

    def get_feature_importance_gini(self):
        """
        Returns Decision Tree feature importances based on Mean Decrease in Impurity (Gini or Entropy).

        Returns
        -------
        pd.DataFrame
            DataFrame of features sorted by importance.
        """
        clf = self.model.named_steps['clf']
        importance = clf.feature_importances_

        importance_df = pd.DataFrame({
            'Feature': self.features,
            'Gini_Importance': importance
        })
        importance_df['Normalized_Importance'] = importance_df['Gini_Importance'] / \
            importance_df['Gini_Importance'].sum()
        return importance_df.sort_values(by='Gini_Importance', ascending=False).reset_index(drop=True)

    def get_feature_importance_shap(self, max_display=20):
        """
        Computes SHAP values using TreeExplainer.

        Parameters
        ----------
        max_display : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            SHAP-based feature importance dataframe.
        """
        shap_calc = SHAPExplainer(
            model=self.model.named_steps['clf'],
            X_background=self.X_test
        )
        return shap_calc.compute(X_to_explain=self.X_test, max_display=max_display)

    def _validate_criterion(self, criterion):
        allowed_criteria = ['gini', 'entropy', 'log_loss']
        if criterion not in allowed_criteria:
            raise ValueError(
                f"Invalid criterion '{criterion}'. Must be one of: {allowed_criteria}")
