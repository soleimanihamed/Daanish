# daanish/utils/modelling/classification/random_forest.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from utils.preprocessing.balancing import ImbalanceHandler
from utils.importance.shap_explainer import SHAPExplainer
from utils.modelling.classification.base_classification_model import BaseClassificationModel


class RandomForestModel (BaseClassificationModel):
    """
    A general-purpose Random Forest classification class that supports full training, 
    evaluation, class balancing, feature importance, SHAP/permutation explainability, and more.

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

    X_eval : pd.DataFrame, optional (default=None)
        Optional external evaluation feature set for assessing model generalization.

    y_eval : pd.Series or np.ndarray, optional (default=None)
        Corresponding labels for `X_eval` if provided.

    model parameters:
    -----------------
    n_estimators : int, default=100
        Number of trees in the forest.

    max_depth : int or None
        Maximum tree depth.

    max_features : str, default='sqrt'
        Number of features to consider at each split.

    criterion : str, default='gini'
        Function to measure quality of a split. 'gini' or 'entropy'.
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 max_features='sqrt',
                 criterion='gini',
                 tune_hyperparameters=False,
                 **kwargs):

        # Force scale to False since Random Forest does not need scaling
        kwargs['scale'] = False

        super().__init__(**kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.criterion = criterion
        self.tune_hyperparameters = tune_hyperparameters

    def _get_pipeline(self):
        """
        Constructs a pipeline using the configured balancing method and random forest classifier.

        This method:
        - Initializes a RandomForestClassifier with the specified hyperparameters.
        - Wraps the classifier in an imbalanced-learn pipeline that applies resampling if a balance_method is defined.

        Returns
        -------
        imblearn.pipeline.Pipeline
            A pipeline consisting of optional resampling and the random forest classifier.
        """

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            criterion=self.criterion,
            random_state=self.random_state
        )

        imbalance_handler = ImbalanceHandler(
            balance_method=self.balance_method, random_state=self.random_state)
        return imbalance_handler.build_pipeline(clf)

    def fit_model(self):
        """
        Fit the random forest classifier to the training data.

        If `tune_hyperparameters` is True, the method performs a grid search 
        over predefined values of hyperparameters such as number of estimators and maximum depth 
        using cross-validation to select the best model. Otherwise, it fits a random forest model 
        directly using the training data.

        Returns
        -------
        None
            The trained model is stored in self.model.
        """

        pipeline = self._get_pipeline()

        if self.tune_hyperparameters:
            param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [None, 10, 20],
                'clf__max_features': ['sqrt', 'log2']
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
        Retrieve Gini-based feature importances from the trained random forest model.

        Gini importance (also known as mean decrease in impurity) measures how much 
        each feature contributes to reducing impurity across all trees in the forest. 
        It is useful for quick insights but can be biased toward high-cardinality features.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing:
            - 'Feature': Feature names used in the model.
            - 'Gini_Importance': Raw importance values (mean decrease in impurity).
            - 'Normalized_Importance': Importances scaled to sum to 1, for comparison.
        """

        clf = self.model.named_steps['clf']
        importances = clf.feature_importances_
        normalized = importances / importances.sum()

        return pd.DataFrame({
            'Feature': self.X_train.columns,
            'Gini_Importance': importances,
            'Normalized_Importance': normalized
        }).sort_values(by='Gini_Importance', ascending=False).reset_index(drop=True)

    def get_feature_importance_shap(self, max_display=20):
        """
        Compute SHAP (SHapley Additive exPlanations) values for global feature importance.

        SHAP provides a unified measure of feature impact by attributing each prediction 
        to feature contributions, based on cooperative game theory. Unlike permutation 
        or Gini, SHAP explains individual predictions and aggregates them for global importance.

        Parameters
        ----------
        max_display : int, optional (default=20)
            Maximum number of top features to display in the output.

        Returns
        -------
        pd.DataFrame
            A DataFrame sorted by SHAP importance, containing:
            - 'Feature': Feature name.
            - 'Mean_ABS_SHAP_Value': Average absolute SHAP value across all samples.
        """
        shap_calc = SHAPExplainer(
            model=self.model.named_steps['clf'],
            X_background=self.X_test
        )
        return shap_calc.compute(X_to_explain=self.X_test, max_display=max_display)
