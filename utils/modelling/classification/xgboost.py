# daanish/utils/modelling/classification/xgboost.py

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from utils.preprocessing.balancing import ImbalanceHandler
from utils.importance.shap_explainer import SHAPExplainer
from utils.modelling.classification.base_classification_model import BaseClassificationModel


class XGBoostModel(BaseClassificationModel):
    """
    General-purpose XGBoost classification class supporting training, evaluation, balancing,
    and feature importance via gain, permutation, and SHAP values.

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
    n_estimators : int, default=100
    max_depth : int, default=6
    learning_rate : float, default=0.1
    subsample : float, default=1.0
    colsample_bytree : float, default=1.0
    eval_metric : 'logloss'
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 tune_hyperparameters=False,
                 eval_metric='logloss',
                 **kwargs):

        kwargs['scale'] = False  # XGBoost handles its own internal scaling
        super().__init__(**kwargs)

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tune_hyperparameters = tune_hyperparameters

        self._validate_eval_metric(eval_metric)
        self.eval_metric = eval_metric

    def _get_pipeline(self):
        """
        Constructs an imbalanced-learn pipeline that includes optional resampling 
        and an XGBoost classifier with specified hyperparameters.

        This method:
        - Initializes an XGBoost classifier (`XGBClassifier`) with the provided model parameters.
        - Integrates the classifier into a pipeline with a resampling strategy (e.g., SMOTE, undersampling),
        using the configured `balance_method`.

        Returns
        -------
        imblearn.pipeline.Pipeline
            A pipeline that includes the resampling step (if applicable) followed by the XGBoost classifier.
        """

        clf = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            use_label_encoder=False,
            eval_metric=self.eval_metric,
            random_state=self.random_state
        )

        imbalance_handler = ImbalanceHandler(
            balance_method=self.balance_method, random_state=self.random_state)
        return imbalance_handler.build_pipeline(clf)

    def fit_model(self):
        """
        Trains the XGBoost classifier using the training data.

        If `tune_hyperparameters` is True, it performs a grid search over a set of
        XGBoost hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`) 
        using cross-validation to select the best configuration based on the provided scoring metric.

        Otherwise, the model is trained directly with the current parameters.

        After training, the final model is stored in `self.model`.

        Returns
        -------
        None
            The trained pipeline is saved to `self.model`.
        """

        pipeline = self._get_pipeline()

        if self.tune_hyperparameters:
            param_grid = {
                'clf__n_estimators': [100, 200],
                'clf__max_depth': [4, 6, 10],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0]
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

    def get_feature_importance_gain(self):
        """
        Returns XGBoost gain-based feature importances.

        Returns
        -------
        pd.DataFrame
            DataFrame sorted by gain importance.
        """
        clf = self.model.named_steps['clf']
        booster = clf.get_booster()
        importance_dict = booster.get_score(importance_type='gain')

        # Map feature names
        importance_df = pd.DataFrame([
            {'Feature': f, 'Gain_Importance': imp}
            for f, imp in importance_dict.items()
        ])

        importance_df['Normalized_Importance'] = importance_df['Gain_Importance'] / \
            importance_df['Gain_Importance'].sum()
        return importance_df.sort_values(by='Gain_Importance', ascending=False).reset_index(drop=True)

    def get_feature_importance_shap(self, max_display=20):
        """
        Compute SHAP values for global feature importance (mean absolute SHAP value per feature).

        Parameters
        ----------
        max_display : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            Sorted dataframe of SHAP importance values.
        """
        shap_calc = SHAPExplainer(
            model=self.model.named_steps['clf'],
            X_background=self.X_test
        )
        return shap_calc.compute(X_to_explain=self.X_test, max_display=max_display)

    def _validate_eval_metric(self, eval_metric):
        """
        Validates the provided eval_metric against supported XGBoost metrics for
        binary classification, multiclass classification, and regression tasks.

        Raises:
            ValueError: If eval_metric is not supported.
        """
        supported_metrics = {
            'binary': ['logloss', 'error', 'auc', 'aucpr', 'map'],
            'multiclass': ['mlogloss', 'merror'],
            'regression': ['rmse', 'mae', 'rmsle', 'mape']
        }

        # Flatten all allowed metrics into a single list
        allowed = sum(supported_metrics.values(), [])

        if eval_metric not in allowed:
            raise ValueError(
                f"Invalid eval_metric '{eval_metric}'. Must be one of: {allowed}"
            )
