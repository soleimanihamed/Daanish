# daanish/utils/modelling/classification/logistic_regression.py


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils.modelling.classification.base_classification_model import BaseClassificationModel
from utils.preprocessing.balancing import ImbalanceHandler
import pandas as pd


class LogisticModel(BaseClassificationModel):
    """
    A general-purpose logistic regression modeling class for binary classification tasks,
    supporting class balancing, hyperparameter tuning, feature importance extraction,
    and evaluation on both internal and external datasets.

    This class enables flexible logistic regression modeling by allowing:
    - Selection of input features and target column
    - Optional handling of class imbalance through under/over sampling or SMOTE
    - Optional hyperparameter tuning via GridSearchCV
    - Integration with a separate evaluation dataset or portion of current dataset for model generalization checks
    - Interpretation of feature importance through model coefficients
    - Evaluation using a wide range of classification metrics

    Parameters
    ----------
    df : pd.DataFrame
        The primary dataset containing both features and the target variable.

    features : list of str
        The list of feature column names to be used for training the model.

    target : str
        Name of the target column (must be binary classification).

    id_column : str, optional (default=None)
        Optional column containing unique identifiers for each record. 
        This column will be excluded from model training but can be retained in outputs.

    test_size : float, optional (default=0.2)
        The proportion of the dataset to allocate as the test set during train/test split.

    eval_size : float, default=0.0
        Proportion of the dataset to include in the evaluation split. If > 0, dataset will be split into
        train, test, and eval sets internally.

    random_state : int, optional (default=42)
        Random seed used for reproducibility in train/test split and sampling methods.

    balance_method : str, optional (default='none')
        Specifies the method for handling class imbalance:
        - 'none': No balancing applied
        - 'undersample': Use RandomUnderSampler to reduce majority class
        - 'oversample': Use RandomUnderSampler with replacement to boost minority class
        - 'smote': Use SMOTE to synthetically create samples for the minority class

    tune_hyperparameters : bool, optional (default=False)
        If True, performs hyperparameter tuning using GridSearchCV.

    scoring : str, optional (default='roc_auc')
        The performance metric used for evaluation and hyperparameter tuning.
        Must be one of the predefined keys in `SCORER_FUNCTIONS`, such as:
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', etc.

    solver : str, default='liblinear'
        Algorithm to use in the optimization problem. Permissible values are:
        'liblinear', 'lbfgs', 'newton-cg', 'sag', and 'saga'.

    X_eval : pd.DataFrame, optional (default=None)
        Optional external evaluation feature set for assessing model generalization.

    y_eval : pd.Series or np.ndarray, optional (default=None)
        Corresponding labels for `X_eval` if provided.

    scale : bool, default=False
        Whether to scale numeric features (excluding one-hot and WOE columns).

    scaling_method : str, default='zscore'
        Scaling method to apply. Options: 'zscore', 'minmax'.

    handle_skew : bool, default=True
        Whether to reduce skewness in numeric features.

    skew_method : str, default='log'
        Transformation method for skewed variables. Options: 'log', 'sqrt'.

    skew_threshold : float, default=1.0
        Threshold of absolute skewness above which skew correction is applied.

    Raises
    ------
    ValueError
        If the specified scoring metric is not supported.
    """

    def __init__(self, tune_hyperparameters=False, solver: str = 'liblinear', **kwargs):
        super().__init__(**kwargs)
        self._validate_solver(solver)
        self.tune_hyperparameters = tune_hyperparameters
        self.solver = solver

    def _get_pipeline(self):
        """
        Constructs a pipeline using the configured balancing method and logistic regression.

        Returns
        -------
        imblearn.pipeline.Pipeline
            Pipeline with optional balancing and a logistic regression estimator.
            If hyperparameter tuning is enabled, the pipeline exposes `clf__C` for GridSearchCV.
        """

        clf = LogisticRegression(
            solver=self.solver, random_state=self.random_state)
        imbalance_handler = ImbalanceHandler(
            balance_method=self.balance_method, random_state=self.random_state)
        return imbalance_handler.build_pipeline(clf)

    def fit_model(self):
        """
        Fit the logistic regression model to the training data.

        If `tune_hyperparameters` is True, the method performs a grid search 
        over predefined values of the regularization parameter C using cross-validation 
        to select the best model. Otherwise, it fits a logistic regression model 
        directly using the training data.

        Returns
        -------
        None
        """
        pipeline = self._get_pipeline()

        if self.tune_hyperparameters:
            param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
            grid = GridSearchCV(pipeline, param_grid,
                                cv=5, scoring=self.scoring_name)
            grid.fit(self.X_train, self.y_train)
            self.model = grid.best_estimator_
        else:
            pipeline.fit(self.X_train, self.y_train)
            self.model = pipeline

    def get_coefficients(self):
        """
        Retrieve the coefficients of the trained logistic regression model.

        Useful for interpreting feature importance. The output includes the raw 
        coefficients and their absolute values, sorted by magnitude in descending order.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing:
            - 'Feature': Feature names used in the model.
            - 'Coefficient': Raw coefficient values from logistic regression.
            - 'Abs_Coefficient': Absolute values of the coefficients, for ranking importance.
        """
        clf = self.model.named_steps['clf']
        coef = clf.coef_[0]
        return pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': coef,
            'Abs_Coefficient': abs(coef)
        }).sort_values(by='Abs_Coefficient', ascending=False).reset_index(drop=True)

    def _validate_solver(self, solver):
        """
        Internal method to validate the specified solver.

        Parameters
        ----------
        solver : str
            The solver name to validate.

        Raises
        ------
        ValueError
            If the solver is not supported.
        """
        valid_solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
        if solver not in valid_solvers:
            raise ValueError(
                f"Invalid solver: '{solver}'. Must be one of: {valid_solvers}."
            )
