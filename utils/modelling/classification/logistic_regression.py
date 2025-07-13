# daanish/utils/modelling/classification/logistic_regression.py


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report, accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, matthews_corrcoef, balanced_accuracy_score,
                             cohen_kappa_score, brier_score_loss, confusion_matrix
                             )
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
from utils.preprocessing.scaler import Scaler
from utils.preprocessing.balancing import ImbalanceHandler

SCORER_FUNCTIONS = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'f1': f1_score,
    'roc_auc': roc_auc_score,
    'log_loss': log_loss,
    'matthews_corrcoef': matthews_corrcoef,
    'balanced_accuracy': balanced_accuracy_score,
    'cohen_kappa': cohen_kappa_score,
    'brier_score': brier_score_loss
}


class LogisticModel:
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

    def __init__(self, df, features, target, id_column=None, test_size=0.2, eval_size=0.0,
                 random_state=42, balance_method='none', tune_hyperparameters=False,
                 scoring: str = 'roc_auc', X_eval=None, y_eval=None, solver: str = 'liblinear',
                 scale=False, scaling_method='zscore', handle_skew=True, skew_method='log',
                 skew_threshold=1.0):

        if scoring not in SCORER_FUNCTIONS:
            raise ValueError(f"Unsupported scoring metric: {scoring}")
        self.scoring_name = scoring
        self.scorer = SCORER_FUNCTIONS[scoring]

        self.df = df.copy()
        self.features = features
        self.target = target
        self.id_column = id_column
        self.test_size = test_size
        self.eval_size = eval_size
        self.random_state = random_state
        self.balance_method = balance_method
        self.tune_hyperparameters = tune_hyperparameters
        self.solver = solver

        self.X_eval = X_eval
        self.y_eval = y_eval

        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

        self.scale = scale
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold

        self.scaler = None  # Initialized later if scaling is True

        self._prepare_data()

        if self.eval_size == 0 and self.has_eval_data():
            self._validate_external_eval_data()

    def _prepare_data(self):
        """
        Internal method to split the dataset into train, test, and evaluation sets depending on eval_size.

        This method:
        - Selects the specified features and target from the input DataFrame.
        - Optionally removes the ID column (if provided) to avoid data leakage.
        - Performs a stratified train-test-evaluation split to preserve the class distribution.
            -- If eval_size > 0, evaluation set is internally derived.
            -- If eval_size == 0, X_eval and y_eval can be provided externally.

        The resulting subsets are stored in instance attributes:
        - self.X_train, self.X_test, self.X_eval : Feature sets for training, testing, and evaluation
        - self.y_train, self.y_test, self.y_eval : Corresponding target labels

        Returns
        -------
        None
            The split data is stored internally in the model instance.
        """

        X = self.df[self.features].copy()
        y = self.df[self.target]

        if self.id_column and self.id_column in X:
            X.drop(columns=[self.id_column], inplace=True)

        # Validation: Ensure all features are numeric or boolean
        non_numeric = X.select_dtypes(
            exclude=['number', 'bool']).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"The following features are not numeric or boolean: {non_numeric}. "
                f"Please encode categorical features before passing them to LogisticModel."
            )

        if self.eval_size > 0:
            # Split into train and temp (test+eval)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=self.test_size + self.eval_size,
                random_state=self.random_state,
                stratify=y
            )

            # Split temp into test and eval
            test_ratio = self.test_size / (self.test_size + self.eval_size)
            X_test, X_eval, y_test, y_eval = train_test_split(
                X_temp, y_temp,
                test_size=(1 - test_ratio),
                random_state=self.random_state,
                stratify=y_temp
            )

            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test
            self.X_eval, self.y_eval = X_eval, y_eval

        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
            # External X_eval and y_eval may still be used

        # 🔍 Check imbalance after split
        imbalance_handler = ImbalanceHandler(random_state=self.random_state)
        if imbalance_handler.is_imbalanced(self.y_train):
            print(
                "⚠️ Warning: Class imbalance detected in training set. Consider setting a balance_method.")

        if self.scale:
            self.scaler = Scaler(
                method=self.scaling_method,
                handle_skew=self.handle_skew,
                skew_method=self.skew_method,
                skew_threshold=self.skew_threshold
            )

            cols_to_scale = self._select_numeric_to_scale(self.df)

            self.X_train[cols_to_scale] = self.scaler.fit_transform(
                self.X_train, cols_to_scale)
            self.X_test[cols_to_scale] = self.scaler.transform(self.X_test)
            if self.X_eval is not None:
                self.X_eval[cols_to_scale] = self.scaler.transform(self.X_eval)

    def _get_pipeline(self):
        """
        Constructs a pipeline using the configured balancing method and logistic regression.

        Returns
        -------
        imblearn.pipeline.Pipeline
            A pipeline consisting of optional resampling and the logistic regression classifier.
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

    def predict_test(self, include_id=False):
        """
        Predicts probabilities and classes on the test set.

        Parameters
        ----------
        include_id : bool, default=False
            If True, returns ID column in the prediction output.

        Returns
        -------
        pd.DataFrame
            DataFrame with probability of default, predicted class, and optional ID.
        """
        probs = self.model.predict_proba(self.X_test)[:, 1]
        preds = self.model.predict(self.X_test)

        results = pd.DataFrame({
            'PD': probs,
            'Prediction': preds
        })

        if include_id and self.id_column:
            ids = self.df.loc[self.X_test.index, self.id_column]
            results[self.id_column] = ids.values

        return results

    def predict_eval(self, include_id=False):
        """
        Generate predictions and predicted probabilities on the evaluation dataset.

        This method returns both the predicted class labels and the probability of default (PD)
        for each record in the evaluation set. If an ID column was specified during initialization
        and `include_id=True`, it will be included in the result for traceability.

        Parameters
        ----------
        include_id : bool, default=False
            If True, includes the record identifier (from the `id_column`) in the returned DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the predicted probabilities ('PD'), predicted classes ('Prediction'),
            and optionally the ID column.

        Raises
        ------
        ValueError
            If no evaluation dataset was provided.
        """
        if not self.has_eval_data():
            raise ValueError("No evaluation dataset was provided.")

        probs = self.model.predict_proba(self.X_eval)[:, 1]
        preds = self.model.predict(self.X_eval)

        results = pd.DataFrame({
            'PD': probs,
            'Prediction': preds
        })

        if include_id and self.id_column:
            results[self.id_column] = self.df.loc[self.X_eval.index,
                                                  self.id_column].values

        return results

    def evaluate_model(self, X_test, y_test, y_proba=None):
        """
        Evaluate the model using the selected scoring metric.

        This method calculates the performance score using the configured metric,
        such as accuracy, ROC AUC, precision, recall, etc. If the metric requires
        predicted probabilities (e.g., 'roc_auc', 'log_loss', 'brier_score'), it will
        use the provided `y_proba`, or compute it internally if not provided.

        Parameters
        ----------
        X_test : pd.DataFrame or np.ndarray
            Test feature data.
        y_test : pd.Series or np.ndarray
            True target values for the test set.
        y_proba : array-like, optional
            Predicted probabilities for the positive class. Required only for probability-based metrics.

        Returns
        -------
        float
            The model performance score based on the specified scoring metric.
        """

        y_pred = self.model.predict(X_test)

        # Use y_proba if metric requires probability
        if self.scoring_name in ['roc_auc', 'log_loss', 'brier_score']:
            if y_proba is None:
                y_proba = self.model.predict_proba(X_test)[:, 1]
            score = self.scorer(y_test, y_proba)
        else:
            score = self.scorer(y_test, y_pred)

        return score

    def evaluate_test_model(self):
        """
        Evaluate the model performance on the test set using the selected scoring metric.

        This method is a convenience wrapper around `evaluate_model`, using the internal
        test data split created during initialization.

        Returns
        -------
        float
            Performance score of the model on the test set.
        """
        return self.evaluate_model(self.X_test, self.y_test)

    def evaluate_eval_model(self):
        """
        Evaluate the model performance on an optional external evaluation dataset.

        This method is useful when a separate hold-out or future dataset is used to
        test model generalizability. It raises an error if no evaluation dataset
        was provided.

        Returns
        -------
        float
            Performance score of the model on the evaluation dataset.

        Raises
        ------
        ValueError
            If no evaluation dataset was provided.
        """
        if not self.has_eval_data():
            raise ValueError("No evaluation dataset was provided.")
        return self.evaluate_model(self.X_eval, self.y_eval)

    def evaluate_all_metrics(self, X, y):
        """
        Evaluate the model using a comprehensive set of classification metrics.

        Computes a wide range of standard classification metrics such as accuracy,
        precision, recall, F1 score, ROC AUC, log loss, Brier score, and more.
        Metrics that require probability estimates (like ROC AUC or log loss) will
        automatically use predicted probabilities.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data to evaluate.
        y : pd.Series or np.ndarray
            True target values corresponding to X.

        Returns
        -------
        dict
            Dictionary containing metric names as keys and their computed values.
            Example:
            {
                'accuracy': 0.91,
                'precision': 0.87,
                'roc_auc': 0.94,
                ...
            }
        """

        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        return {
            name: func(y, y_proba if name in [
                       'roc_auc', 'log_loss', 'brier_score'] else y_pred)
            for name, func in SCORER_FUNCTIONS.items()
        }

    def evaluate_all_test_metrics(self):
        """
        Evaluate the model on the internal test dataset using all supported metrics.

        This method wraps `evaluate_all_metrics` with the test data held internally
        by the class after splitting.

        Returns
        -------
        dict
            Dictionary of metric names and scores on the test dataset.
        """
        return self.evaluate_all_metrics(self.X_test, self.y_test)

    def evaluate_all_eval_metrics(self):
        """
        Evaluate the model on the optional external evaluation dataset using all supported metrics.

        This method uses the evaluation dataset provided by the user during initialization
        or via method input. If no evaluation data is available, it raises an error.

        Returns
        -------
        dict
            Dictionary of metric names and scores on the evaluation dataset.

        Raises
        ------
        ValueError
            If no evaluation dataset was provided.
        """
        if not self.has_eval_data():
            raise ValueError("No evaluation dataset was provided.")
        return self.evaluate_all_metrics(self.X_eval, self.y_eval)

    def get_classification_report(self, X, y, output_dict=True):
        """
        Generate a classification report for a given dataset.

        This report includes precision, recall, F1-score, and support for each class.
        It is useful for understanding the model's performance on both individual
        classes and overall. It can return the report either as a formatted string
        or a dictionary.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data to generate predictions from.
        y : pd.Series or np.ndarray
            True target labels corresponding to X.
        output_dict : bool, default=True
            If True, returns the report as a dictionary (useful for logging or evaluation pipelines).
            If False, returns a formatted string suitable for printing.

        Returns
        -------
        dict or str
            Classification report with precision, recall, F1-score, and support for each class.
        """

        y_pred = self.model.predict(X)
        return classification_report(y, y_pred, output_dict=output_dict)

    def get_test_classification_report(self, output_dict=True):
        """
        Generate the classification report for the internal test set.

        Parameters
        ----------
        output_dict : bool, default=True
            If True, returns the report as a dictionary. If False, returns a string.

        Returns
        -------
        dict or str
            Classification report for the test set.
        """
        return self.get_classification_report(self.X_test, self.y_test, output_dict)

    def get_eval_classification_report(self, output_dict=True):
        """
        Generate the classification report for the external evaluation set.

        Raises an error if no evaluation data has been provided.

        Parameters
        ----------
        output_dict : bool, default=True
            If True, returns the report as a dictionary. If False, returns a string.

        Returns
        -------
        dict or str
            Classification report for the evaluation set.

        Raises
        ------
        ValueError
            If evaluation data is not available.
        """
        if not self.has_eval_data():
            raise ValueError("No evaluation dataset was provided.")
        return self.get_classification_report(self.X_eval, self.y_eval, output_dict)

    def get_confusion_matrix(self, X, y, normalize=None):
        """
        Compute and return the confusion matrix for given data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature data to make predictions on.
        y : pd.Series or np.ndarray
            True target values corresponding to X.
        normalize : {'true', 'pred', 'all'}, default=None
            If specified, normalizes confusion matrix over the true labels (rows), predicted labels (columns),
            or the entire matrix:
                - 'true'  : Normalize by the number of actual samples per class (row-wise).
                - 'pred'  : Normalize by the number of predicted samples per class (column-wise).
                - 'all'   : Normalize by total sample count.
                - None    : Return raw counts.

        Returns
        -------
        np.ndarray
            Confusion matrix as a 2D array.
        """
        y_pred = self.model.predict(X)
        return confusion_matrix(y, y_pred, normalize=normalize)

    def get_test_confusion_matrix(self, normalize=None):
        """
        Computes the confusion matrix on the test set used during internal model training.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            See `get_confusion_matrix` for normalization options.

        Returns
        -------
        np.ndarray
            Confusion matrix for the internal test set.
        """
        return self.get_confusion_matrix(self.X_test, self.y_test, normalize)

    def get_eval_confusion_matrix(self, normalize=None):
        """
        Computes the confusion matrix on an external evaluation set.

        Parameters
        ----------
        normalize : {'true', 'pred', 'all'}, default=None
            See `get_confusion_matrix` for normalization options.

        Returns
        -------
        np.ndarray
            Confusion matrix for the external evaluation set.

        Raises
        ------
        ValueError
            If no evaluation dataset was provided to the model.
        """
        if not self.has_eval_data():
            raise ValueError("No evaluation dataset was provided.")
        return self.get_confusion_matrix(self.X_eval, self.y_eval, normalize)

    def has_eval_data(self):
        """
        Check if evaluation dataset has been provided.

        Returns
        -------
        bool
            True if both X_eval and y_eval exist, False otherwise.
        """
        return self.X_eval is not None and self.y_eval is not None

    def _validate_external_eval_data(self):
        """
        Validates that external evaluation data matches training feature structure and types.
        """
        if self.X_eval is None or self.y_eval is None:
            raise ValueError(
                "Both X_eval and y_eval must be provided together.")

        # Check column names and order
        if list(self.X_eval.columns) != self.features:
            raise ValueError(
                "Feature columns in X_eval must match the training features exactly and in order.")

        # Check data types
        non_numeric = self.X_eval.select_dtypes(
            exclude=['number', 'bool']).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"The following columns in X_eval are not numeric or boolean: {non_numeric}. "
                f"Please ensure all categorical features are encoded."
            )

        # Check matching lengths
        if len(self.X_eval) != len(self.y_eval):
            raise ValueError(
                "X_eval and y_eval must have the same number of rows.")

        # Check binary target
        unique_vals = set(self.y_eval.unique())
        if not unique_vals <= {0, 1, True, False}:
            raise ValueError(
                f"y_eval contains non-binary values: {unique_vals}. It must be binary.")

    def _select_numeric_to_scale(self, df: pd.DataFrame) -> list:
        """
        Automatically selects numeric columns to scale, excluding binary features and WOE-transformed columns.

        Returns
        -------
        list
            List of column names to apply scaling and skew correction.
        """
        numeric_cols = df[self.features].select_dtypes(
            include='number').columns

        cols_to_scale = []
        for col in numeric_cols:
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):  # Binary / one-hot
                continue
            if col.lower().endswith('_woe'):
                continue
            cols_to_scale.append(col)

        return cols_to_scale
