# daanish/utils/modelling/classification/base_classification_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils.modelling.classification.scoring import SCORER_FUNCTIONS, get_scorer
from utils.importance.permutation_importance import PermutationImportance
from utils.preprocessing.balancing import ImbalanceHandler
from utils.preprocessing.scaler import Scaler


class BaseClassificationModel:
    """
    Base class for binary classification models with common functionality
    for training, evaluation, prediction, and validation.

    This base class is intended to be inherited by model-specific implementations
    such as LogisticModel and RandomForestModel.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and the target.

    features : list of str
        List of feature column names to be used in the model.

    target : str
        Target column name (must be binary: 0 or 1).

    test_size : float, default=0.2
        Proportion of data to reserve for testing.

    eval_size : float, default=0
        Proportion of data to reserve for evaluation (optional).

    X_eval : pd.DataFrame, optional
        External evaluation features. Used only if eval_size=0.

    y_eval : pd.Series, optional
        External evaluation target labels. Used only if eval_size=0.

    scoring_name : str, default='roc_auc'
        The performance metric to use (must exist in SCORER_FUNCTIONS).

    balance_method : str or None, default=None
        Optional imbalance handling method: 'undersample', 'oversample', 'smote', etc.

    id_column : str or None, default=None
        Optional column to track individual records in predictions.

    random_state : int, default=42
        Random seed for reproducibility.

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

    def __init__(
        self,
        df,
        features,
        target,
        id_column=None,
        test_size=0.2,
        eval_size=0.0,
        X_eval=None,
        y_eval=None,
        scoring="roc_auc",
        balance_method=None,
        random_state=42,
        scale=False,
        scaling_method='zscore',
        handle_skew=True,
        skew_method='log',
        skew_threshold=1.0
    ):
        self.df = df
        self.features = features
        self.target = target
        self.test_size = test_size
        self.eval_size = eval_size
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.balance_method = balance_method
        self.id_column = id_column
        self.random_state = random_state

        self.scoring_name = scoring
        self.scorer = SCORER_FUNCTIONS.get(scoring)
        if self.scorer is None:
            raise ValueError(f"Scoring method '{scoring}' is not supported.")

        self.scale = scale
        self.scaling_method = scaling_method
        self.handle_skew = handle_skew
        self.skew_method = skew_method
        self.skew_threshold = skew_threshold
        self.scaler = None  # Initialized if scaling is enabled

        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

        if self.eval_size == 0 and self.has_eval_data():
            self._validate_external_eval_data()

        self._prepare_data()

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
        """Validates that external evaluation data matches training feature structure and types."""
        if self.X_eval is None or self.y_eval is None:
            raise ValueError(
                "Both X_eval and y_eval must be provided together.")

        if list(self.X_eval.columns) != self.features:
            raise ValueError(
                "Feature columns in X_eval must match training features exactly.")

        non_numeric = self.X_eval.select_dtypes(
            exclude=["number", "bool"]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"The following columns in X_eval are not numeric or boolean: {non_numeric}."
            )

        if len(self.X_eval) != len(self.y_eval):
            raise ValueError(
                "X_eval and y_eval must have the same number of rows.")

        if not set(self.y_eval.unique()) <= {0, 1, True, False}:
            raise ValueError("y_eval must be binary (0/1 or True/False).")

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

        non_numeric = X.select_dtypes(
            exclude=["number", "bool"]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"The following features are not numeric or boolean: {non_numeric}. "
                f"Please encode categorical features before passing them to the model."
            )

        if self.eval_size > 0:
            # Split into train and temp (test+eval)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=self.test_size + self.eval_size,
                stratify=y,
                random_state=self.random_state
            )

            # Split temp into test and eval
            test_ratio = self.test_size / (self.test_size + self.eval_size)
            X_test, X_eval, y_test, y_eval = train_test_split(
                X_temp, y_temp,
                test_size=(1 - test_ratio),
                stratify=y_temp,
                random_state=self.random_state
            )

            self.X_train, self.y_train = X_train, y_train
            self.X_test, self.y_test = X_test, y_test
            self.X_eval, self.y_eval = X_eval, y_eval

        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state
            )

        # Check imbalance after split
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
            "PD": probs,
            "Prediction": preds
        })

        if include_id and self.id_column:
            ids = self.df.loc[self.X_test.index, self.id_column]
            results[self.id_column] = ids.values

        return results

    def predict_eval(self, include_id=False):
        """
        Generate predictions and predicted probabilities on the evaluation dataset.

        This method returns both the predicted class labels and the probability of of class 1 (e.g. default)
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
            "PD": probs,
            "Prediction": preds
        })

        if include_id and self.id_column:
            results[self.id_column] = self.df.loc[self.X_eval.index,
                                                  self.id_column].values

        return results

    def evaluate_model(self, X_test, y_test, y_proba=None):
        """
        Evaluate the model using the selected scoring metric.

        This method calculates the performance score using the configured metric,
        such as accuracy, ROC AUC, F1-score, recall, etc. If the metric requires
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

        # Use y_proba if metric requires probability
        y_pred = self.model.predict(X_test)

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
        test model generalization across datasets. It raises an error if no evaluation dataset
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

    def get_permutation_importance(self, dataset='test', scoring='roc_auc', n_repeats=10, random_state=None):
        """
        Compute feature importance using permutation importance on the test set.

        Permutation importance measures the decrease in model performance 
        when the values of a feature are randomly shuffled. It captures 
        the dependency of the model on each feature and is model-agnostic.

        Returns
        -------
        pd.DataFrame
            A DataFrame sorted by importance, containing:
            - 'Feature': Feature names.
            - 'Importance_Mean': Mean drop in performance (e.g., ROC AUC) when permuted.
            - 'Importance_Std': Standard deviation across permutation rounds.
            - 'Rank': Importance ranking (1 = most important).
        """
        if dataset == 'test':
            X, y = self.X_test, self.y_test
        elif dataset == 'eval':
            if not self.has_eval_data():
                raise ValueError("No evaluation data available.")
            X, y = self.X_eval, self.y_eval
        else:
            raise ValueError("dataset must be either 'test' or 'eval'")

        perm = PermutationImportance(
            model=self.model,
            X=X,
            y=y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state or self.random_state
        )
        return perm.compute()

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
