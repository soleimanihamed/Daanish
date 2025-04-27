# utils/eda/correlation.py

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pointbiserialr, chi2_contingency
from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import warnings


class CorrelationAnalyzer:
    """
    A class to calculate and analyze correlations between different types of variables
    (numerical and categorical) within a pandas DataFrame.

    It provides methods to calculate pairwise correlations using various statistical
    measures and to identify pairs of variables with high correlation.
    """

    def __init__(self, data):
        """
        Initializes the CorrelationAnalyzer with the input DataFrame.

        Args:
            data (pd.DataFrame): The pandas DataFrame containing the data to analyze.
        """
        self.data = data

    def _cramers_v(self, x, y):
        """
        Calculates Cramer's V, a measure of association between two nominal
        categorical variables.

        Args:
            x (pd.Series): The first categorical variable.
            y (pd.Series): The second categorical variable.

        Returns:
            float: Cramer's V statistic, ranging from 0 (no association) to 1
                   (perfect association).
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

    def _correlation_ratio(self, categories, measurements):
        """
        Calculates the correlation ratio (Eta squared), a measure of the variance
        in the numerical variable that is explained by the categorical variable.

        Args:
            categories (pd.Series): The categorical variable.
            measurements (pd.Series): The numerical variable.

        Returns:
            float: The correlation ratio, ranging from 0 to 1.
        """
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0, cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.mean(cat_measures)
        y_total_avg = np.sum(y_avg_array*n_array)/np.sum(n_array)
        numerator = np.sum(n_array * (y_avg_array - y_total_avg)**2)
        denominator = np.sum((measurements - y_total_avg)**2)
        if denominator == 0:
            return 0.0
        return np.sqrt(numerator/denominator)

    def _calculate_numerical_numerical_correlation(self, numerical_cols, method="pearson"):
        """
        Calculates the correlation matrix for numerical columns using the specified method.

        Args:
            numerical_cols (list): A list of numerical column names.
            method (str, optional): The correlation method to use.
                Defaults to "pearson". Options are "pearson", "spearman", "kendall".

        Returns:
            pd.DataFrame: The correlation matrix for the numerical columns.

        Raises:
            ValueError: If an unsupported numerical correlation method is provided.
        """

        if method in ["pearson", "spearman", "kendall"]:
            return self.data[numerical_cols].corr(method=method)
        else:
            raise ValueError(
                f"Unsupported numerical correlation method: {method}")

    def _calculate_categorical_categorical_correlation(self, categorical_cols, method="cramers_v"):
        """
        Calculates the association matrix for categorical columns using the specified method.

        Args:
            categorical_cols (list): A list of categorical column names.
            method (str, optional): The association method to use.
                Defaults to "cramers_v". Options are "cramers_v", "mutual_info".

        Returns:
            pd.DataFrame: The association matrix for the categorical columns.

        Raises:
            ValueError: If an unsupported categorical association method is provided.
        """
        cat_cat = pd.DataFrame(index=categorical_cols,
                               columns=categorical_cols)
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 != col2:
                    try:
                        if method == "cramers_v":
                            score = self._cramers_v(
                                self.data[col1], self.data[col2])
                        elif method == "mutual_info":
                            le = LabelEncoder()
                            x = le.fit_transform(self.data[col1].astype(str))
                            y = le.fit_transform(self.data[col2].astype(str))
                            score = mutual_info_classif(
                                x.reshape(-1, 1), y, discrete_features=True).mean()
                        else:
                            raise ValueError(
                                f"Unsupported categorical-categorical method: {method}")
                        cat_cat.loc[col1, col2] = score
                    except:
                        cat_cat.loc[col1, col2] = np.nan
        return cat_cat.astype(float)

    def _calculate_categorical_numerical_correlation(self, categorical_cols, numerical_cols, method="correlation_ratio"):
        """
        Calculates the association scores between categorical and numerical columns
        using the specified method.

        Args:
            categorical_cols (list): A list of categorical column names.
            numerical_cols (list): A list of numerical column names.
            method (str, optional): The association method to use.
                Defaults to "correlation_ratio". Options are "correlation_ratio", "f_test", "mutual_info".

        Returns:
            pd.DataFrame: A DataFrame with categorical columns as index and
                          numerical columns as columns, containing the association scores.

        Raises:
            ValueError: If an unsupported categorical-numerical association method
                        is provided.
        """

        cat_num = pd.DataFrame(index=categorical_cols, columns=numerical_cols)
        for cat_col in categorical_cols:
            for num_col in numerical_cols:
                try:
                    values = self.data[[cat_col, num_col]].dropna()
                    x = values[cat_col]
                    y = values[num_col]
                    if method == "correlation_ratio":
                        score = self._correlation_ratio(x, y)
                    elif method == "f_test":
                        x_enc = LabelEncoder().fit_transform(x.astype(str))
                        score = f_classif(x_enc.reshape(-1, 1), y)[0][0]
                    elif method == "mutual_info":
                        x_enc = LabelEncoder().fit_transform(x.astype(str))
                        score = mutual_info_regression(
                            x_enc.reshape(-1, 1), y)[0]
                    else:
                        raise ValueError(
                            f"Unsupported categorical-numerical method: {method}")
                    cat_num.loc[cat_col, num_col] = score
                except:
                    cat_num.loc[cat_col, num_col] = np.nan
        return cat_num.astype(float)

    def _calculate_correlations(self, num_method="pearson", cat_method="cramers_v",
                                cat_num_method="correlation_ratio"):
        """
        Calculates pairwise correlations/associations between all numerical and
        categorical columns in the DataFrame.

        Args:
            num_method (str, optional): The method for numerical-numerical correlation.
                Defaults to "pearson".
            cat_method (str, optional): The method for categorical-categorical association.
                Defaults to "cramers_v".
            cat_num_method (str, optional): The method for categorical-numerical association.
                Defaults to "correlation_ratio".

        Returns:
            dict: A dictionary containing correlation/association matrices/DataFrames
                  for each variable type combination:
                  - "numerical_numerical": DataFrame of numerical correlations.
                  - "categorical_categorical": DataFrame of categorical associations.
                  - "categorical_numerical": DataFrame of categorical-numerical associations.
        """

        numerical = self.data.select_dtypes(
            include=[np.number]).columns.tolist()
        categorical = self.data.select_dtypes(
            include=["object", "category", "bool"]).columns.tolist()

        results = {}
        results["numerical_numerical"] = self._calculate_numerical_numerical_correlation(
            numerical, num_method)
        results["categorical_categorical"] = self._calculate_categorical_categorical_correlation(
            categorical, cat_method)
        results["categorical_numerical"] = self._calculate_categorical_numerical_correlation(
            categorical, numerical, cat_num_method)

        return results

    def correlation_matrix(self, num_method="pearson", cat_method="cramers_v",
                           cat_num_method="correlation_ratio", return_matrix=False):
        """
        Calculates and returns a unified table of pairwise correlations/associations
        between all variables.

        Args:
            num_method (str, optional): The method for numerical-numerical correlation.
                Defaults to "pearson".
            cat_method (str, optional): The method for categorical-categorical association.
                Defaults to "cramers_v".
            cat_num_method (str, optional): The method for categorical-numerical association.
                Defaults to "correlation_ratio".
            return_matrix (bool, optional): Whether to also return a symmetric matrix form. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Var1', 'Var2', and 'Correlation',
                        sorted by the absolute value of the correlation in descending order.
            pd.DataFrame (optional): A symmetric correlation matrix if return_matrix=True.
        """

        corr_results = self._calculate_correlations(num_method=num_method, cat_method=cat_method,
                                                    cat_num_method=cat_num_method)
        records = []

        # Unpack numerical vs numerical
        cols = corr_results["numerical_numerical"].columns
        for i, var1 in enumerate(cols):
            for j in range(i + 1, len(cols)):
                var2 = cols[j]
                corr = corr_results["numerical_numerical"].iloc[i, j]
                if pd.notna(corr):
                    records.append((var1, var2, corr))

        # Unpack categorical vs categorical
        for var1 in corr_results["categorical_categorical"].index:
            for var2 in corr_results["categorical_categorical"].columns:
                if var1 != var2:
                    corr = corr_results["categorical_categorical"].loc[var1, var2]
                    if pd.notna(corr):
                        records.append(
                            (var1, var2, corr))

        # Unpack categorical vs numerical
        for var1 in corr_results["categorical_numerical"].index:
            for var2 in corr_results["categorical_numerical"].columns:
                corr = corr_results["categorical_numerical"].loc[var1, var2]
                if pd.notna(corr):
                    records.append((var1, var2, corr))

        unified_df = pd.DataFrame(records, columns=["Var1", "Var2", "Correlation"]).sort_values(
            by="Correlation", key=abs, ascending=False).reset_index(drop=True)

        if return_matrix:
            # Build symmetric matrix
            features = pd.unique(unified_df[["Var1", "Var2"]].values.ravel())
            matrix = pd.DataFrame(
                index=features, columns=features, dtype=float)

            for _, row in unified_df.iterrows():
                matrix.loc[row["Var1"], row["Var2"]] = row["Correlation"]
                matrix.loc[row["Var2"], row["Var1"]] = row["Correlation"]

            np.fill_diagonal(matrix.values, 1.0)

            return unified_df, matrix

        return unified_df

    def get_high_correlations(self, num_method="pearson", cat_method="cramers_v",
                              cat_num_method="correlation_ratio", threshold=0.7):
        """
        Identifies pairs of variables with high absolute correlation based on a specified threshold
        and the correlation methods used for the initial calculation.

        Args:
            num_method (str, optional): The numerical correlation method used.
                Defaults to "pearson". Options are "pearson", "spearman", "kendall".
            cat_method (str, optional): The categorical association method used.
                Defaults to "cramers_v". Options are "cramers_v", "mutual_info".
            cat_num_method (str, optional): The categorical-numerical association method used.
                Defaults to "correlation_ratio". Options are "correlation_ratio", "f_test", "mutual_info".
            threshold (float, optional): The minimum absolute correlation value to consider as "high".
                Defaults to 0.7.

        Returns:
            dict: A dictionary where keys indicate the type of variable pair
                  (e.g., "numerical_numerical", "categorical_categorical",
                  "categorical_numerical") and values are pandas DataFrames. Each
                  DataFrame contains columns 'Var1', 'Var2', and 'Correlation' for
                  pairs with an absolute correlation greater than or equal to the threshold.
                  Returns an empty dictionary if no high correlations are found or if
                  the threshold is not met.
        """
        corr_results = self._calculate_correlations(num_method=num_method,
                                                    cat_method=cat_method,
                                                    cat_num_method=cat_num_method)
        high_corr = {}
        if threshold:
            for key, df in corr_results.items():
                if isinstance(df, pd.DataFrame):
                    corr_pairs = (
                        df.where(np.triu(np.ones(df.shape), k=1).astype(bool))
                        .stack()
                        .reset_index()
                    )
                    corr_pairs.columns = ['Var1', 'Var2', 'Correlation']
                    high_corr[key] = corr_pairs[corr_pairs['Correlation'].abs()
                                                >= threshold]
        return high_corr
