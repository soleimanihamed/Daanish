# utils/preprocessing/data_encoder.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class DataEncoder:
    """
    Encodes categorical features (nominal and ordinal) for machine learning pipelines.

    Supports:
    - One-hot encoding for nominal features
    - Ordinal encoding for ordinal features

    Parameters
    ----------
    nominal_features : list of str
        List of nominal (categorical unordered) feature names.
    ordinal_features : list of str
        List of ordinal (categorical ordered) feature names.
    drop_original : bool
        To keep the original encoded features or drop them
    """

    def __init__(self, nominal_features=None, ordinal_features=None, drop_original: bool = True):
        self.nominal_features = nominal_features or []
        self.ordinal_features = ordinal_features or []
        self.drop_original = drop_original

        self.nominal_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, df: pd.DataFrame):
        """
        Fit the encoders to the training data.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the categorical features.
        """
        if self.nominal_features:
            self.nominal_encoder.fit(df[self.nominal_features])
        if self.ordinal_features:
            self.ordinal_encoder.fit(df[self.ordinal_features])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted encoders.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to encode.

        Returns
        -------
        pd.DataFrame
            Encoded DataFrame.
        """
        encoded_parts = []

        if self.nominal_features:
            nominal_array = self.nominal_encoder.transform(
                df[self.nominal_features])
            nominal_df = pd.DataFrame(
                nominal_array,
                columns=self.nominal_encoder.get_feature_names_out(
                    self.nominal_features),
                index=df.index
            )
            encoded_parts.append(nominal_df)

        if self.ordinal_features:
            ordinal_array = self.ordinal_encoder.transform(
                df[self.ordinal_features])
            ordinal_df = pd.DataFrame(
                ordinal_array, columns=self.ordinal_features, index=df.index)
            encoded_parts.append(ordinal_df)

        return pd.concat(encoded_parts, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataset by applying encoding to nominal and ordinal features.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame with original features plus encoded features.
        """

        df_copy = df.copy()

        encoded_parts = []

        # Encode nominal features
        if self.nominal_features:
            nominal_encoded = self.nominal_encoder.fit_transform(
                df_copy[self.nominal_features])
            nominal_encoded_df = pd.DataFrame(
                nominal_encoded,
                columns=self.nominal_encoder.get_feature_names_out(
                    self.nominal_features),
                index=df_copy.index
            )
            encoded_parts.append(nominal_encoded_df)

        # Encode ordinal features
        if self.ordinal_features:
            ordinal_encoded = self.ordinal_encoder.fit_transform(
                df_copy[self.ordinal_features])
            ordinal_encoded_df = pd.DataFrame(
                ordinal_encoded,
                columns=self.ordinal_features,
                index=df_copy.index
            )
            encoded_parts.append(ordinal_encoded_df)

        # Drop encoded original features from original DataFrame
        drop_cols = []
        if self.nominal_features:
            drop_cols.extend(self.nominal_features)
        if self.ordinal_features:
            drop_cols.extend(self.ordinal_features)

        if self.drop_original:
            df_remaining = df_copy.drop(columns=drop_cols, errors='ignore')
        else:
            df_remaining = df_copy

        # Combine remaining + encoded parts
        final_df = pd.concat([df_remaining] + encoded_parts, axis=1)

        return final_df
