# utils/CSV_data_utils.py

import pandas as pd
import os


class CSVDataUtils:
    @staticmethod
    def read_csv_file(filename, field_name=None, index_col=None, parse_dates=True):
        """
        Reads a CSV file and returns a DataFrame or Series.

        Args:
            filename (str): Path to the CSV file.
            field_name (str): Specific field to extract.
            index_col (str): Column to set as index.
            parse_dates (bool): Whether to parse dates.

        Returns:
            pd.DataFrame or pd.Series: Loaded data.
        """
        try:
            df = pd.read_csv(filename, index_col=index_col,
                             parse_dates=parse_dates)
            if field_name:
                if field_name not in df.columns:
                    raise ValueError(
                        f"Field '{field_name}' not found in the CSV file.")
                return df[field_name]
            return df
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except ValueError as e:
            print(e)

    @staticmethod
    def save_dataframe_to_file(df, filename, overwrite=True, index=True):
        """
        Saves a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            filename (str): Output file path.
            overwrite (bool): Overwrite existing file if True.
            index (bool): Include index in the saved file.
        """
        if os.path.exists(filename) and not overwrite:
            print(
                f"File {filename} already exists. Set overwrite=True to overwrite it.")
            return
        df.to_csv(filename, index=index)
        print(f"Data saved to {filename} successfully.")
