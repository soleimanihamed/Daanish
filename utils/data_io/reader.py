# utils/data_io/reader.py

import pandas as pd
import json
from typing import Union


class DataReader:
    """
    Unified reader for CSV, Excel, SQL, and custom JSON format.

    Methods
    -------
    read_csv(filepath, **kwargs) -> pd.DataFrame
        Reads a CSV file into a DataFrame.

    read_excel(filepath, **kwargs) -> pd.DataFrame
        Reads an Excel file into a DataFrame.

    read_sql_query(query, connection) -> pd.DataFrame or None
        Executes an SQL query and returns a DataFrame, if connection is provided.

    read_json_custom_format(filepath) -> pd.DataFrame
        Reads a custom-format JSON (from file path, string, or list),
        and the remaining arrays contain data rows.
    """

    @staticmethod
    def read_csv(filepath, **kwargs):
        try:
            return pd.read_csv(filepath, **kwargs)
        except Exception as e:
            print(f"Error reading CSV: {e}")

    @staticmethod
    def read_excel(filepath, **kwargs):
        try:
            return pd.read_excel(filepath, **kwargs)
        except Exception as e:
            print(f"Error reading Excel: {e}")

    @staticmethod
    def read_sql_query(query, connection):
        if not connection:
            print("No valid DB connection provided.")
            return None
        try:
            return pd.read_sql(query, connection)
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    @staticmethod
    def read_json_custom_format(source: Union[str, list]) -> pd.DataFrame:
        """
        Reads a custom-format JSON where:
            - The first element is a list of column names
            - The remaining elements are lists representing rows of data

        Accepts:
            - File path to a .json file
            - JSON string (e.g., from an API)
            - Direct Python list (already parsed)

        Parameters
        ----------
        source : str or list
            Path to a JSON file, a JSON string, or a list of lists.

        Returns
        -------
        pd.DataFrame
            DataFrame constructed from the JSON content.
        """
        try:
            # Case 1: It's a filepath
            if isinstance(source, str):
                if source.strip().endswith(".json"):
                    with open(source, 'r') as file:
                        data = json.load(file)
                else:
                    # Treat it as a JSON string
                    data = json.loads(source)
            elif isinstance(source, list):
                data = source
            else:
                raise ValueError(
                    "Unsupported input type. Expected file path, JSON string, or list of lists.")

            if not data or not isinstance(data, list) or not isinstance(data[0], list):
                raise ValueError(
                    "Invalid JSON format: Expected a list of lists with first row as column names.")

            columns = data[0]
            rows = data[1:]
            return pd.DataFrame(rows, columns=columns)

        except Exception as e:
            print(f"Error reading custom JSON format: {e}")
            return None
