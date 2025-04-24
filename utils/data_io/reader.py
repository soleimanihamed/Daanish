# utils/data_io/reader.py

import pandas as pd


class DataReader:
    """
    Unified reader for CSV, Excel, and SQL (requires external connection).

    Methods
    -------
    read_csv(filepath, ...)
    read_excel(filepath, ...)
    read_sql_query(query, connection)
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
