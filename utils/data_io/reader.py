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
    def read_csv(filepath, field_name=None, index_col=None, parse_dates=True):
        try:
            df = pd.read_csv(filepath, index_col=index_col,
                             parse_dates=parse_dates)
            if field_name:
                if field_name not in df.columns:
                    raise ValueError(f"Field '{field_name}' not found in CSV.")
                return df[field_name]
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")

    @staticmethod
    def read_excel(filepath, sheet_name=0, index_col=None, parse_dates=True):
        try:
            return pd.read_excel(filepath, sheet_name=sheet_name, index_col=index_col, parse_dates=parse_dates)
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
