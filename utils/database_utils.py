# utils/database_utils.py

import pandas as pd
import pyodbc


class DatabaseUtils:
    def __init__(self, server, database, username, password):
        """
        Initializes the DatabaseUtils class with connection details.

        Args:
            server (str): The SQL Server address.
            database (str): The database name.
            username (str): Username for authentication.
            password (str): Password for authentication.
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection = None

    def connect(self):
        """
        Establishes a connection to the SQL Server database.

        Returns:
            pyodbc.Connection: Database connection object.
        """
        conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}'
        try:
            self.connection = pyodbc.connect(conn_str)
            print("Connection to SQL Server database was successful.")
        except Exception as e:
            print(f"Error connecting to SQL Server: {e}")

    def read_sql_query(self, query):
        """
        Executes a SQL query and returns the result as a DataFrame.

        Args:
            query (str): SQL query to execute.

        Returns:
            pd.DataFrame: Query result as a DataFrame.
        """
        if self.connection is None:
            print("No database connection. Call connect() first.")
            return None
        try:
            df = pd.read_sql(query, self.connection)
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def close_connection(self):
        """
        Closes the database connection.
        """
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
