# utils/data_io/database_connector.py

import pyodbc


class SQLServerConnector:
    """
    Handles SQL Server connection logic.

    Methods
    -------
    connect() -> pyodbc.Connection
        Establishes and returns the SQL Server connection.

    close()
        Closes the SQL Server connection.
    """

    def __init__(self, server, database, username, password):
        self.conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};DATABASE={database};UID={username};PWD={password}"
        )
        self.connection = None

    def connect(self):
        try:
            self.connection = pyodbc.connect(self.conn_str)
            print("Connected to SQL Server.")
            return self.connection
        except Exception as e:
            print(f"Connection failed: {e}")
            return None

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            print("Connection closed.")
