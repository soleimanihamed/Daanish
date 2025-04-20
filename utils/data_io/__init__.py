# utils/data_io/__init__.py

from .loader import load_data
from .reader import DataReader
from .database_connector import SQLServerConnector

# Instantiate DataReader
_reader = DataReader()

read_csv = _reader.read_csv
read_excel = _reader.read_excel
read_sql_query = _reader.read_sql_query

# A friendlier alias
SQLConnector = SQLServerConnector
