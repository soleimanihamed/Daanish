# utils/data_io/loader.py

from utils.data_io.reader import DataReader
from utils.data_io.database_connector import SQLServerConnector
from utils.core.config import get_database_config


def load_data(source_type, input_path=None, query=None, global_config=None, **kwargs):
    """
    Loads dataset from a specified source.

    Args:
        source_type (str): One of 'csv', 'sql', 'excel', or 'json'.
        input_path (str or None): File path if using 'csv', 'excel', or 'json'.
        query (str): SQL query if using SQL
        global_config (dict): DB credentials
        **kwargs: Additional arguments to pass to the reader (e.g., sheet_name, usecols, json_source)

    Returns:
        pd.DataFrame
    """
    if source_type == 'csv':
        return DataReader.read_csv(input_path, **kwargs)
    elif source_type == 'excel':
        return DataReader.read_excel(input_path, **kwargs)
    elif source_type == 'sql':
        db_config = get_database_config(global_config)
        db = SQLServerConnector(**db_config)
        conn = db.connect()
        df = DataReader.read_sql_query(query, conn)
        db.close()
        return df
    elif source_type == 'json':
        # Priority: json_source from kwargs > input_path
        json_source = kwargs.get('json_source', input_path)
        if json_source is None:
            raise ValueError(
                "JSON source not provided. Use 'input_path' or 'json_source'.")
        return DataReader.read_json_custom_format(json_source)

    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
