# utils/data_io/loader.py

from utils.data_io.reader import DataReader
from utils.data_io.database_connector import SQLServerConnector
from utils.core.config import get_database_config


def load_data(source_type, input_path=None, query=None, global_config=None):
    """
    Loads dataset from a specified source.

    Args:
        source_type (str): 'csv', 'sql', or 'excel'
        input_path (str): File path if using csv/excel
        query (str): SQL query if using SQL
        global_config (dict): DB credentials

    Returns:
        pd.DataFrame
    """
    if source_type == 'csv':
        return DataReader.read_csv(input_path)
    elif source_type == 'excel':
        return DataReader.read_excel(input_path)
    elif source_type == 'sql':
        db_config = get_database_config(global_config)
        db = SQLServerConnector(**db_config)
        conn = db.connect()
        df = DataReader.read_sql_query(query, conn)
        db.close()
        return df
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")
