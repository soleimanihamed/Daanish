# utils/main_dataset_manager.py
import pandas as pd
from utils.csv_data_utils import CSVDataUtils
from utils.database_utils import DatabaseUtils
from utils.config_utils import get_database_config


class MainDatasetManager:
    """
    Handles loading and preprocessing of the main dataset.
    """

    def __init__(self, source_type, source_path=None, global_config=None, query=None):
        """
        Initialize MainDatasetManager.

        Args:
            source_type (str): 'csv' or 'sql', indicating the data source type.
            source_path (str, optional): Path to the CSV file (if using CSV).
            global_config (dict, optional): Global configuration for database credentials (if using SQL).
            query (str, optional): SQL query to fetch data.

        Raises:
            ValueError: If the required parameters are missing or invalid.
        """
        self.data = None  # Stores the loaded dataset

        # 🔹 Load dataset based on source type
        if source_type == 'csv' and source_path:
            self._load_from_csv(source_path)
        elif source_type == 'sql' and global_config and query:
            self._load_from_sql(global_config, query)
        else:
            raise ValueError(
                "Invalid source_type or missing required parameters.")

    def _load_from_csv(self, csv_path):
        """Load main dataset from a CSV file."""
        self.data = CSVDataUtils.read_csv_file(csv_path)
        if self.data is None:
            raise ValueError(f"Failed to load dataset from {csv_path}")

    def _load_from_sql(self, global_config, query):
        """Load main dataset from a SQL database."""
        db_config = get_database_config(global_config)
        db = DatabaseUtils(
            db_config['server'], db_config['database'], db_config['username'], db_config['password']
        )
        db.connect()
        self.data = db.read_sql_query(query)
        db.close_connection()

        if self.data is None:
            raise ValueError("SQL query returned no data.")

    def get_data(self):
        """Return the loaded dataset."""
        return self.data

    def describe_data(self):
        """Return basic dataset statistics."""
        if self.data is not None:
            return self.data.describe()
        return None

    def handle_missing_values(self, strategy="drop"):
        """
        Handle missing values in the dataset.

        Args:
            strategy (str): "drop" to remove missing values, "fill_mean" to replace with column mean.
        """
        if self.data is None:
            raise ValueError("No dataset loaded.")

        if strategy == "drop":
            self.data.dropna(inplace=True)
        elif strategy == "fill_mean":
            self.data.fillna(self.data.mean(), inplace=True)
        else:
            raise ValueError("Invalid missing value handling strategy.")
