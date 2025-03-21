# feature_manager.py
import pandas as pd
from utils.csv_data_utils import CSVDataUtils
from utils.database_utils import DatabaseUtils
from utils.config_utils import get_database_config


class FeatureManager:
    """
    Handles feature configuration, including identifying numerical, nominal, and ordinal features,
    as well as detecting the target variable.
    """

    def __init__(self, source_type, source_path=None, global_config=None,  query=None):
        """
        Initialize FeatureManager.

        Args:
            source_type (str): 'csv' or 'sql', indicating the data source type.
            source_path (str, optional): Path to the CSV file (if source_type is 'csv').
            global_config (dict, optional): Global configuration for database credentials (if using SQL).
            query (str, optional): SQL query to fetch feature configuration.

        Raises:
            ValueError: If the required parameters are missing or invalid.
        """
        self.nominal_features = []
        self.ordinal_features = []
        self.numerical_features = []
        self.target_variable = None
        self.all_features = []

        if source_type == 'csv' and source_path:
            self._load_from_csv(source_path)
        elif source_type == 'sql' and global_config and query:
            self._load_from_sql(global_config, query)
        else:
            raise ValueError(
                "Invalid source_type or missing required parameters.")

    def _load_from_csv(self, csv_path):
        """Load feature configuration from a CSV file using CSVDataUtils."""
        df = CSVDataUtils.read_csv_file(csv_path)

        if df is None or 'feature' not in df.columns or 'type' not in df.columns:
            raise ValueError(
                "CSV file must contain 'feature' and 'type' columns.")

        self._process_feature_dataframe(df)

    def _load_from_sql(self, global_config, query):
        """Load feature configuration from a SQL database using global_config."""
        db_config = get_database_config(global_config)
        db = DatabaseUtils(
            db_config['server'], db_config['database'], db_config['username'], db_config['password']
        )
        db.connect()
        df = db.read_sql_query(query)
        db.close_connection()

        if df is None or 'feature' not in df.columns or 'type' not in df.columns:
            raise ValueError(
                "SQL query must return 'feature' and 'type' columns.")

        self._process_feature_dataframe(df)

    def _process_feature_dataframe(self, df):
        """Processes the feature DataFrame and categorizes feature types."""
        self.nominal_features = df[df['type'] == 'nominal']['feature'].tolist()
        self.ordinal_features = df[df['type'] == 'ordinal']['feature'].tolist()
        self.numerical_features = df[df['type']
                                     == 'numerical']['feature'].tolist()
        self.target_variable = df[df['target_variable']
                                  == 1]['feature'].tolist()

        # Store all features in a single list
        self.all_features_list = (
            self.nominal_features + self.ordinal_features + self.numerical_features
        )

    def get_nominal_features(self):
        """Return a list of nominal features."""
        return self.nominal_features

    def get_ordinal_features(self):
        """Return a list of ordinal features."""
        return self.ordinal_features

    def get_numerical_features(self):
        """Return a list of numerical features."""
        return self.numerical_features

    def get_target_variable(self):
        """Return the target variable."""
        return self.target_variable

    def get_all_features(self):
        """Return a dictionary of all feature categories."""
        return self.all_features_list
