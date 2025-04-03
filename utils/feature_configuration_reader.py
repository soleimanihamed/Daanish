import pandas as pd
from utils.csv_data_utils import DataUtils
from utils.database_utils import DatabaseUtils


class FeatureConfigurationReader:
    def __init__(self, feature_config=None, config_source=None, db_params=None):
        """
        Initializes the FeatureConfigurationReader to read feature types from different sources.

        Args:
            feature_config (dict): Dictionary containing feature types (nominal, ordinal, numerical).
            config_source (str): Source of feature configuration ('csv', 'sql').
            db_params (dict): Dictionary containing parameters for the data source.
        """
        if feature_config is not None:
            self.nominal_features = feature_config.get('nominal_features', [])
            self.ordinal_features = feature_config.get('ordinal_features', [])
            self.numerical_features = feature_config.get(
                'numerical_features', [])
        elif config_source == 'csv' and db_params is not None:
            csv_path = db_params.get('csv_path')
            if csv_path:
                # Read feature configuration from CSV
                df = DataUtils.read_csv_file(csv_path)
                if df is None:
                    raise ValueError(
                        f"CSV file at '{csv_path}' could not be read or does not exist.")
                if 'feature' not in df.columns or 'type' not in df.columns:
                    raise ValueError(
                        "The CSV file must contain 'feature' and 'type' columns.")

                self.nominal_features = df[df['type']
                                           == 'nominal']['feature'].tolist()
                self.ordinal_features = df[df['type']
                                           == 'ordinal']['feature'].tolist()
                self.numerical_features = df[df['type']
                                             == 'numerical']['feature'].tolist()
            else:
                raise ValueError(
                    "CSV path must be provided in db_params for CSV configuration.")
        elif config_source == 'sql' and db_params is not None:
            # Read feature configuration from SQL database
            db = DatabaseUtils(
                db_params['server'], db_params['database'], db_params['username'], db_params['password'])
            db.connect()
            query = db_params['query']
            df = db.read_sql_query(query)
            if df is None:
                raise ValueError(f"SQL query '{query}' returned no data.")
            self.nominal_features = df[df['type']
                                       == 'nominal']['feature'].tolist()
            self.ordinal_features = df[df['type']
                                       == 'ordinal']['feature'].tolist()
            self.numerical_features = df[df['type']
                                         == 'numerical']['feature'].tolist()
            db.close_connection()
        else:
            raise ValueError(
                "Feature configuration must be provided either through a dictionary, CSV, or SQL database.")
