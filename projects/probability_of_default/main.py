
from utils.config_utils import initialize_daanish,  load_project_config, get_database_config
from utils.csv_data_utils import CSVDataUtils
from utils.database_utils import DatabaseUtils
import os
from configparser import ConfigParser
import pandas as pd


def load_project_configuration():
    """
    Load project-specific configuration from project_config.ini.

    Returns:
        dict: A dictionary containing project configuration values.
    """
    # Step 1: Load project-specific config
    project_root = os.path.dirname(__file__)
    project_config = load_project_config(project_root)

    # Step 2: Extract configuration values
    config = {
        'input_data_folder': project_config.get('paths', 'input_data_folder'),
        'output_data_folder': project_config.get('paths', 'output_data_folder'),
        'report_output_folder': project_config.get('paths', 'report_output_folder'),
        'input_data_file': project_config.get('input_files', 'input_data_file'),
        'feature_config_file': project_config.get('input_files', 'feature_config_file'),
        'use_database': project_config.getboolean('DATABASE', 'use_database', fallback=False)
    }

    return config


def load_feature_config(project_root, input_data_folder, feature_config_file, use_database, db_utils=None):
    """
    Load feature configuration from a CSV file or database table.

    Args:
        project_root (str): Root directory of the project.
        input_data_folder (str): Folder containing the feature configuration file.
        feature_config_file (str): Name of the feature configuration file.
        use_database (bool): Whether to use the database.
        db_utils (DatabaseUtils): Database utility object (if using database).

    Returns:
        pd.DataFrame: Feature configuration as a DataFrame.
    """
    if use_database:
        # Load feature configuration from the database
        if db_utils is None:
            raise ValueError(
                "DatabaseUtils object is required when using the database.")
        query = "SELECT * FROM dbo.PD_Model_Features"  # Replace with your table name
        feature_config = db_utils.read_sql_query(query)
    else:
        # Load feature configuration from a CSV file
        file_path = os.path.join(
            project_root, input_data_folder, feature_config_file)
        feature_config = CSVDataUtils.read_csv_file(file_path)

    return feature_config


def main():
    """
    Main function to run the project.
    """
    # Step 1: Initialize Daanish core setup
    global_config = initialize_daanish()

    # Step 2: Load project-specific configuration
    project_config = load_project_configuration()

    # Step 3: Access configuration values
    input_data_folder = project_config['input_data_folder']
    output_data_folder = project_config['output_data_folder']
    report_output_folder = project_config['report_output_folder']
    input_data_file = project_config['input_data_file']
    feature_config_file = project_config['feature_config_file']
    use_database = project_config['use_database']

    project_root = os.path.dirname(__file__)

    # Step 4: Load data
    if use_database:
        # Use database
        db_config = get_database_config(global_config)
        db_utils = DatabaseUtils(
            db_config['server'], db_config['database'], db_config['username'], db_config['password'])
        db_utils.connect()
        query = "SELECT * FROM dbo.loan"
        df = db_utils.read_sql_query(query)

    else:
        # Use CSV
        file_path = os.path.join(
            project_root, input_data_folder, input_data_file)
        df = CSVDataUtils.read_csv_file(file_path)

   # Step 5: Load feature configuration
    feature_config = load_feature_config(
        project_root, input_data_folder, feature_config_file, use_database, db_utils if use_database else None)

    # Step 6: Print a sample of the DataFrames
    print("Sample of the main DataFrame:")
    print(df.head())
    print("\nSample of the feature configuration DataFrame:")
    print(feature_config.head())

    # Step 7: Close the database connection (if using database)
    if use_database:
        db_utils.close_connection()


if __name__ == "__main__":
    main()
