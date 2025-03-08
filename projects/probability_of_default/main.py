
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

    # Step 4: Load data
    if use_database:
        # Use database
        db_config = get_database_config(global_config)
        db_utils = DatabaseUtils(
            db_config['server'], db_config['database'], db_config['username'], db_config['password'])
        db_utils.connect()
        query = "SELECT * FROM dbo.loan"
        df = db_utils.read_sql_query(query)
        db_utils.close_connection()
    else:
        # Use CSV
        project_root = os.path.dirname(__file__)
        file_path = os.path.join(
            project_root, input_data_folder, input_data_file)
        df = CSVDataUtils.read_csv_file(file_path)

    # Step 5: Print a sample of the DataFrame
    print("Sample of the loaded DataFrame:")
    print(df.head())


if __name__ == "__main__":
    main()
