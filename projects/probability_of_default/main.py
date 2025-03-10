
from utils.config_utils import initialize_daanish,  load_project_config, get_database_config
from utils.csv_data_utils import CSVDataUtils
from utils.database_utils import DatabaseUtils
from utils.eda_sweetviz import SweetvizEDA
from utils.save_utils import SaveUtils
import os
from configparser import ConfigParser
import pandas as pd


def load_project_configuration():
    """
    Load project-specific configuration from project_config.ini.

    Returns:
        dict: A dictionary containing project configuration values.
    """

    project_root = os.path.dirname(__file__)
    project_config = load_project_config(project_root)

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


def load_data(project_root, input_data_folder, input_data_file, use_database, global_config=None, query=None):
    """
    Load a dataset from a CSV file or database.

    Args:
        project_root (str): Root directory of the project.
        input_data_folder (str): Folder containing the input data file.
        input_data_file (str): Name of the input data file.
        use_database (bool): Whether to use the database.
        global_config (dict, optional): Global configuration containing database credentials.
        query (str, optional): SQL query to execute (if using database).

    Returns:
        pd.DataFrame: The dataset as a DataFrame.
        DatabaseUtils: The database utility object (if using database).
    """
    db_utils = None
    if use_database:
        if global_config is None:
            raise ValueError(
                "Global configuration is required when using the database.")
        if query is None:
            raise ValueError("Query is required when using the database.")

        # Step 1: Get database configuration and connect
        db_config = get_database_config(global_config)
        db_utils = DatabaseUtils(
            db_config['server'], db_config['database'], db_config['username'], db_config['password'])
        db_utils.connect()

        # Step 2: Load data from the database
        df = db_utils.read_sql_query(query)
    else:
        # Step 3: Load data from a CSV file
        file_path = os.path.join(
            project_root, input_data_folder, input_data_file)
        df = CSVDataUtils.read_csv_file(file_path)

    return df, db_utils


def perform_eda_Sweetviz(df, project_root, report_output_folder):
    """
    Perform exploratory data analysis (EDA) using Sweetviz.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        report_output_folder (str): Folder to save the EDA report.
        report_output_folder (str): Folder to save the EDA report (relative to project_root).
    """

    # Construct the full output path
    full_report_output_folder = os.path.join(
        project_root, report_output_folder)

    # Define the full path for the Sweetviz report
    sweetviz_report_path = os.path.join(
        full_report_output_folder, 'raw_sweetviz_report.html')

    eda_service = SweetvizEDA(df)
    eda_service.generate_report(output_file=sweetviz_report_path)


def print_data_samples(df, feature_config):
    """
    Print samples of the main DataFrame and feature configuration DataFrame.

    Args:
        df (pd.DataFrame): The main dataset.
        feature_config (pd.DataFrame): The feature configuration dataset.
    """
    print("\nSample of the feature configuration DataFrame:")
    print(feature_config.head())
    print("Sample of the main DataFrame:")
    print(df.head())


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

    # Step 4: Load main dataset
    main_data_query = "SELECT * FROM dbo.loan"  # Replace with your actual query
    df, db_utils = load_data(project_root, input_data_folder, input_data_file,
                             use_database, global_config, query=main_data_query)

    # Step 5: Load feature configuration dataset
    # Replace with your actual query
    feature_config_query = "SELECT * FROM dbo.PD_Model_Features"
    feature_config, _ = load_data(project_root, input_data_folder, feature_config_file,
                                  use_database, global_config, query=feature_config_query)

    # Step 6: Print data samples
    print_data_samples(df, feature_config)

    # Step 7: Perform EDA
    perform_eda_Sweetviz(df, project_root, report_output_folder)

    # Step 8: Close the database connection (if using database)
    if use_database:
        db_utils.close_connection()


if __name__ == "__main__":
    main()
