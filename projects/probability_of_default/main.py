
from utils.config_utils import initialize_daanish,  load_project_config
from utils.csv_data_utils import CSVDataUtils
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
        'feature_config_file': project_config.get('input_files', 'feature_config_file')
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

    # Step 4: Construct the full file path relative to the project root
    project_root = os.path.dirname(__file__)
    file_path = os.path.join(project_root, input_data_folder, input_data_file)

    # Step 5: Load data using CSVDataUtils
    df = CSVDataUtils.read_csv_file(file_path)


if __name__ == "__main__":
    main()
