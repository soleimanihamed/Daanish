
from utils.config_utils import initialize_daanish,  load_project_config, get_database_config
from utils.csv_data_utils import CSVDataUtils
from utils.database_utils import DatabaseUtils
from utils.eda_descriptive_analysis import DescriptiveEDAAnalysis
from utils.eda_sweetviz import SweetvizEDA
from utils.save_utils import SaveUtils
import os
from configparser import ConfigParser
import pandas as pd
from utils.visualisation import Visualisation


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
        'main_dataset': project_config.get('input_files', 'main_dataset'),
        'features_attributes': project_config.get('input_files', 'features_attributes'),
        'use_database': project_config.getboolean('DATABASE', 'use_database', fallback=False)
    }

    return config


def load_data(Input_Path, use_database, global_config=None, query=None):
    """
    Load a dataset from a CSV file or database.

    Args:
        Input_Path (str): Complete path of the input file.
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
        df = CSVDataUtils.read_csv_file(Input_Path)

    return df, db_utils


# Construct the full input path
def Construct_Input_Path(input_folder, file_name):

    # Construct the full input path
    project_root = os.path.dirname(__file__)

    full_input_folder = os.path.join(
        project_root, input_folder)

    # Define the full path
    full_report_path = os.path.join(
        full_input_folder, file_name)

    return full_report_path


# Construct the full output path
def Construct_Output_Path(output_folder, file_name):

    # Construct the full output path
    project_root = os.path.dirname(__file__)

    full_report_output_folder = os.path.join(
        project_root, output_folder)

    # Define the full path
    full_report_path = os.path.join(
        full_report_output_folder, file_name)

    return full_report_path


def main():
    """
    Main function to run the project.
    """
    # Step 1: Initialize Daanish core setup
    global_config = initialize_daanish()

    # ----------------------------------------------------------------------------------

    # Step 2: Load project-specific configuration
    project_config = load_project_configuration()

    # ----------------------------------------------------------------------------------

    # Step 3: Access configuration values
    input_data_folder = project_config['input_data_folder']
    output_data_folder = project_config['output_data_folder']
    report_output_folder = project_config['report_output_folder']
    use_database = project_config['use_database']

    # ----------------------------------------------------------------------------------

    # Step 4: Load main dataset
    main_data_query = "SELECT * FROM dbo.loan"  # Replace with your actual query
    dataset_file = project_config['main_dataset']
    main_df, db_utils = load_data(Construct_Input_Path(input_data_folder, dataset_file),
                                  use_database, global_config, query=main_data_query)

    # ----------------------------------------------------------------------------------

    # Step 5: Load feature configuration dataset
    # Replace with your actual query
    feature_config_query = "SELECT * FROM dbo.PD_Model_Features"
    model_features = project_config['features_attributes']
    main_features, db_utils = load_data(Construct_Input_Path(input_data_folder, model_features),
                                        use_database, global_config, query=feature_config_query)
    eda_service = DescriptiveEDAAnalysis(main_features)
    eda_service.print_data_samples(main_features)

    # ----------------------------------------------------------------------------------
    # Step 6: Exploratory Data Analysis (EDA)

    # Step 6_1: Descriptive Analysis for Raw Data
    eda_service = DescriptiveEDAAnalysis(main_df)
    eda_service.print_data_samples(main_df, 10)
    eda_service.save_summary_to_json(Construct_Output_Path(output_data_folder,
                                                           'main_detailed_EDA.json'))
    # eda_service.print_detailed_summary()
    # eda_service.print_high_level_summary()
    eda_service.dataset_summary()
    eda_service.save_high_level_summary_to_csv(Construct_Output_Path(output_data_folder,
                                                                     'main_EDA.csv'))

# ------------------------------------------
    # Step 6_2: Analyze probability distributions for specific numeric variables

    numeric_variables = main_features[main_features['type']
                                      == 'numerical']['feature'].tolist()

    # Opionally the 'method' variable can be passed to determine the method for finding the best fit
    # THis method accepts these values: 'sumsquare_error','aic' or 'bic'
    # By default it is set to 'sumsquare_error'
    # To pass variable based on its index: [numeric_variables[0]] or simpley write its name: "person_age"
    # distribution_results = eda_service.fit_best_distribution(
    #     numeric_variables, method='sumsquare_error', common_distributions=True, timeout=120)

    # -----------------------------------------

    # Step 6_3: Exploratory Data Analysis (EDA) - Alternative to above with Sweetviz for Raw Data
    # eda_service = SweetvizEDA(main_df)
    # eda_service.generate_report(output_file=Construct_Output_Path(project_root, report_output_folder,
    #                                                               'raw_sweetviz_report.html'))

    # ----------------------------------------------------------------------------------
    # Step 7: Visualisation
    # Initialize Visualization class with the dataset
    viz = Visualisation(main_df)

    # Call the visualization function
    # viz.plot_distributions(
    #     fitted_distributions=distribution_results, variables=numeric_variables)

    viz.plot_histogram(variables=numeric_variables, orientation="vertical")

    # ----------------------------------------------------------------------------------
    # Step 8: Close the database connection (if using database)
    if use_database:
        db_utils.close_connection()


if __name__ == "__main__":
    main()
