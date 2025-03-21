
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
from utils.feature_manager import FeatureManager
from utils.main_dataset_manager import MainDatasetManager


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
        'source_type': project_config.get('datasource_type', 'source_type'),
        'model_features_query': project_config.get('db_queries', 'model_features_query'),
        'main_dataset_query': project_config.get('db_queries', 'main_dataset_query')
    }

    return config


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
    main_dataset = project_config['main_dataset']
    model_features = project_config['features_attributes']
    source_type = project_config['source_type']
    model_features_query = project_config['model_features_query']
    main_dataset_query = project_config['main_dataset_query']

    # ----------------------------------------------------------------------------------

    # Step 4: Load main dataset

    dataset_manager = MainDatasetManager(
        source_type=source_type,
        source_path=Construct_Input_Path(input_data_folder, main_dataset),
        global_config=global_config,
        query=main_dataset_query
    )

    # Get the dataset
    main_df = dataset_manager.get_data()

    # ----------------------------------------------------------------------------------

    # Step 5: Load model's features and their attributes (data type category, target variable)

    feature_manager = FeatureManager(
        source_type=source_type,
        source_path=Construct_Input_Path(input_data_folder, model_features),
        global_config=global_config,
        query=model_features_query
    )

    nominal_features = feature_manager.get_nominal_features()
    ordinal_features = feature_manager.get_ordinal_features()
    numerical_features = feature_manager.get_numerical_features()
    target_variable = feature_manager.get_target_variable()
    all_features = feature_manager.get_all_features()

    print("Nominal Features:", nominal_features)
    print("Ordinal Features:", ordinal_features)
    print("Numerical Features:", numerical_features)
    print("Target Variable:", target_variable)
    print("All Features:", all_features)

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

    # Opionally the 'method' variable can be passed to determine the method for finding the best fit
    # THis method accepts these values: 'sumsquare_error','aic' or 'bic'
    # By default it is set to 'sumsquare_error'
    # To pass variable based on its index: [numerical_features[0]] or simpley write its name: "person_age"
    # distribution_results = eda_service.fit_best_distribution(
    #     numerical_features, method='sumsquare_error', common_distributions=True, timeout=120)

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
    #     fitted_distributions=distribution_results, variables=numerical_features)

    viz.plot_histogram(variables=numerical_features, orientation="vertical")

    # ----------------------------------------------------------------------------------
    # Step 8: Close the database connection (if using database)
    if source_type == "sql":
        db_utils.close_connection()


if __name__ == "__main__":
    main()
