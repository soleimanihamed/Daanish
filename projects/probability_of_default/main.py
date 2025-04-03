# projects\probability_of_default\main.py

import os
from utils.config_utils import initialize_daanish,  load_project_config
from utils.eda_descriptive_analysis import DescriptiveEDAAnalysis
from utils.eda_sweetviz import SweetvizEDA
from utils.save_utils import SaveUtils
from utils.visualisation import Visualisation
from utils.feature_manager import FeatureManager
from utils.main_dataset_manager import MainDatasetManager
from utils.eda_statistical_analysis import StatisticalEDAAnalysis
from utils.data_preprocessor import DataPreprocessor
from utils.generate_reports import ReportGenerator


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

    # ------------------------------
    save_utils = SaveUtils(
        output_dir=Construct_Output_Path(output_data_folder, ""))

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
    missing_value_strategies = feature_manager.get_missing_value_strategies()
    missing_fill_values = feature_manager.get_missing_fill_values()

    # print("Nominal Features:", nominal_features)
    # print("Ordinal Features:", ordinal_features)
    # print("Numerical Features:", numerical_features)
    # print("Target Variable:", target_variable)
    # print("All Features:", all_features)
    # print("Missing Value Strategies:", missing_value_strategies)
    # print("Missing Fill Values:", missing_fill_values)

    # ----------------------------------------------------------------------------------
    # Step 6: Exploratory Data Analysis (EDA)

    # ----------------------------------
    # Step 6_1: Descriptive Analysis for Raw Data
    eda_service = DescriptiveEDAAnalysis(main_df)

    # ----------------------------------
    # Print summaries of data samples
    # print(eda_service.get_data_samples(10))

    # ----------------------------------
    # Print summaries of dataset
    # ReportGenerator.print_dataset_summary(eda_service.get_dataset_summary())

    # ----------------------------------
    # Print summaries of feature(s) into console

    # ReportGenerator.print_high_level_summary(
    #     eda_service.get_all_feature_summaries())

    # ----------------------------------
    # Print summaries of feature(s) into a csv file

    # Create csv output for descriptive analysis of a specific feature
    # feature_name = 'person_age'
    # feature_summary = eda_service.get_feature_summary(feature_name)
    # ReportGenerator.save_high_level_summary_to_csv({feature_name: feature_summary},
    #                                                Construct_Output_Path(output_data_folder, 'features_descriptive_summary.csv'))

    # Create csv output for descriptive analysis of all features
    # ReportGenerator.save_high_level_summary_to_csv(eda_service.get_all_feature_summaries(),
    #                                                Construct_Output_Path(output_data_folder, 'features_descriptive_summary.csv'))

    # ----------------------------------
    # Print summaries of features into a json file

    save_utils.save_json(eda_service.get_all_feature_summaries(),
                         Construct_Output_Path(output_data_folder, 'features_descriptive_summary.json'))

# ------------------------------------------
    # Step 6_2: Analyze probability distributions for specific numeric variables

    # Opionally the 'method' variable can be passed to determine the method for finding the best fit
    # THis method accepts these values: 'sumsquare_error','aic' or 'bic'
    # By default it is set to 'sumsquare_error'

    # distribution_results = eda_service.fit_best_distribution(
    #     ['person_age', 'loan_amnt'], method='sumsquare_error', common_distributions=True, timeout=120)

    # -----------------------------------------

    # Step 6_3: Exploratory Data Analysis (EDA) - Alternative to above with Sweetviz for Raw Data
    # eda_service = SweetvizEDA(main_df)
    # eda_service.generate_report(output_file=Construct_Output_Path(project_root, report_output_folder,
    #                                                               'raw_sweetviz_report.html'))

    # ----------------------------------------------------------------------------------
    # Step 7: Visualisation
    # Initialize Visualization class with the dataset
    # Call the visualization function to plot distribution/histogram
    viz = Visualisation(main_df)

    # --------------------------------
    # Plot fitted distributions
    # viz.plot_distributions(
    #     fitted_distributions=distribution_results, variables=numerical_features)

    # --------------------------------
    # Plot histograms
    # viz.plot_histogram(variables=numerical_features, orientation="vertical")
    # viz.plot_histogram(variables=nominal_features, orientation="horizontal")

    # --------------------------------
    # Scatter Plot

    # Scatter plot with color based on `loan_grade`
    # viz.plot_scatter(x_var="loan_amnt", y_var="loan_int_rate",
    #                 hue_var="loan_grade")

    # Scatter plot
    # viz.plot_scatter(x_var="person_age",
    #                  y_var="person_income", trendline=True)

    # --------------------------------
    # Box Plot
    # viz.plot_boxplot(column='loan_percent_income', by='loan_status',
    #                  title='Loan Percent Income by Loan Status')
    # viz.plot_boxplot(column='loan_amnt', by='loan_grade',
    #                  title='Loan Amount by Loan Grade')

    # ----------------------------------------------------------------------------------
    # Step 8: Statistical Analysis
    eda = StatisticalEDAAnalysis(main_df)

    # For two variables
    # crosstab_result_two = eda.crosstab(
    #     "loan_intent", "loan_status", normalize="index")

    # # For three variables
    # crosstab_result_three = eda.crosstab_three_way("person_home_ownership",
    #                                                "loan_status", "loan_grade")

    # Use SaveUtils to generate HTML tables
    # save_utils = SaveUtils(
    #     output_dir=Construct_Output_Path(output_data_folder, ""))
    # save_utils.generate_styled_html_tables(
    #     dataframes=[crosstab_result_two, crosstab_result_three],
    #     filenames=["crosstab_two_styled.html", "crosstab_three_styled.html"]
    # )

    # ----------------------------------------------------------------------------------
    # Step 9: Data Cleaning
    dp = DataPreprocessor(main_df)

    # -----------------------------------
    # Handle missing values
    # imputed_records, imputed_dataset = dp.handle_missing_values(
    #     all_features, strategies=missing_value_strategies, fill_values=missing_fill_values)
    # print(imputed_records)
    # print(imputed_dataset)

    # save_utils.save_dataframe_to_csv(
    #     imputed_records, "imputed_records.csv", overwrite=True)
    # --------------------------------
    # Outlier detection

    # outliers = dp.detect_outliers_iqr(
    #     imputed_dataset, ['person_age', 'person_home_ownership', 'loan_amnt'], threshold=3)

    # outliers = dp.detect_custom_outliers(imputed_dataset, ['person_age', 'person_home_ownership', 'loan_amnt'],
    #                                      lower_bounds={
    #                                          'person_age': 15, 'loan_amnt': 1000},
    #                                      upper_bounds={'person_age': 80, 'loan_amnt': 1000000000})

    # outliers = dp.detect_outliers_zscore(imputed_dataset, [
    #                                      'person_age', 'person_home_ownership', 'loan_amnt'], threshold=3, alpha=0.05)

    # outliers = dp.detect_outliers_distribution(imputed_dataset,
    #                                            distribution_results, confidence_interval=0.99)

    # outliers = dp.detect_outliers_isolation_forest(imputed_dataset,
    #                                                features=["person_age", "person_emp_length"], contamination=0.001, n_estimators=500)

    # outliers = dp.detect_outliers_lof(imputed_dataset,
    #                                   features=["person_age"], n_neighbors=10, contamination=0.01)
    # print(outliers)

    # cleaned_df = dp.remove_outliers(imputed_dataset, outliers)
    # print(cleaned_df)

    # --------------------------------

    # print(outliers)
    # save_utils.save_dataframe_to_csv(
    #     outliers, "outliers_isolation_forest.csv", overwrite=True)


if __name__ == "__main__":
    main()
