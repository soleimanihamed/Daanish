import sys
import os
import pandas as pd
from configparser import ConfigParser
from utils.eda_sweetviz import SweetvizEDA
from utils.save_utils import SaveUtils
from utils.data_variable_encoder import DataVariableEncoder
from utils.feature_configuration_reader import FeatureConfigurationReader
from utils.eda_custom_analysis import CustomEDAAnalysis
from dotenv import load_dotenv

# Step 1: Load Environment Variables from .env or config.ini
load_dotenv()

# Load global config from config.ini
global_config = ConfigParser()
global_config.read(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'config.ini')))

# Add PYTHONPATH from global config
project_root = global_config.get('global', 'PYTHONPATH', fallback=None)
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

# Load project-specific config
project_config = ConfigParser()
project_config.read(os.path.join(
    os.path.dirname(__file__), 'project_config.ini'))

# Get folder paths
input_data_folder = project_config.get('paths', 'input_data_folder')
output_data_folder = project_config.get('paths', 'output_data_folder')
report_output_folder = project_config.get('paths', 'report_output_folder')

# Get input file names from config
input_data_file = project_config.get('input_files', 'input_data_file')
feature_config_file = project_config.get('input_files', 'feature_config_file')

# Step 2: Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
input_data_path = os.path.join(
    script_dir, input_data_folder, input_data_file)
data = pd.read_csv(input_data_path)

# Step 3: Define Variable Types Using Feature Configuration
feature_config_path = os.path.join(
    script_dir, input_data_folder, feature_config_file)
feature_config_reader = FeatureConfigurationReader(
    config_source='csv', feature_config=None, db_params={'csv_path': feature_config_path})

# Step 4: Exploratory Data Analysis (EDA) - Raw Data
# Choose between 'sweetviz' and 'custom'
eda_tool = project_config.get('eda', 'eda_tool', fallback='sweetviz')

raw_report_output_path = os.path.join(
    script_dir, report_output_folder, 'raw_sweetviz_report.html')

if eda_tool.lower() == 'sweetviz':
    eda_service = SweetvizEDA(data)
    eda_service.generate_report(output_file=raw_report_output_path)

elif eda_tool.lower() == 'custom':
    eda_service = CustomEDAAnalysis(data)
    raw_report_output_json_path = os.path.join(
        script_dir, report_output_folder, 'raw_custom_eda_report.json')
    eda_service.save_summary_to_json(output_file=raw_report_output_json_path)

# Step 5: Encode Variables
encoder = DataVariableEncoder(feature_config_reader)
encoded_data = encoder.preprocess_data(data)

# Step 6: Exploratory Data Analysis (EDA) - Encoded Data
encoded_report_output_path = os.path.join(
    script_dir, report_output_folder, 'encoded_sweetviz_report.html')

if eda_tool.lower() == 'sweetviz':
    eda_service_encoded = SweetvizEDA(encoded_data)
    eda_service_encoded.generate_report(output_file=encoded_report_output_path)

elif eda_tool.lower() == 'custom':
    encoded_report_output_json_path = os.path.join(
        script_dir, report_output_folder, 'encoded_custom_eda_report.json')
    eda_service_encoded = CustomEDAAnalysis(encoded_data)
    eda_service_encoded.save_summary_to_json(
        output_file=encoded_report_output_json_path)

# Step 7: Save Encoded Data
encoded_data_output_path = os.path.join(script_dir, output_data_folder)

# Create an instance of SaveUtils with the absolute output directory path
save_utils = SaveUtils(output_dir=encoded_data_output_path)

# Use the save method to save the encoded data to CSV
save_utils.save_dataframe_to_csv(
    encoded_data, filename='encoded_data.csv', overwrite=True)

print("Bank Marketing Analysis Completed.")
