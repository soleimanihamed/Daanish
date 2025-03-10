# utils/eda_sweetviz.py

from utils.save_utils import SaveUtils
import sweetviz as sv
import os


class SweetvizEDA:
    def __init__(self, data, selected_features=None):
        """
        Initializes SweetvizEDA with the dataset and selected features.

        Args:
            data (pd.DataFrame): The dataset to analyze.
            selected_features (list, optional): A list of features to include in the analysis.
        """
        if selected_features:
            self.data = data[selected_features]
        else:
            self.data = data
        self.save_utils = SaveUtils()  # Initialize SaveUtils for saving reports

    def generate_report(self, output_file='sweetviz_report.html', overwrite=True):
        """
        Generates an EDA report using Sweetviz.

        Args:
            output_file (str): Output file name (with directory).
            overwrite (bool): Whether to overwrite the file if it already exists. Default is True.
        """
        # Generate the Sweetviz report
        report = sv.analyze(self.data)

        # Save the report directly to the output file
        report.show_html(filepath=output_file, open_browser=True)

        # # Create a temporary HTML report in the same directory as output_file
        # output_dir = os.path.dirname(output_file)
        # temp_file = os.path.join(
        #     output_dir, f"temp_{os.path.basename(output_file)}")

        # # Save the report as a temporary HTML file
        # report.show_html(temp_file)

        # # Read the content from the temporary file and use SaveUtils to save it properly
        # try:
        #     with open(temp_file, 'r', encoding='utf-8') as file:
        #         html_content = file.read()
        #     self.save_utils.save_html_report(
        #         html_content, output_file, overwrite)

        # # Remove the temporary file after successful save
        #     os.remove(temp_file)

        # except Exception as e:
        #     print(f"Error while generating Sweetviz report: {e}")
        #     # Optionally, remove the temporary file if an error occurs
        #     if os.path.exists(temp_file):
        #         os.remove(temp_file)
