# utils/core/save_manager.py

import os
import pandas as pd
import json


class SaveUtils:
    def __init__(self, output_dir='data/output/'):
        """
        Initializes the SaveUtils class with an output directory.

        Args:
            output_dir (str): Directory where output files will be saved.
        """
        self.output_dir = output_dir

    def save_dataframe_to_csv(self, df, full_path_name, overwrite=True, index=False):
        """
        Saves a pandas DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            full_path_name (str): Path and name of the CSV file to save.
            overwrite (bool): Whether to overwrite the file if it exists. Default is True.
        """
        try:
            if os.path.exists(full_path_name) and not overwrite:
                print(
                    f"File {full_path_name} already exists. Set overwrite=True to overwrite it.")
                return

            df.to_csv(full_path_name, index=index)
            print(f"Data saved to {full_path_name} successfully.")
        except Exception as e:
            print(f"Error occurred while saving data: {e}")

    def save_dataframe_to_excel(self, df, full_path_name, sheet_name='Sheet1', overwrite=True, index=False):
        """
        Saves a pandas DataFrame to an Excel file with the given sheet name.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            full_path_name (str): Full path and filename for the Excel file.
            sheet_name (str): Name of the sheet in the Excel file.
            overwrite (bool): Whether to overwrite the file if it exists.
        """
        try:
            if os.path.exists(full_path_name) and not overwrite:
                print(
                    f"File {full_path_name} already exists. Set overwrite=True to overwrite it.")
                return

            df.to_excel(full_path_name, sheet_name=sheet_name,
                        index=index, engine='openpyxl')
            print(f"Excel file saved to {full_path_name} successfully.")
        except Exception as e:
            print(f"Error occurred while saving Excel file: {e}")

    def save_html_report(self, html_content, full_path_name, overwrite=True):
        """
        Saves HTML content to an HTML file.

        Args:
            html_content (str): HTML content to save.
            full_path_name (str): Path and name of the HTML file to save.
            overwrite (bool): Whether to overwrite the file if it exists. Default is True.
        """

        try:
            if os.path.exists(full_path_name) and not overwrite:
                print(
                    f"File {full_path_name} already exists. Set overwrite=True to overwrite it.")
                return

            with open(full_path_name, 'w', encoding='utf-8') as file:
                file.write(html_content)
            print(f"HTML report saved to {full_path_name} successfully.")
        except Exception as e:
            print(f"Error occurred while saving HTML report: {e}")

    def save_json(self, data, full_path_name, overwrite=True):
        """
        Saves data to a JSON file.

        Args:
            data (dict): Data to save in JSON format.
            full_path_name (str): Path and name of the JSON file to save.
            overwrite (bool): Whether to overwrite the file if it exists. Default is True.
        """
        try:
            if os.path.exists(full_path_name) and not overwrite:
                print(
                    f"File {full_path_name} already exists. Set overwrite=True to overwrite it.")
                return

            with open(full_path_name, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            print(f"JSON data saved to {full_path_name} successfully.")
        except Exception as e:
            print(f"Error occurred while saving JSON data: {e}")

    def generate_styled_html_tables(self, dataframes, filenames, overwrite=True):
        """
        Generates HTML tables with styled borders and saves them to files.

        Args:
            dataframes (list of pandas.DataFrame): List of DataFrames to convert to HTML.
            filenames (list of str): List of filenames for the output HTML files.
            overwrite (bool): Whether to overwrite existing files.
        """
        if len(dataframes) != len(filenames):
            raise ValueError(
                "Number of DataFrames and filenames must be the same.")

        table_style = """
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
            }
            th, td {
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }
        </style>
        """

        for df, filename in zip(dataframes, filenames):
            html_content = df.to_html()
            styled_html = table_style + html_content
            filepath = os.path.join(self.output_dir, filename)

            try:
                if os.path.exists(filepath) and not overwrite:
                    print(
                        f"File {filename} already exists. Set overwrite=True to overwrite it.")
                else:
                    with open(filepath, "w") as f:
                        f.write(styled_html)
                    print(
                        f"Styled HTML file saved to {filename} successfully.")

            except Exception as e:
                print(
                    f"Error occurred while saving HTML table {filename}: {e}")
