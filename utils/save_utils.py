# utils/save_utils.py

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
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_dataframe_to_csv(self, df, filename, overwrite=True):
        """
        Saves a pandas DataFrame to a CSV file.

        Args:
            df (pandas.DataFrame): DataFrame to save.
            filename (str): Name of the CSV file to save.
            overwrite (bool): Whether to overwrite the file if it exists. Default is True.
        """
        filepath = os.path.join(self.output_dir, filename)
        try:
            if os.path.exists(filepath) and not overwrite:
                print(
                    f"File {filename} already exists. Set overwrite=True to overwrite it.")
                return

            df.to_csv(filepath, index=False)
            print(f"Data saved to {filename} successfully.")
        except Exception as e:
            print(f"Error occurred while saving data: {e}")

    def save_html_report(self, html_content, filename, overwrite=True):
        """
        Saves HTML content to an HTML file.

        Args:
            html_content (str): HTML content to save.
            filename (str): Name of the HTML file to save.
            overwrite (bool): Whether to overwrite the file if it exists. Default is True.
        """
        filepath = os.path.join(self.output_dir, filename)
        try:
            if os.path.exists(filepath) and not overwrite:
                print(
                    f"File {filename} already exists. Set overwrite=True to overwrite it.")
                return

            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(html_content)
            print(f"HTML report saved to {filename} successfully.")
        except Exception as e:
            print(f"Error occurred while saving HTML report: {e}")

    def save_json(self, data, filename, overwrite=True):
        """
        Saves data to a JSON file.

        Args:
            data (dict): Data to save in JSON format.
            filename (str): Name of the JSON file to save.
            overwrite (bool): Whether to overwrite the file if it exists. Default is True.
        """
        filepath = os.path.join(self.output_dir, filename)
        try:
            if os.path.exists(filepath) and not overwrite:
                print(
                    f"File {filename} already exists. Set overwrite=True to overwrite it.")
                return

            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            print(f"JSON data saved to {filename} successfully.")
        except Exception as e:
            print(f"Error occurred while saving JSON data: {e}")
