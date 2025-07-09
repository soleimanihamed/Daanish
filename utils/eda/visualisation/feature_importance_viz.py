# utils/eda/visualisation/feature_importance_viz.py

import matplotlib.pyplot as plt


class FeatureImportanceVisualiser:
    """
    Class for visualising feature importance metrics such as WoE trends, IV rankings, 
    model-based feature importances, etc.

    Attributes:
        woe_map (dict): A dictionary containing WoE values for each feature and bin index.
    """

    def __init__(self, woe_map=None):
        """
        Initialise the visualiser with optional WoE map.

        Parameters:
            woe_map (dict, optional): Dictionary of {feature_name: {bin_index: woe_value}}.
        """
        self.woe_map = woe_map if woe_map else {}

    def plot_woe_trend(self, feature_name):
        """
        Plots WoE values across bins for a single binned feature.

        Parameters:
            feature_name (str): Name of the binned feature.
        """
        if feature_name not in self.woe_map:
            print(f"Feature '{feature_name}' not found in WoE map.")
            return

        bin_indices = list(self.woe_map[feature_name].keys())
        woe_values = list(self.woe_map[feature_name].values())

        plt.figure(figsize=(7, 4))
        plt.bar(bin_indices, woe_values, color='steelblue', edgecolor='black')
        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.title(f"WoE per Bin - {feature_name}")
        plt.xlabel("Bin Index")
        plt.ylabel("WoE Value")
        plt.xticks(bin_indices)
        plt.tight_layout()
        plt.show()

    def plot_all_woe_trends(self):
        """
        Plots WoE bar plots for all features in the WoE map.
        """
        for feature in self.woe_map.keys():
            self.plot_woe_trend(feature)
