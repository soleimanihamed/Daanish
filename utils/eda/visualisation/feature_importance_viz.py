# utils/eda/visualisation/feature_importance_viz.py

import matplotlib.pyplot as plt
import seaborn as sns


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

    def plot_iv_scores(self, iv_df):
        """
        Plots Information Value (IV) scores with interpretation color bands.

        Parameters
        ----------
        iv_df : pd.DataFrame
            A dataframe with 'Feature' and 'IV' columns.
        """
        # Categorize IV interpretation
        def categorize_iv(iv):
            if iv < 0.02:
                return 'Not useful'
            elif iv < 0.1:
                return 'Weak'
            elif iv < 0.3:
                return 'Medium'
            elif iv < 0.5:
                return 'Strong'
            else:
                return 'Suspiciously strong'

        iv_df = iv_df.copy()
        iv_df['IV Category'] = iv_df['IV'].apply(categorize_iv)

        # Sort features
        iv_df = iv_df.sort_values(by='IV', ascending=True)

        # Set color palette
        palette = {
            'Not useful': '#d62728',
            'Weak': '#ff7f0e',
            'Medium': '#1f77b4',
            'Strong': '#2ca02c',
            'Suspiciously strong': '#9467bd'
        }

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=iv_df,
            x='IV',
            y='Feature',
            hue='IV Category',
            dodge=False,
            palette=palette
        )
        plt.title('Information Value (IV) by Feature')
        plt.xlabel('Information Value (IV)')
        plt.ylabel('Feature')
        plt.legend(title='IV Strength')
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
