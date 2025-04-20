import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


class VisualEDAAnalysis:
    def __init__(self, data):
        self.data = data

    def plot_histogram(self, feature):
        """
        Plots a histogram of a feature.

        Args:
            feature (str): Feature name.
        """
        sns.histplot(self.data[feature])
        plt.title(f'Histogram of {feature}')
        plt.show()

    def plot_correlation_heatmap(self):
        """
        Plots a correlation heatmap of the dataset.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_interactive_scatter(self, feature1, feature2, category):
        """
        Plots an interactive scatter plot using Plotly.

        Args:
            feature1 (str): Name of the first feature.
            feature2 (str): Name of the second feature.
            category (str): Name of the categorical feature.
        """
        fig = px.scatter(self.data, x=feature1, y=feature2, color=category)
        fig.show()
