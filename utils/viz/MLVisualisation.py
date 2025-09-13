import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix


class MLVisualisation:
    def __init__(self, y_true, y_pred, y_prob=None, model_name="Model"):
        """
        y_true: true labels
        y_pred: predicted labels
        y_prob: predicted probabilities for positive class (optional, needed for ROC/PR)
        model_name: string, name of the model
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.model_name = model_name

    def plot_confusion_matrix(self, normalize=False):
        cm = confusion_matrix(self.y_true, self.y_pred,
                              normalize='true' if normalize else None)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True,
                    fmt='.2f' if normalize else 'd', cmap='Blues')
        plt.title(f"{self.model_name} Confusion Matrix")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def plot_roc_curve(self):
        if self.y_prob is None:
            print("Predicted probabilities are required for ROC curve.")
            return
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{self.model_name} ROC Curve")
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        if self.y_prob is None:
            print("Predicted probabilities are required for Precision-Recall curve.")
            return
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='purple', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"{self.model_name} Precision-Recall Curve")
        plt.grid(True)
        plt.show()

    def plot_predicted_probabilities(self, bins=20):
        if self.y_prob is None:
            print("Predicted probabilities are required for this plot.")
            return
        plt.figure(figsize=(6, 5))
        sns.histplot(self.y_prob[self.y_true == 0], color='green',
                     label='Non-default', bins=bins, alpha=0.6)
        sns.histplot(self.y_prob[self.y_true == 1],
                     color='red', label='Default', bins=bins, alpha=0.6)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title(f"{self.model_name} Predicted Probability Distribution")
        plt.legend()
        plt.show()
