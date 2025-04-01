import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from trainer.train import Trainer
from plot_utils import plot_confusion_matrix, plot_roc_curve

class Evaluator:
    def __init__(self, model_path=None):
        """Initialize evaluator with optional model path."""
        self.trainer = Trainer()
        if model_path:
            self.trainer.load_model(model_path)

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance on test data."""
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        y_pred_proba = self.trainer.predict(X_test)

        if np.isscalar(y_pred_proba):
            y_pred_proba = np.array([y_pred_proba])

        # create binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        if np.isscalar(y_pred):
            y_pred = np.array([y_pred])

        # calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'mse': np.mean((y_test - y_pred_proba) ** 2)
        }

        # generate plots
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_pred_proba)

        print("\nModel Evaluation Metrics:")
        print("-" * 25)
        for metric, value in metrics.items():
            print(f"{metric.upper():10s}: {value:.4f}")

        return metrics