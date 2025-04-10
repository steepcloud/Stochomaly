import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, average_precision_score, precision_recall_curve)
from sklearn.model_selection import KFold, StratifiedKFold
from trainer.train import Trainer
from plot_utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

class Evaluator:
    def __init__(self, model_path=None):
        """Initialize evaluator with optional model path."""
        self.trainer = Trainer()
        if model_path:
            self.trainer.load_model(model_path)

    def evaluate(self, X_test, y_test, threshold=None):
        """Evaluate model performance on test data."""
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        y_pred_proba = self.trainer.predict(X_test)

        if np.isscalar(y_pred_proba):
            y_pred_proba = np.array([y_pred_proba])

        # find optimal threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(y_test, y_pred_proba)
            print(f"Optimal threshold: {threshold:.4f}")

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
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'mse': np.mean((y_test - y_pred_proba) ** 2)
        }

        # generate plots
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_pred_proba)
        plot_precision_recall_curve(y_test, y_pred_proba)

        print("\nModel Evaluation Metrics:")
        print("-" * 25)
        for metric, value in metrics.items():
            print(f"{metric.upper():10s}: {value:.4f}")

        return metrics

    def _find_optimal_threshold(self, y_true, y_scores, metric='f1'):
        """Find the optimal threshold that maximizes the given metric"""
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            scores.append(score)

        best_score_idx = np.argmax(scores)
        return thresholds[best_score_idx]

    def cross_validate(self, X, y, n_splits=5, threshold=None):
        """Perform cross-validation for more reliable evaluation."""
        X, y = np.array(X), np.array(y)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'roc_auc': [], 'avg_precision': []
        }

        for train_idx, test_idx in kf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # train model on this fold
            self.trainer.train(X_train, y_train)

            # evaluate
            fold_metrics = self.evaluate(X_test, y_test, threshold)

            # store metrics
            for metric, value in fold_metrics.items():
                if metric in cv_metrics:
                    cv_metrics[metric].append(value)

        # calculate mean and std for each metric
        print("\nCross-Validation Results:")
        print("-" * 25)
        for metric, values in cv_metrics.items():
            print(f"{metric.upper():10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

        return cv_metrics