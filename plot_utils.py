import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import os

def plot_loss(loss_history, optimizer, activation, save_path="plots/training_loss.png"):
    """Plots and saves the training loss graph."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.plot(range(len(loss_history)), loss_history, label=f"{optimizer} - {activation}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path='plots/confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path='plots/roc_curve.png'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def plot_2d_projection(X, y, title='2D Projection', save_path='plots/2d_projection.png'):
    """Plot 2D projection of features with class coloring."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8, edgecolors='w')
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(save_path)
    plt.close()


def plot_feature_comparison(X_original, X_transformed, y, methods, save_path='plots/feature_comparison.png'):
    """Compare original data with transformed features."""
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5 * (len(methods) + 1), 5))

    dim1, dim2 = 0, 1
    if X_original.shape[1] > 2:
        vars = np.var(X_original, axis=0)
        dim1, dim2 = np.argsort(vars)[-2:]

    axes[0].scatter(X_original[:, dim1], X_original[:, dim2], c=y, cmap='viridis', alpha=0.8)
    axes[0].set_title('Original Data')

    for i, (method_name, X_method) in enumerate(zip(methods, X_transformed)):
        axes[i + 1].scatter(X_method[:, 0], X_method[:, 1], c=y, cmap='viridis', alpha=0.8)
        axes[i + 1].set_title(f'{method_name}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve(rewards_history, avg_rewards_history, save_path='plots/learning_curve.png', show=True):
    """Plot and save reinforcement learning training curve.

    Args:
        rewards_history: List of rewards for each episode
        avg_rewards_history: List of average rewards (e.g., over last 100 episodes)
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label='Episode Reward')
    plt.plot(avg_rewards_history, label='Avg Reward (100 ep)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('RL Agent Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

def plot_precision_recall_curve(y_true, y_scores, save_path='plots/precision_recall_curve.png'):
    """Plot precision-recall curve.

    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities or scores
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()