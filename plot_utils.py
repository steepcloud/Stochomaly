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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels)
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

def plot_decision_boundary(model, X, y, save_path='plots/decision_boundary.png'):
    """Plot decision boundary of binary classifier.
    
    Args:
        model: Model with predict method
        X: 2D feature matrix (or will be reduced to 2D)
        y: Target labels
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        feature_names = ['PCA Component 1', 'PCA Component 2']
    else:
        X_2d = X
        feature_names = ['Feature 1', 'Feature 2']
    
    margin = 0.5
    x_min, x_max = X_2d[:, 0].min() - margin, X_2d[:, 0].max() + margin
    y_min, y_max = X_2d[:, 1].min() - margin, X_2d[:, 1].max() + margin
    h = (x_max - x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    if X.shape[1] > 2:
        grid_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    else:
        grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model.predict(grid_points).reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                          edgecolor='k', marker='o', cmap='viridis')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('Decision Boundary')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.savefig(save_path)
    plt.close()

def plot_prediction_distribution(y_pred_proba, y_true=None, save_path='plots/prediction_distribution.png'):
    """Plot distribution of prediction probabilities.
    
    Args:
        y_pred_proba: Prediction probabilities from model
        y_true: Optional true labels for coloring
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    if y_true is not None:
        for class_val in np.unique(y_true):
            mask = y_true == class_val
            plt.hist(y_pred_proba[mask].flatten(), 
                    alpha=0.5, bins=20, 
                    label=f'Class {class_val}')
        plt.legend()
    else:
        plt.hist(y_pred_proba.flatten(), bins=20)
        
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Model Predictions')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_threshold_sensitivity(y_true, y_scores, save_path='plots/threshold_sensitivity.png'):
    """Plot metrics vs threshold to analyze sensitivity.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores or probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    possible_y_pred = [(y_scores >= t).astype(int) for t in [0.25, 0.5, 0.75]]
    all_classes= np.unique(np.concatenate([y_true] + possible_y_pred))
    thresholds = np.linspace(0.01, 0.99, 50)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, labels=all_classes, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, labels=all_classes, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, labels=all_classes, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.plot(thresholds, accuracies, 'y-', label='Accuracy')
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Classification Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

