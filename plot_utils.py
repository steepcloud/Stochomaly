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


def plot_confusion_matrix(y_true, y_pred, save_path='plots/confusion_matrix.png', dataset_name=None, class_label_names=None):
    """Plot confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    unique_true_labels = np.unique(y_true)
    unique_pred_labels = np.unique(y_pred)
    all_numeric_labels = np.unique(np.concatenate((unique_true_labels, unique_pred_labels))).astype(int)

    tick_labels = all_numeric_labels
    if class_label_names is not None:
        try:
            current_tick_labels = [class_label_names[i] for i in all_numeric_labels if i < len(class_label_names)]
            if len(current_tick_labels) == len(all_numeric_labels): # Ensure all labels could be mapped
                 tick_labels = current_tick_labels
            else: # Fallback if mapping is incomplete
                print(f"[Plotting Warning] CM: class_label_names ({class_label_names}) might not cover all labels present ({all_numeric_labels}). Using numeric labels.")
                tick_labels = all_numeric_labels
        except IndexError:
            print(f"[Plotting Warning] CM: IndexError with class_label_names. Using numeric labels.")
            tick_labels = all_numeric_labels

    cm = confusion_matrix(y_true, y_pred, labels=all_numeric_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=tick_labels, yticklabels=tick_labels)
    title_str = 'Confusion Matrix'
    if dataset_name:
        title_str += f' for {dataset_name.capitalize()}'
    plt.title(title_str)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_path='plots/roc_curve.png', dataset_name=None):
    """Plot ROC curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    if len(np.unique(y_true)) > 2:
        print(f"ROC curve not plotted for {dataset_name}: only one class present in y_true.")
        return
    
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
    title_str = 'Receiver Operating Characteristic'
    if dataset_name:
        title_str += f' for {dataset_name.capitalize()}'
    plt.title(title_str)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
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

def plot_learning_curve(rewards_history, avg_rewards_history, save_path='plots/learning_curve.png', dataset_name=None):
    """Plot and save reinforcement learning training curve.

    Args:
        rewards_history: List of rewards for each episode
        avg_rewards_history: List of average rewards (e.g., over last 100 episodes)
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history, label='Episode Reward')
    plt.plot(avg_rewards_history, label='Avg Reward (100 ep)', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    title_str = 'RL Agent Learning Curve'
    if dataset_name:
        title_str += f' for {dataset_name.capitalize()}'
    plt.title(title_str)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
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

def plot_prediction_distribution(y_pred_proba, y_true=None, save_path='plots/prediction_distribution.png', dataset_name=None, class_label_names=None):
    """Plot distribution of prediction probabilities.
    
    Args:
        y_pred_proba: Prediction probabilities from model
        y_true: Optional true labels for coloring
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    plt.figure(figsize=(10, 6))
    
    if y_true is not None:
        y_true = np.asarray(y_true).flatten()
        unique_true_labels = np.unique(y_true)
        for true_label_val in unique_true_labels:
            mask = (y_true == true_label_val)
            label_str = str(true_label_val)
            if class_label_names is not None and int(true_label_val) < len(class_label_names):
                label_str = class_label_names[int(true_label_val)]

            plt.hist(y_pred_proba[mask], 
                    alpha=0.5, bins=20, 
                    label=f'True: {label_str}', density=True)
        plt.legend()
    else:
        plt.hist(y_pred_proba, bins=20, density=True, alpha=0.7, label='All predictions')
        
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Prediction Probability / Score')
    plt.ylabel('Density')
    title_str = 'Distribution of Model Prediction Scores'
    if dataset_name:
        title_str += f' for {dataset_name.capitalize()}'
    plt.title(title_str)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_threshold_sensitivity(y_true, y_scores, save_path='plots/threshold_sensitivity.png', dataset_name=None):
    """Plot metrics vs threshold to analyze sensitivity.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores or probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()

    if len(np.unique(y_true)) < 2:
        print(f"Threshold sensitivity plot not generated for {dataset_name}: only one class present in y_true.")
        return

    thresholds = np.linspace(np.min(y_scores) + 1e-6, np.max(y_scores) - 1e-6, 100) # Avoid edge cases for some scores
    if len(thresholds) == 0: thresholds = np.array([0.5]) # Fallback if scores are all same

    precisions, recalls, f1s, accuracies = [], [], [], []
    
    for thresh in thresholds:
        y_pred_binary = (y_scores >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred_binary, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_binary, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_binary, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred_binary))
        
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1-score')
    plt.plot(thresholds, accuracies, label='Accuracy')
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    title_str = 'Metrics vs. Prediction Threshold'
    if dataset_name:
        title_str += f' for {dataset_name.capitalize()}'
    plt.title(title_str)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_anomaly_scatter(X, y_true_original, predictions_binary, feature_names=None, dataset_name=None, 
                         class_label_names=None, save_path='plots/anomaly_scatter.png'):
    """Plot scatter of two most important features with anomalies highlighted.
       Uses y_true_original for coloring points by their original class.
       Uses predictions_binary to mark predicted anomalies.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    X = np.asarray(X)
    y_true_original = np.asarray(y_true_original).flatten()
    predictions_binary = np.asarray(predictions_binary).flatten()
    
    if X.shape[1] < 1:
        print(f"Warning: plot_anomaly_scatter requires at least 1 feature. Found {X.shape[1]}. Skipping plot.")
        return
    if X.shape[0] == 0:
        print(f"Warning: plot_anomaly_scatter received empty X. Skipping plot.")
        return

    if X.shape[1] == 1: # Handle 1D data by plotting against an index or zeros
        X_plot = np.column_stack((X[:, 0], np.arange(len(X))))
        indices_to_plot = [0, 1]
        plot_feature_names = [feature_names[0] if feature_names else "Feature 0", "Sample Index"]
    elif X.shape[1] >= 2:
        # Determine most important features based on distinguishing predicted normal/anomaly
        if len(np.unique(predictions_binary)) > 1 and X.shape[0] > 1 : # Need at least 2 classes in predictions for ExtraTrees
            from sklearn.ensemble import ExtraTreesClassifier
            model = ExtraTreesClassifier(n_estimators=50, random_state=42, bootstrap=True, class_weight='balanced' if len(np.unique(y_true_original)) > 1 else None)
            try:
                model.fit(X, predictions_binary)
                importances = model.feature_importances_
                indices_to_plot = np.argsort(importances)[-2:] # Top 2
            except ValueError: # Handles cases like all predictions are the same class
                print("[Plotting Warning] Scatter: Could not determine feature importances (e.g., all predictions are one class). Using first two features.")
                indices_to_plot = [0, 1]
        else: # Not enough diversity in predictions or samples
            indices_to_plot = [0, 1]
        
        X_plot = X[:, indices_to_plot]
        current_feature_names_full = [f"Feature {i}" for i in range(X.shape[1])]
        if feature_names is not None and len(feature_names) == X.shape[1]:
            current_feature_names_full = feature_names
        plot_feature_names = [current_feature_names_full[indices_to_plot[0]], current_feature_names_full[indices_to_plot[1]]]

    plt.figure(figsize=(12, 8))
    
    # Plot points colored by their original true class
    unique_true_labels = np.unique(y_true_original)
    colors = plt.cm.get_cmap('viridis', len(unique_true_labels))

    for i, true_label_val in enumerate(unique_true_labels):
        rows = np.where(y_true_original == true_label_val)[0]
        if len(rows) > 0:
            label_str = str(true_label_val)
            if class_label_names is not None and int(true_label_val) >= 0 and int(true_label_val) < len(class_label_names):
                label_str = class_label_names[int(true_label_val)]
            
            plt.scatter(X_plot[rows, 0], X_plot[rows, 1], 
                        alpha=0.6, label=f"True: {label_str}", color=colors(i))
    
    # Highlight predicted anomalies with a red circle
    anomaly_predicted_rows = np.where(predictions_binary == 1)[0]
    if len(anomaly_predicted_rows) > 0:
        plt.scatter(X_plot[anomaly_predicted_rows, 0], X_plot[anomaly_predicted_rows, 1], 
                    facecolors='none', edgecolors='red', linewidth=2, 
                    s=100, label="Predicted Anomaly")
    
    plt.xlabel(plot_feature_names[0])
    plt.ylabel(plot_feature_names[1])
    title_str = "Anomaly Detection Scatter Plot"
    if dataset_name:
        title_str += f" for {dataset_name.capitalize()}"
    plt.title(title_str)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(X, predictions, feature_names=None, dataset_name=None, 
                          save_path='plots/feature_importance.png'):
    """Plot feature importance for anomaly detection.

    Args:
        X: Feature matrix
        predictions: Model predictions
        feature_names: Optional list of feature names
        dataset_name: Name of the dataset for the title
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier(n_estimators=50)
    model.fit(X, predictions)
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    indices = np.argsort(importances)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    
    title = "Feature Importance for Anomaly Detection"
    if dataset_name:
        title += f" ({dataset_name.capitalize()})"
    plt.title(title)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_anomaly_distribution(X, predictions_binary, feature_names=None, dataset_name=None,
                             y_true_original=None, class_label_names=None,
                             save_path='plots/anomaly_distribution.png'):
    """Plot distribution of features for predicted normal vs predicted anomalous points."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X = np.asarray(X)
    predictions_binary = np.asarray(predictions_binary).flatten()

    if X.shape[1] == 0 or X.shape[0] == 0:
        print(f"Anomaly distribution plot not generated for {dataset_name}: X is empty or has no features.")
        return

    current_feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    if feature_names is not None and len(feature_names) == X.shape[1]:
        current_feature_names = feature_names
    
    n_features_to_plot = min(X.shape[1], 6) # Show at most 6 features
    
    # Determine feature importance based on distinguishing predicted anomalies from predicted normals
    # if there's more than one class in predictions
    important_indices = list(range(min(n_features_to_plot, X.shape[1]))) # Default to first N features
    if len(np.unique(predictions_binary)) > 1 and X.shape[0] > 1:
        from sklearn.ensemble import ExtraTreesClassifier
        try:
            model = ExtraTreesClassifier(n_estimators=50, random_state=42, bootstrap=True, class_weight='balanced')
            model.fit(X, predictions_binary)
            importances = model.feature_importances_
            important_indices = np.argsort(importances)[::-1][:n_features_to_plot] # Top N
        except ValueError:
            print(f"[Plotting Warning] Anomaly Dist: Could not determine feature importances. Using first {n_features_to_plot} features.")
            
    fig, axes = plt.subplots(n_features_to_plot, 1, figsize=(12, 3.5 * n_features_to_plot), squeeze=False)
    axes = axes.flatten()
    
    for i, feature_idx in enumerate(important_indices):
        ax = axes[i]
        normal_predicted_rows = np.where(predictions_binary == 0)[0]
        anomaly_predicted_rows = np.where(predictions_binary == 1)[0]

        if len(normal_predicted_rows) > 0:
            sns.histplot(X[normal_predicted_rows, feature_idx], ax=ax, kde=True, label='Predicted Normal', stat="density", color="skyblue", element="step")
        if len(anomaly_predicted_rows) > 0:
            sns.histplot(X[anomaly_predicted_rows, feature_idx], ax=ax, kde=True, label='Predicted Anomaly', stat="density", color="salmon", element="step")
        
        ax.set_xlabel(current_feature_names[feature_idx])
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    title_str = "Feature Distribution: Predicted Normal vs. Predicted Anomaly"
    if dataset_name:
        title_str += f" ({dataset_name.capitalize()})"
    
    # Add original class info to title if available and y_true_original is provided
    if y_true_original is not None and class_label_names is not None:
        y_true_original = np.asarray(y_true_original).flatten()
        unique_original_labels = np.unique(y_true_original)
        if len(unique_original_labels) < 5: # Don't make title too long
            try:
                original_classes_str_list = [class_label_names[int(l)] for l in unique_original_labels if int(l) >= 0 and int(l) < len(class_label_names)]
                if original_classes_str_list:
                    original_classes_str = ", ".join(original_classes_str_list)
                    title_str += f"\n(Original True Classes in Test Set: {original_classes_str})"
            except IndexError:
                 print(f"[Plotting Warning] Anomaly Dist: IndexError with class_label_names for title.")


    plt.suptitle(title_str, y=1.0, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96 if "\n" in title_str else 0.98])
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve_custom(y_true, y_scores, save_path='plots/precision_recall_curve.png', dataset_name=None):
    """Plot precision-recall curve."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_true = np.asarray(y_true).flatten()
    y_scores = np.asarray(y_scores).flatten()

    if len(np.unique(y_true)) < 2:
        print(f"Precision-Recall curve not plotted for {dataset_name}: only one class present in y_true.")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title_str = 'Precision-Recall Curve'
    if dataset_name:
        title_str += f' for {dataset_name.capitalize()}'
    plt.title(title_str)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()