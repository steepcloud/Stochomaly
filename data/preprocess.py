import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data():
    """Load dataset from a file or database."""
    # This is just a placeholder for actual data loading
    # In your case, it can be reading a CSV file, database, or a custom data loader
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example XOR dataset
    y = np.array([[0], [1], [1], [0]])  # Example XOR output
    return X, y


def preprocess_data(X, y, scaler_type='standard', test_size=0.25, random_state=42):
    """Preprocess the dataset: split, scale, and shuffle."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scaling/Normalization
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler type. Use 'standard' or 'minmax'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def encode_labels(y, num_classes=2):
    """One-hot encode the labels."""
    # Assuming binary classification (can be expanded for multi-class)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.flatten()] = 1
    return one_hot
