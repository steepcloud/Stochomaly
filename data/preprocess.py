import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def load_data(source="xor", **kwargs):
    """Load dataset from a file or database."""
    if source == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example XOR dataset
        y = np.array([[0], [1], [1], [0]])  # Example XOR output
    elif source == "csv":
        X, y = load_csv_data(**kwargs)
    elif source == "database":
        X, y = load_db_data(**kwargs)
    elif source == "sklearn":
        X, y = load_sklearn_dataset(**kwargs)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    return X, y


def load_csv_data(filepath, features_cols=None, target_col=None, **kwargs):
    """Load dataset from a CSV file.

    Args:
        filepath: Path to the CSV file
        features_cols: List of column names to use as features (None for all except target)
        target_col: Column name to use as target
        **kwargs: Additional arguments for pd.read_csv
    """
    import pandas as pd

    # Load the data
    df = pd.read_csv(filepath, **kwargs)

    # Extract features and target
    if features_cols is None:
        if target_col:
            features_cols = [col for col in df.columns if col != target_col]
        else:
            features_cols = df.columns[:-1]
            target_col = df.columns[-1]

    X = df[features_cols].values
    y = df[target_col].values.reshape(-1, 1)

    return X, y


def load_db_data(connection_string, query, feature_cols=None, target_col=None):
    """Load dataset from a database.

    Args:
        connection_string: Database connection string
        query: SQL query to execute
        feature_cols: List of column names for features
        target_col: Column name for target
    """
    import pandas as pd
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    df = pd.read_sql(query, engine)

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    return X, y


def load_sklearn_dataset(dataset_name, return_X_y=True, **kwargs):
    """Load a dataset from scikit-learn.

    Args:
        dataset_name: Name of the dataset ('iris', 'digits', 'wine', etc.)
        return_X_y: If True, returns features and target separately
        **kwargs: Additional arguments for the dataset loader
    """
    from sklearn import datasets

    dataset_loaders = {
        'iris': datasets.load_iris,
        'digits': datasets.load_digits,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'diabetes': datasets.load_diabetes
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(dataset_loaders.keys())}")

    data = dataset_loaders[dataset_name](**kwargs)

    if return_X_y:
        X = data.data
        y = data.target.reshape(-1, 1)
        return X, y

    return data


def preprocess_data(X, y, scaler_type='standard', test_size=0.25, random_state=42):
    """Preprocess the dataset: split, scale, and shuffle."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scaling/Normalization
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Use 'standard', 'minmax', or 'robust'.")

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def encode_labels(y, num_classes=2):
    """One-hot encode the labels."""
    # Assuming binary classification (can be expanded for multi-class)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.flatten()] = 1
    return one_hot


class DataLoader:
    """Base class for data loading with batch support."""

    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.indices = np.arange(self.n_samples)
        self.current_idx = 0

        if shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_idx:
                                     min(self.current_idx + self.batch_size,
                                         self.n_samples)]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        self.current_idx += self.batch_size
        return X_batch, y_batch