import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from sklearn import datasets


def load_data(source="xor", return_feature_names=False, **kwargs):
    """Load dataset from a file or database."""
    feature_names = None
    X, y = None, None

    if source == "xor":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example XOR dataset
        y = np.array([[0], [1], [1], [0]])  # Example XOR output
        if return_feature_names:
            feature_names = ["feature_0", "feature_1"] # Example names for XOR
    elif source == "csv":
        csv_kwargs = {k: v for k, v in kwargs.items() if k in ['filepath', 'features_cols', 'target_col']}
        if return_feature_names:
            X, y, feature_names = load_csv_data(return_feature_names=True, **csv_kwargs)
        else:
            X, y = load_csv_data(return_feature_names=False, **csv_kwargs)
    elif source == "database":
        db_kwargs = {k:v for k,v in kwargs.items() if k in ['connection_string', 'query', 'feature_cols', 'target_col']}
        if return_feature_names:
            print("[WARNING] load_data: Feature name return not fully implemented for 'database' source.")
            X, y = load_db_data(return_feature_names=False, **db_kwargs)
        else:
            X, y = load_db_data(return_feature_names=False, **db_kwargs)
    elif source == "sklearn":
        dataset_name_to_load = kwargs.get('dataset_name', 'iris') 
        if return_feature_names:
            X, y, feature_names = load_sklearn_dataset(dataset_name=dataset_name_to_load, return_feature_names=True)
        else:
            X, y = load_sklearn_dataset(dataset_name=dataset_name_to_load, return_feature_names=False)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    if y is not None and y.ndim == 1:
        y = y.reshape(-1, 1)
        
    if return_feature_names:
        return X, y, feature_names
    else:
        return X, y


def load_csv_data(filepath, features_cols=None, target_col=None, return_feature_names=False, **kwargs):
    """Load dataset from a CSV file.

    Args:
        filepath: Path to the CSV file
        features_cols: List of column names to use as features (None for all except target)
        target_col: Column name to use as target
        **kwargs: Additional arguments for pd.read_csv
    """
    # Load the data
    df = pd.read_csv(filepath, **kwargs)
    current_feature_names = None

    # Extract features and target
    if features_cols is None:
        if target_col:
            features_cols = [col for col in df.columns if col != target_col]
        else:
            if len(df.columns) > 1:
                features_cols = df.columns[:-1]
                target_col = df.columns[-1]
            else: # single column CSV, assume it's the target, no features
                features_cols = []
                target_col = df.columns[0]

    if features_cols:
        X = df[features_cols].values
        current_feature_names = list(features_cols)
    else:
        X = np.empty((len(df), 0))
        current_feature_names = []

    y = df[target_col].values.reshape(-1, 1) if target_col in df else np.array([]).reshape(-1, 1)

    if return_feature_names:
        return X, y, current_feature_names
    else:
        return X, y


def load_db_data(connection_string, query, feature_cols=None, target_col=None, return_feature_names=False):
    """Load dataset from a database.

    Args:
        connection_string: Database connection string
        query: SQL query to execute
        feature_cols: List of column names for features
        target_col: Column name for target
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)
    df = pd.read_sql(query, engine)
    current_feature_names = None

    if feature_cols is None:
        if target_col:
            feature_cols = [col for col in df.columns if col != target_col]
        else:
            if len(df.columns) > 1:
                feature_cols = list(df.columns[:-1])
                target_col = df.columns[-1]
            else:
                feature_cols = []
                target_col = df.columns[0]

    if feature_cols:
        X = df[feature_cols].values
        current_feature_names = list(feature_cols)
    else:
        X = np.empty((len(df), 0))
        current_feature_names = []
    
    y = df[target_col].values.reshape(-1, 1) if target_col in df else np.array([]).reshape(-1,1)

    if return_feature_names:
        return X, y, current_feature_names
    else:
        return X, y


def load_sklearn_dataset(dataset_name, return_feature_names=False):
    """Load a dataset from scikit-learn.

    Args:
        dataset_name: Name of the dataset ('iris', 'digits', 'wine', etc.)
        return_feature_names: If True, also return feature names
    """
    dataset_loaders = {
        'iris': datasets.load_iris,
        'digits': datasets.load_digits,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer,
        'diabetes': datasets.load_diabetes
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(dataset_loaders.keys())}")

    data_obj = dataset_loaders[dataset_name]()

    X = data_obj.data
    y = data_obj.target

    current_feature_names = None
    if hasattr(data_obj, 'feature_names'):
        current_feature_names = list(data_obj.feature_names)
    else:
        current_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if return_feature_names:
        return X, y, current_feature_names
    else:
        return X, y


def preprocess_data(X, y, scaler_type='standard', test_size=0.25, random_state=42):
    """Preprocess the dataset: split, scale, and shuffle."""
    if y.ndim > 1 and y.shape[1] == 1:
        y_flat = y.ravel()
    else:
        y_flat = y

    n_samples = len(y_flat)
    unique_classes, class_counts = np.unique(y_flat, return_counts=True)
    n_classes = len(unique_classes)

    min_test_samples_for_stratification = n_classes

    current_test_samples_count = int(np.ceil(test_size * n_samples))

    final_test_size_param = test_size

    if isinstance(test_size, float) and (test_size <= 0.0 or test_size >= 1.0) :
        print(f"[WARNING] preprocess_data: Initial test_size proportion {test_size} is outside (0.0, 1.0). Clamping or using default.")
        test_size = 0.25
        current_test_samples_count = int(np.ceil(test_size * n_samples))

    if current_test_samples_count < min_test_samples_for_stratification:
        final_test_size_param = int(min_test_samples_for_stratification)
        print(f"[INFO] preprocess_data: Original test_size {test_size} resulted in {current_test_samples_count} test samples, "
              f"which is less than n_classes ({n_classes}). Adjusted test_size to {final_test_size_param} (absolute count) " # MODIFIED: Log message
              f"to ensure at least {min_test_samples_for_stratification} test samples.")
    elif n_samples <= min_test_samples_for_stratification:
        print(f"[WARNING] preprocess_data: Total samples ({n_samples}) is very small compared to n_classes ({n_classes}). "
              f"Stratification might be problematic or disabled. Consider a larger dataset.")
        
        if n_samples > 1: 
             final_test_size_param = 1 
        else:
            print("[ERROR] preprocess_data: Cannot split data with only 1 sample.")
            return X, np.array([]), y_flat, np.array([])
    
    can_stratify = False
    
    num_test_for_strat_check = final_test_size_param
    if isinstance(final_test_size_param, float):
        num_test_for_strat_check = int(np.ceil(final_test_size_param * n_samples))
    
    if n_classes > 1:
        if np.all(class_counts >= 1) and num_test_for_strat_check >= n_classes:
            can_stratify = True
    
    stratify_param = y_flat if can_stratify else None
    if not can_stratify and n_classes > 1:
        print(f"[INFO] preprocess_data: Stratification disabled due to small class sizes or insufficient samples for test split relative to number of classes.")

    # Split the data into training and testing sets
    X_train, X_test, y_train_flat, y_test_flat = train_test_split(
        X, y_flat, 
        test_size=final_test_size_param, 
        random_state=random_state, 
        stratify=stratify_param
    )

    # Scaling/Normalization
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type is None or scaler_type.lower() == 'none':
        scaler = None
    else:
        raise ValueError("Unsupported scaler type. Use 'standard', 'minmax', or 'robust'.")

    if scaler and X_train.shape[0] > 0:
        X_train_scaled = scaler.fit_transform(X_train)
        if X_test.shape[0] > 0:
            X_test_scaled = scaler.transform(X_test)
        else:
            X_test_scaled = np.array([])
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    return X_train_scaled, X_test_scaled, y_train_flat, y_test_flat


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