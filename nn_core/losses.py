import numpy as np
from numpy import floating
from typing import Any


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    """Mean Squared Error loss function."""
    return np.mean((y_true - y_pred) ** 2)
