import numpy as np
from numpy import floating
from typing import Any


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    """Mean Squared Error loss function."""
    return np.mean((y_true - y_pred) ** 2)

def mae_loss(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    """Mean Absolute Error loss function."""
    return np.mean(np.abs(y_true - y_pred))


def binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) -> floating[Any]:
    """Binary Cross-Entropy loss function."""
    epsilon = 1e-15  # small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))