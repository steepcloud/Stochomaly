import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    return x * (1 - x)

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Derivative of the Leaky ReLU function."""
    return np.where(x > 0, 1, alpha)

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Exponential Linear Unit (ELU) activation function."""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Derivative of the ELU function."""
    return np.where(x > 0, 1, alpha * np.exp(x))

def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function."""
    return x * sigmoid(x)

def swish_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the Swish function."""
    return sigmoid(x) + x * sigmoid_derivative(x)

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the GELU function."""
    return 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))