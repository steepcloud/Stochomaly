import numpy as np

def xavier_initializer(shape):
    """Xavier initialization (Glorot) for weights."""
    return np.random.randn(*shape) * np.sqrt(2. / (shape[0] + shape[1]))

def he_initializer(shape):
    """He initialization for ReLU activations."""
    return np.random.randn(*shape) * np.sqrt(2. / shape[0])

def random_initializer(shape):
    """Random initialization using normal distribution."""
    return np.random.randn(*shape)

def zeros_initializer(shape):
    """Initialize weights or biases to zeros."""
    return np.zeros(shape)

def constant_initializer(shape, constant=0.1):
    """Initialize weights or biases to a constant."""
    return np.full(shape, constant)

def lecun_initializer(shape):
    """LeCun initialization for networks with tanh activation."""
    return np.random.randn(*shape) * np.sqrt(1. / shape[0])

def truncated_normal_initializer(shape, mean=0.0, std=0.05):
    """Truncated normal initialization with clipped values."""
    values = np.random.normal(mean, std, size=shape)
    return np.clip(values, mean - 2*std, mean + 2*std)

def orthogonal_initializer(shape):
    """Orthogonal initialization for RNN weights."""
    random_matrix = np.random.randn(*shape)
    if len(shape) < 2:
        return random_matrix

    # Get orthogonal matrix through QR decomposition
    q, r = np.linalg.qr(random_matrix)
    q = q * np.sign(np.diag(r))
    return q[:shape[0], :shape[1]]
