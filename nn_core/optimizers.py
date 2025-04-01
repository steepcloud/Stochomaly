import numpy as np

class SGD:
    """Vanilla Stochastic Gradient Descent."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, gradients):
        return weights - self.learning_rate * gradients

class Momentum:
    """Momentum-based gradient descent."""
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = 0

    def update(self, weights, gradients):
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return weights + self.velocity

class RMSprop:
    """Root Mean Square Propagation."""
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = 0

    def update(self, weights, gradients):
        self.cache = self.decay * self.cache + (1 - self.decay) * gradients**2
        return weights - (self.learning_rate * gradients) / (np.sqrt(self.cache) + self.epsilon)

class Adam:
    """Adaptive Moment Estimation (Adam)."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, weights, gradients):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return weights - (self.learning_rate * m_hat) / (np.sqrt(v_hat) + self.epsilon)
