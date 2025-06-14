import numpy as np

class SGD:
    """Vanilla Stochastic Gradient Descent with L2 Regularization."""
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(self, weights, gradients):
        weights -= self.learning_rate * gradients
        weights -= self.weight_decay * weights
        return weights

class Momentum:
    """Momentum-based gradient descent with L2 Regularization."""
    def __init__(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = 0

    def update(self, weights, gradients):
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        weights += self.velocity
        weights -= self.weight_decay * weights
        return weights

class RMSprop:
    """Root Mean Square Propagation with L2 Regularization."""
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.cache = 0

    def update(self, weights, gradients):
        self.cache = self.decay * self.cache + (1 - self.decay) * gradients**2
        weights -= (self.learning_rate * gradients) / (np.sqrt(self.cache) + self.epsilon)
        weights -= self.weight_decay * weights
        return weights

class Adam:
    """Adaptive Moment Estimation (Adam) with L2 Regularization."""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = dict()
        self.v = dict()
        self.t = 0

    def update(self, weights, gradients):
        param_id = id(weights)

        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(weights)
            self.v[param_id] = np.zeros_like(weights)

        self.t += 1

        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * gradients
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * np.square(gradients)

        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        weights -= self.weight_decay * weights

        return weights
