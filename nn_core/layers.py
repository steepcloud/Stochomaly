import numpy as np

class BatchNorm:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Learnable parameters (initialized as identity transformation)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running mean and variance (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Compute batch mean and variance
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Normalize input
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Scale and shift
            out = self.gamma * x_normalized + self.beta

            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running statistics for inference
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out, x_normalized  # Return normalized input for backpropagation

    def backward(self, dout, x_normalized, batch_mean, batch_var, x):
        """ Backpropagate gradients through Batch Normalization """
        m = x.shape[0]

        # Gradients for gamma and beta
        dgamma = np.sum(dout * x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)

        # Gradient for normalized input
        dx_normalized = dout * self.gamma

        # Compute gradients w.r.t. input x
        dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * (batch_var + self.epsilon) ** (-1.5), axis=0)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + self.epsilon), axis=0) + dvar * np.sum(-2 * (x - batch_mean), axis=0) / m

        dx = dx_normalized / np.sqrt(batch_var + self.epsilon) + dvar * 2 * (x - batch_mean) / m + dmean / m

        return dx, dgamma, dbeta
