import numpy as np
from nn_core.activations import sigmoid, relu, leaky_relu, elu, swish, gelu
from nn_core.activations import sigmoid_derivative, relu_derivative, leaky_relu_derivative, elu_derivative, \
    swish_derivative, gelu_derivative
from nn_core.losses import mse_loss
from nn_core.optimizers import SGD, Momentum, RMSprop, Adam

class NeuralNetwork:
    """A neural network with customizable activation functions and optimizers."""

    def __init__(self, input_size, hidden_size, output_size, activation="relu", optimizer="sgd", learning_rate=0.01):
        """Initialize weights, biases, activation functions, and optimizers."""
        self.learning_rate = learning_rate
        self.activation = activation
        self.activation_func, self.activation_derivative = self.get_activation_pair(activation)
        self.optimizer = self.get_optimizer(optimizer)

        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def get_activation_pair(self, name):
        """Returns both activation function and its derivative."""
        pairs = {
            "sigmoid": (sigmoid, sigmoid_derivative),
            "relu": (relu, relu_derivative),
            "leaky_relu": (leaky_relu, leaky_relu_derivative),
            "elu": (elu, elu_derivative),
            "swish": (swish, swish_derivative),
            "gelu": (gelu, gelu_derivative)
        }
        return pairs.get(name, (relu, relu_derivative))  # Default to ReLU if invalid

    def get_optimizer(self, name):
        """Returns the selected optimizer."""
        optimizers = {
            "sgd": SGD(self.learning_rate),
            "momentum": Momentum(self.learning_rate),
            "rmsprop": RMSprop(self.learning_rate),
            "adam": Adam(self.learning_rate)
        }
        return optimizers.get(name, SGD(self.learning_rate))  # Default to SGD

    def forward(self, X):
        """Forward pass."""
        self.hidden_layer = self.activation_func(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def backward(self, X, y):
        """Backward propagation with shape-aware bias handling."""
        # Forward pass results should already be stored in self.hidden_layer and self.output_layer

        # Output layer gradients
        output_error = self.output_layer - y
        output_delta = output_error * (self.output_layer * (1 - self.output_layer))

        # Hidden layer gradients
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.activation_derivative(self.hidden_layer)

        # Calculate gradients with proper shapes
        dW2 = np.dot(self.hidden_layer.T, output_delta)  # hidden -> output weights
        dW1 = np.dot(X.T, hidden_delta)  # input -> hidden weights
        db2 = np.sum(output_delta, axis=0, keepdims=True)  # output bias
        db1 = np.sum(hidden_delta, axis=0, keepdims=True)  # hidden bias

        # Update parameters
        self.weights_hidden_output = self.optimizer.update(self.weights_hidden_output, dW2)
        self.weights_input_hidden = self.optimizer.update(self.weights_input_hidden, dW1)
        self.bias_output = self.optimizer.update(self.bias_output, db2)
        self.bias_hidden = self.optimizer.update(self.bias_hidden, db1)

    def train(self, X, y, epochs=1000, return_loss=False):
        """Train the neural network and optionally return loss history."""
        loss_history = []
        output = self.forward(X)
        loss = mse_loss(y, output)
        self.backward(X, y)
        loss_history.append(loss)

        if return_loss:
            if epochs % 100 == 0:
                print(f"Batch Loss: {loss:.4f}")

        return loss_history if return_loss else None

    def predict(self, X):
        """Predict function."""
        return self.forward(X)
