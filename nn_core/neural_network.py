import numpy as np
from nn_core.activations import sigmoid, relu, leaky_relu, elu, swish, gelu
from nn_core.losses import mse_loss
from nn_core.optimizers import SGD, Momentum, RMSprop, Adam

class NeuralNetwork:
    """A neural network with customizable activation functions and optimizers."""

    def __init__(self, input_size, hidden_size, output_size, activation="relu", optimizer="sgd", learning_rate=0.01):
        """Initialize weights, biases, activation functions, and optimizers."""
        self.learning_rate = learning_rate
        self.activation_func = self.get_activation(activation)
        self.optimizer = self.get_optimizer(optimizer)

        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def get_activation(self, name):
        """Returns the selected activation function."""
        activations = {
            "sigmoid": sigmoid,
            "relu": relu,
            "leaky_relu": leaky_relu,
            "elu": elu,
            "swish": swish,
            "gelu": gelu
        }
        return activations.get(name, relu)  # Default to ReLU if invalid

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
        """Backward propagation with selectable optimizer."""
        output_error = self.output_layer - y
        output_delta = output_error * (self.output_layer * (1 - self.output_layer))  # Sigmoid derivative

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * (self.hidden_layer * (1 - self.hidden_layer))

        # Update weights using optimizer
        self.weights_hidden_output = self.optimizer.update(self.weights_hidden_output, np.dot(self.hidden_layer.T, output_delta))
        self.bias_output = self.optimizer.update(self.bias_output, np.sum(output_delta, axis=0, keepdims=True))
        self.weights_input_hidden = self.optimizer.update(self.weights_input_hidden, np.dot(X.T, hidden_delta))
        self.bias_hidden = self.optimizer.update(self.bias_hidden, np.sum(hidden_delta, axis=0, keepdims=True))

    def train(self, X, y, epochs=1000):
        """Train the neural network."""
        for epoch in range(epochs):
            output = self.forward(X)
            loss = mse_loss(y, output)
            self.backward(X, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """Predict function."""
        return self.forward(X)
