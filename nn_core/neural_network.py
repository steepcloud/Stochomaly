import numpy as np
from nn_core.activations import sigmoid, relu, leaky_relu, elu, swish, gelu
from nn_core.activations import sigmoid_derivative, relu_derivative, leaky_relu_derivative, elu_derivative, \
    swish_derivative, gelu_derivative
from nn_core.losses import mse_loss
from nn_core.optimizers import SGD, Momentum, RMSprop, Adam
from utils.initializers import xavier_initializer, he_initializer, zeros_initializer

class NeuralNetwork:
    """A neural network with customizable activation functions and optimizers."""

    def __init__(self, input_size, hidden_size, output_size, activation="relu", optimizer="sgd", learning_rate=0.01,
                 weight_decay=0.0, momentum=0.9):
        """Initialize weights, biases, activation functions, and optimizers."""
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.activation = activation
        self.activation_func, self.activation_derivative = self.get_activation_pair(activation)
        self.optimizer = self.get_optimizer(optimizer, momentum)

        if activation in ["relu", "leaky_relu", "elu"]:
            weight_init = he_initializer
        else:
            weight_init = xavier_initializer

        self.weights_input_hidden = weight_init((input_size, hidden_size))
        self.weights_hidden_output = weight_init((hidden_size, output_size))
        self.bias_hidden = zeros_initializer((1, hidden_size))
        self.bias_output = zeros_initializer((1, output_size))

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

    def get_optimizer(self, name, momentum):
        """Returns the selected optimizer with proper hyperparameters."""
        optimizers = {
            "sgd": SGD(self.learning_rate, weight_decay=self.weight_decay),
            "momentum": Momentum(self.learning_rate, momentum=momentum, weight_decay=self.weight_decay),
            "rmsprop": RMSprop(self.learning_rate, weight_decay=self.weight_decay),
            "adam": Adam(self.learning_rate, weight_decay=self.weight_decay)
        }
        return optimizers.get(name, SGD(self.learning_rate))  # Default to SGD

    def forward(self, X):
        """Forward pass."""
        self.hidden_layer = self.activation_func(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output)
        return self.output_layer

    def backward(self, X, y):
        """Backward propagation with weight decay support."""

        # Compute gradients
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

        # Update parameters using optimizer
        self.weights_hidden_output = self.optimizer.update(self.weights_hidden_output, dW2)
        self.weights_input_hidden = self.optimizer.update(self.weights_input_hidden, dW1)
        self.bias_output = self.optimizer.update(self.bias_output, db2)
        self.bias_hidden = self.optimizer.update(self.bias_hidden, db1)

    def train(self, X, y, return_loss=False):
        """Perform one step of forward pass, loss computation, and backpropagation."""
        output = self.forward(X)
        loss = mse_loss(y, output)
        self.backward(X, y)

        if return_loss:
            return loss  # Return single loss value (not a list)

        return None

    def predict(self, X):
        """Predict function."""
        return self.forward(X)
