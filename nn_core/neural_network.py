import numpy as np
from nn_core.activations import sigmoid, relu, leaky_relu, elu, swish, gelu
from nn_core.activations import sigmoid_derivative, relu_derivative, leaky_relu_derivative, elu_derivative, \
    swish_derivative, gelu_derivative
from nn_core.losses import mse_loss
from nn_core.optimizers import SGD, Momentum, RMSprop, Adam
from nn_core.layers import BatchNorm
from nn_core.bayesian_layers import BayesianLinear
from utils.initializers import xavier_initializer, he_initializer, zeros_initializer

class NeuralNetwork:
    """A neural network with customizable activation functions and optimizers."""

    def __init__(self, input_size, hidden_size, output_size, activation="relu", output_activation="sigmoid",
                 optimizer="sgd", learning_rate=0.01, weight_decay=0.0, momentum=0.9, dropout_rate=0.0,
                 use_batch_norm=False, use_bayesian=False):
        """Initialize weights, biases, activation functions, optimizers, batch normalization layers and optional Bayesian layers."""
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.output_activation = output_activation
        self.activation_func, self.activation_derivative = self.get_activation_pair(activation)
        self.optimizer = self.get_optimizer(optimizer, momentum)
        self.dropout_rate = dropout_rate
        self.use_bayesian = use_bayesian

        if activation in ["relu", "leaky_relu", "elu"]:
            weight_init = he_initializer
        else:
            weight_init = xavier_initializer

        if self.use_bayesian:
            self.input_hidden_layer = BayesianLinear(input_size, hidden_size)
            self.hidden_output_layer = BayesianLinear(hidden_size, output_size)
        else:
            self.weights_input_hidden = weight_init((input_size, hidden_size))
            self.weights_hidden_output = weight_init((hidden_size, output_size))
            self.bias_hidden = zeros_initializer((1, hidden_size))
            self.bias_output = zeros_initializer((1, output_size))

        if self.use_batch_norm:
            self.bn_hidden = BatchNorm(hidden_size)

    def get_params(self):
        """Return all parameters of the network as a dictionary."""
        params = {}

        if self.use_bayesian:
            # Bayesian parameters
            params['input_hidden'] = {
                'W_mu': self.input_hidden_layer.W_mu.copy(),
                'W_rho': self.input_hidden_layer.W_rho.copy(),
                'b_mu': self.input_hidden_layer.b_mu.copy(),
                'b_rho': self.input_hidden_layer.b_rho.copy()
            }

            params['hidden_output'] = {
                'W_mu': self.hidden_output_layer.W_mu.copy(),
                'W_rho': self.hidden_output_layer.W_rho.copy(),
                'b_mu': self.hidden_output_layer.b_mu.copy(),
                'b_rho': self.hidden_output_layer.b_rho.copy()
            }
        else:
            # standard network parameters
            params['weights_input_hidden'] = self.weights_input_hidden.copy()
            params['weights_hidden_output'] = self.weights_hidden_output.copy()
            params['bias_hidden'] = self.bias_hidden.copy()
            params['bias_output'] = self.bias_output.copy()

        # batch normalization parameters if used
        if self.use_batch_norm:
            params['bn_gamma'] = self.bn_hidden.gamma.copy()
            params['bn_beta'] = self.bn_hidden.beta.copy()
            params['bn_running_mean'] = self.bn_hidden.running_mean.copy()
            params['bn_running_var'] = self.bn_hidden.running_var.copy()

        return params

    def set_params(self, params):
        """Set all parameters from a dictionary."""
        if self.use_bayesian:
            # Bayesian parameters
            self.input_hidden_layer.W_mu = params['input_hidden']['W_mu'].copy()
            self.input_hidden_layer.W_rho = params['input_hidden']['W_rho'].copy()
            self.input_hidden_layer.b_mu = params['input_hidden']['b_mu'].copy()
            self.input_hidden_layer.b_rho = params['input_hidden']['b_rho'].copy()

            self.hidden_output_layer.W_mu = params['hidden_output']['W_mu'].copy()
            self.hidden_output_layer.W_rho = params['hidden_output']['W_rho'].copy()
            self.hidden_output_layer.b_mu = params['hidden_output']['b_mu'].copy()
            self.hidden_output_layer.b_rho = params['hidden_output']['b_rho'].copy()
        else:
            # standard network parameters
            self.weights_input_hidden = params['weights_input_hidden'].copy()
            self.weights_hidden_output = params['weights_hidden_output'].copy()
            self.bias_hidden = params['bias_hidden'].copy()
            self.bias_output = params['bias_output'].copy()

        # batch normalization parameters if used
        if self.use_batch_norm and 'bn_gamma' in params:
            self.bn_hidden.gamma = params['bn_gamma'].copy()
            self.bn_hidden.beta = params['bn_beta'].copy()
            self.bn_hidden.running_mean = params['bn_running_mean'].copy()
            self.bn_hidden.running_var = params['bn_running_var'].copy()

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

    def forward(self, X, training=True):
        """Forward pass with dropout support and optional batch normalization."""
        if self.use_bayesian:
            self.hidden_layer = self.input_hidden_layer.forward(X)
            self.output_layer = self.hidden_output_layer.forward(self.hidden_layer)
        else:
            self.z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden

            if self.use_batch_norm:
                self.z_hidden, self.bn_cache = self.bn_hidden.forward(self.z_hidden, training)

            self.hidden_layer = self.activation_func(self.z_hidden)

            if self.dropout_rate > 0 and training:
                self.dropout_mask = np.random.rand(*self.hidden_layer.shape) < (1 - self.dropout_rate)
                self.hidden_layer *= self.dropout_mask

            self.z_output = np.dot(self.hidden_layer, self.weights_hidden_output) + self.bias_output

            if self.output_activation == "sigmoid":
                self.output_layer = sigmoid(self.z_output)
            else:
                self.output_layer = self.z_output

        return self.output_layer

    def backward(self, X, y):
        """Backward propagation with weight decay support and optional batch normalization."""

        if self.use_bayesian:
            output_error = self.output_layer - y
            output_delta = output_error # this can change depending on the model choice

            hidden_delta = self.hidden_output_layer.backward(X=self.hidden_layer, grad_output=output_delta)
            gradient_to_propagate = hidden_delta[0]
            self.input_hidden_layer.backward(X, gradient_to_propagate)
        else:
            # Compute gradients
            output_error = self.output_layer - y

            if self.output_activation == "sigmoid":
                output_delta = output_error * (self.output_layer * (1 - self.output_layer))
            else:
                # linear output derivative is just 1
                output_delta = output_error

            # Hidden layer gradients
            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)

            if self.use_batch_norm:
                hidden_delta, dgamma, dbeta = self.bn_hidden.backward(hidden_error, self.bn_cache, self.z_hidden,
                                                                      self.z_hidden.var(), self.z_hidden)
                self.bn_hidden.gamma -= self.learning_rate * dgamma
                self.bn_hidden.beta -= self.learning_rate * dbeta
            else:
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
        output = self.forward(X, training=True)
        loss = mse_loss(y, output)
        self.backward(X, y)

        if return_loss:
            return loss  # Return single loss value (not a list)

        return None

    def predict(self, X):
        """Predict function."""
        return self.forward(X, training=False)
