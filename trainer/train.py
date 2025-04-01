import numpy as np
from nn_core.neural_network import NeuralNetwork
from plot_utils import plot_loss

class Trainer:
    def __init__(self, input_size=2, hidden_size=4, output_size=1,
                 activation="relu", optimizer="adam", learning_rate=0.01):
        """Initialize the trainer with hyperparameters."""
        self.nn = NeuralNetwork(input_size, hidden_size, output_size,
                                activation=activation, optimizer=optimizer,
                                learning_rate=learning_rate)

    def train(self, X, y, epochs=1000, save_plot=True):
        """Trains the neural network and optionally saves a loss plot."""
        loss_history = self.nn.train(X, y, epochs=epochs, return_loss=True)
        if save_plot:
            plot_loss(loss_history, self.nn.optimizer, self.nn.activation)
        return loss_history

    def predict(self, X):
        """Predicts using the trained neural network."""
        return self.nn.predict(X)
