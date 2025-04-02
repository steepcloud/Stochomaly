import numpy as np
import pickle
from nn_core.neural_network import NeuralNetwork
from plot_utils import plot_loss

class Trainer:
    def __init__(self, input_size=2, hidden_size=4, output_size=1,
                 activation="relu", optimizer="adam", learning_rate=0.01,
                 weight_decay=0.0, momentum=0.9, dropout_rate=0.0):
        """Initialize the trainer with hyperparameters."""
        self.nn = NeuralNetwork(input_size, hidden_size, output_size,
                                activation=activation, optimizer=optimizer,
                                learning_rate=learning_rate, weight_decay=weight_decay,
                                momentum=momentum, dropout_rate=dropout_rate)

    def train(self, X, y, epochs=1000, batch_size=1, save_plot=True, save_model_path=None):
        """Trains the neural network with mini-batches and optionally saves a loss plot and model."""
        loss_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]  # Shuffle data

            epoch_losses = []
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Call `train` method from `neural_network.py` (one batch at a time)
                loss = self.nn.train(X_batch, y_batch, return_loss=True)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        if save_plot:
            plot_loss(loss_history, self.nn.optimizer, self.nn.activation)

        if save_model_path:
            self.save_model(save_model_path)

        return loss_history

    def predict(self, X):
        """Predicts using the trained neural network."""
        X = np.array(X)
        predictions = self.nn.forward(X)

        if np.isscalar(predictions):
            predictions = np.array([predictions])

        return predictions

    def save_model(self, filepath):
        """Saves the trained model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.nn, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Loads a saved model from a file."""
        with open(filepath, 'rb') as f:
            self.nn = pickle.load(f)
        print(f"Model loaded from {filepath}")