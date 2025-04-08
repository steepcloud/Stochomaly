# feature_engineering/autoencoder.py
import numpy as np
from feature_engineering.base import FeatureTransformer
from nn_core.neural_network import NeuralNetwork


class Autoencoder(FeatureTransformer):
    """Autoencoder for dimensionality reduction using single-hidden-layer networks"""

    def __init__(self, input_dim, hidden_dim, latent_dim, activation='relu',
                 learning_rate=0.001, epochs=100, batch_size=32, optimizer='adam'):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension for encoder and decoder
            latent_dim: Dimension of the encoded representation
            activation: Activation function to use
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            optimizer: Optimizer to use ('sgd', 'momentum', 'rmsprop', 'adam')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer

        # init networks
        self._build_networks()

    def _build_networks(self):
        """Build encoder and decoder networks"""
        # encoder: input -> hidden -> latent
        self.encoder_hidden = NeuralNetwork(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            output_size=self.latent_dim,
            activation=self.activation,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate
        )

        # decoder: latent -> hidden -> output
        self.decoder = NeuralNetwork(
            input_size=self.latent_dim,
            hidden_size=self.hidden_dim,
            output_size=self.input_dim,
            activation=self.activation,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate
        )

    def fit(self, X):
        """Train the autoencoder"""
        # creating mini-batches
        indices = np.arange(len(X))

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            epoch_loss = 0
            batches = 0

            # mini-batch training
            for i in range(0, len(X), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X[batch_indices]

                # forward pass through encoder
                encoded = self.encoder_hidden.predict(X_batch)

                # forward pass through decoder
                reconstructed = self.decoder.predict(encoded)

                # train decoder (latent -> reconstruction)
                decoder_loss = self.decoder.train(encoded, X_batch, return_loss=True)

                # train encoder (input -> latent)
                # backpropagation from reconstruction error through the full network
                encoder_loss = self.encoder_hidden.train(X_batch, encoded, return_loss=True)

                epoch_loss += decoder_loss
                batches += 1

            avg_loss = epoch_loss / batches

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        return self

    def transform(self, X):
        """Transform data to latent representation"""
        return self.encoder_hidden.predict(X)

    def inverse_transform(self, X_encoded):
        """Reconstruct original features from encoded representation"""
        return self.decoder.predict(X_encoded)

    def get_params(self):
        """Get parameters for saving/loading"""
        params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'activation': self.activation,
            'learning_rate': self.learning_rate
        }

        # adding weights if needed for saving/loading
        if hasattr(self.encoder_hidden, 'weights_input_hidden'):
            params['encoder_weights_input_hidden'] = self.encoder_hidden.weights_input_hidden.copy()
            params['encoder_weights_hidden_output'] = self.encoder_hidden.weights_hidden_output.copy()
            params['encoder_bias_hidden'] = self.encoder_hidden.bias_hidden.copy()
            params['encoder_bias_output'] = self.encoder_hidden.bias_output.copy()

            params['decoder_weights_input_hidden'] = self.decoder.weights_input_hidden.copy()
            params['decoder_weights_hidden_output'] = self.decoder.weights_hidden_output.copy()
            params['decoder_bias_hidden'] = self.decoder.bias_hidden.copy()
            params['decoder_bias_output'] = self.decoder.bias_output.copy()

        return params

    def set_params(self, params):
        """Set parameters when loading"""
        if 'encoder_weights_input_hidden' in params:
            self.encoder_hidden.weights_input_hidden = params['encoder_weights_input_hidden']
            self.encoder_hidden.weights_hidden_output = params['encoder_weights_hidden_output']
            self.encoder_hidden.bias_hidden = params['encoder_bias_hidden']
            self.encoder_hidden.bias_output = params['encoder_bias_output']

            self.decoder.weights_input_hidden = params['decoder_weights_input_hidden']
            self.decoder.weights_hidden_output = params['decoder_weights_hidden_output']
            self.decoder.bias_hidden = params['decoder_bias_hidden']
            self.decoder.bias_output = params['decoder_bias_output']