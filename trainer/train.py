import numpy as np
import pickle
from nn_core.neural_network import NeuralNetwork
from nn_core.losses import mse_loss
from plot_utils import plot_loss
from nn_core.schedulers import StepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR



class Trainer:
    def __init__(self, input_size=2, hidden_size=4, output_size=1,
                 activation="relu", optimizer="adam", learning_rate=0.01,
                 weight_decay=0.0, momentum=0.9, dropout_rate=0.0, use_batch_norm=False,
                 early_stopping_patience=10, early_stopping_min_improvement=0.001,
                 scheduler_type=None, scheduler_params=None, use_bayesian=False, kl_weight=1.0):
        """Initialize the trainer with hyperparameters."""
        self.nn = NeuralNetwork(input_size, hidden_size, output_size,
                                activation=activation, optimizer=optimizer,
                                learning_rate=learning_rate, weight_decay=weight_decay,
                                momentum=momentum, dropout_rate=dropout_rate,
                                use_batch_norm=use_batch_norm, use_bayesian=use_bayesian)

        self.use_bayesian = use_bayesian
        self.kl_weight = kl_weight

        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params if scheduler_params else {}

        if self.scheduler_type == "StepLR":
            self.scheduler = StepLR(self.nn.optimizer, **self.scheduler_params)
        elif self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.nn.optimizer, **self.scheduler_params)
        elif self.scheduler_type == "ExponentialLR":
            self.scheduler = ExponentialLR(self.nn.optimizer, **self.scheduler_params)
        elif self.scheduler_type == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(self.nn.optimizer, **self.scheduler_params)
        else:
            self.scheduler = None

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_improvement = early_stopping_min_improvement
        self.best_val_loss = float("inf")
        self.epochs_since_improvement = 0
        self.stopped_early = False

    def compute_kl_divergence(self):
        """Compute the KL divergence for all Bayesian layers."""
        if not self.use_bayesian:
            return 0.0

        kl_div = 0.0
        if hasattr(self.nn, 'input_hidden_layer'):
            kl_div += self.nn.input_hidden_layer.kl_divergence()
        if hasattr(self.nn, 'hidden_output_layer'):
            kl_div += self.nn.hidden_output_layer.kl_divergence()

        return kl_div

    def train(self, X, y, X_val=None, y_val=None, epochs=1000, batch_size=1, save_plot=True, save_model_path=None,
              n_samples=1):
        """Trains the neural network with mini-batches and optionally saves a loss plot and model."""
        loss_history = []
        total_batches = len(X) // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(len(X))
            X, y = X[indices], y[indices]  # Shuffle data

            epoch_losses = []
            epoch_kl_losses = []

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # For Bayesian NN, we need to compute ELBO loss
                # ELBO = log p(y|x, w) - KL(q(w)||p(w))
                if self.use_bayesian:
                    # forward pass (multiple samples for MC dropout if needed)
                    batch_predictions = []
                    for _ in range(n_samples):
                        pred = self.nn.forward(X_batch, training=True)
                        batch_predictions.append(pred)

                    # average predictions across samples
                    avg_pred = np.mean(batch_predictions, axis=0)

                    # compute MSE loss
                    mse = mse_loss(y_batch, avg_pred)

                    # compute KL divergence
                    kl_div = self.compute_kl_divergence()

                    # Scale KL by batch size / dataset size (for proper ELBO calculation)
                    kl_weight = self.kl_weight * (len(X_batch) / len(X))

                    # Total loss = MSE + KL_weight * KL
                    total_loss = mse + kl_weight * kl_div

                    # Backward pass
                    output_error = avg_pred - y_batch
                    self.nn.backward(X_batch, y_batch)

                    epoch_losses.append(total_loss)
                    epoch_kl_losses.append(kl_div)
                else:
                    # Call `train` method from `neural_network.py` (one batch at a time)
                    loss = self.nn.train(X_batch, y_batch, return_loss=True)
                    epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)

            if self.use_bayesian and epoch % 100 == 0:
                avg_kl = np.mean(epoch_kl_losses)
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, KL Div: {avg_kl:.4f}")

            # Validation loss check for early stopping
            if X_val is not None and y_val is not None:
                if self.use_bayesian:
                    # for Bayesian NN, use multiple forward passes and average
                    val_predictions = []
                    for _ in range(n_samples):
                        pred = self.nn.predict(X_val)
                        val_predictions.append(pred)

                    val_pred_avg = np.mean(val_predictions, axis=0)
                    val_loss = mse_loss(y_val, val_pred_avg)

                    # add KL term to validation loss for consistent comparison
                    val_kl = self.compute_kl_divergence() * (len(X_val) / len(X))
                    val_total_loss = val_loss + self.kl_weight * val_kl
                else:
                    val_predictions = self.nn.predict(X_val)
                    val_total_loss = mse_loss(y_val, val_predictions)

                print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_total_loss:.4f}")

                # Early stopping check
                if val_total_loss < self.best_val_loss - self.early_stopping_min_improvement:
                    self.best_val_loss = val_total_loss
                    self.epochs_since_improvement = 0  # Reset count if improvement
                else:
                    self.epochs_since_improvement += 1

                if self.epochs_since_improvement >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    self.stopped_early = True
                    break  # Stop training if no improvement for 'patience' epochs
            else:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

            if self.scheduler:
                if isinstance(self.scheduler, StepLR) or isinstance(self.scheduler, ExponentialLR) or \
                        isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()  # Update learning rate on each epoch
                elif isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(avg_loss)  # Update learning rate based on validation loss

                print(f"Epoch {epoch}/{epochs}, Learning Rate: {self.nn.optimizer.learning_rate:.6f}")

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")

        if save_plot:
            plot_loss(loss_history, self.nn.optimizer, self.nn.activation)

        if save_model_path:
            self.save_model(save_model_path)

        return loss_history

    def predict(self, X, n_samples=10):
        """Predicts using the trained neural network. For Bayesian NN, averages over multiple forward passes."""
        X = np.array(X)

        if self.use_bayesian:
            # Monte Carlo sampling for uncertainty estimation
            predictions = []
            for _ in range(n_samples):
                pred = self.nn.forward(X, training=False)
                predictions.append(pred)

            mean_pred = np.mean(predictions, axis=0)

            # optionally compute standard deviation for uncertainty
            std_pred = np.std(predictions, axis=0)

            return mean_pred, std_pred
        else:
            predictions = self.nn.forward(X, training=False)

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