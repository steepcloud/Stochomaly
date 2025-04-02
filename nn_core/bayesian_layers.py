import numpy as np


class BayesianLinear:
    def __init__(self, in_features, out_features, prior_std=1.0):
        self.in_features = in_features
        self.out_features = out_features

        # Mean and variance of the weight distribution
        self.W_mu = np.random.randn(out_features, in_features) * 0.1
        self.W_rho = np.random.randn(out_features, in_features) * 0.1

        # Mean and variance of the bias distribution
        self.b_mu = np.zeros(out_features)
        self.b_rho = np.zeros(out_features)

        # Prior standard deviation (Gaussian prior)
        self.prior_std = prior_std

    def sample_weights(self):
        """ Sample weights using the reparameterization trick """
        W_sigma = np.log1p(np.exp(self.W_rho))  # Softplus to enforce positivity
        b_sigma = np.log1p(np.exp(self.b_rho))

        W_epsilon = np.random.randn(*self.W_mu.shape)
        b_epsilon = np.random.randn(*self.b_mu.shape)

        W_sample = self.W_mu + W_sigma * W_epsilon
        b_sample = self.b_mu + b_sigma * b_epsilon

        return W_sample, b_sample

    def forward(self, X):
        """ Forward pass with sampled weights """
        W_sample, b_sample = self.sample_weights()
        return X @ W_sample.T + b_sample

    def kl_divergence(self):
        """ Compute KL divergence between approximate posterior and prior """
        W_sigma = np.log1p(np.exp(self.W_rho))
        b_sigma = np.log1p(np.exp(self.b_rho))

        prior_var = self.prior_std ** 2
        W_kl = np.sum(np.log(prior_var) - np.log(W_sigma ** 2) + (W_sigma ** 2 + self.W_mu ** 2) / prior_var - 1)
        b_kl = np.sum(np.log(prior_var) - np.log(b_sigma ** 2) + (b_sigma ** 2 + self.b_mu ** 2) / prior_var - 1)

        return 0.5 * (W_kl + b_kl)

    def backward(self, X, grad_output):
        """
        Backpropagation for Bayesian Linear layer.
        Computes gradients with respect to the weights and biases.
        """

        # Step 1: Compute gradients for sampled weights
        W_sample, b_sample = self.sample_weights()

        # Gradient of the loss with respect to the weights and bias
        grad_W = grad_output.T @ X  # Shape: (out_features, in_features)
        grad_b = np.sum(grad_output, axis=0)  # Shape: (out_features,)

        # Step 2: Compute gradients for variational parameters (rho) and mean (mu) for weights
        W_sigma = np.log1p(np.exp(self.W_rho))
        b_sigma = np.log1p(np.exp(self.b_rho))

        # Gradients with respect to rho (log of std dev for weights)
        grad_W_rho = grad_W * (W_sample - self.W_mu) / (W_sigma ** 2)  # Gradients for log(sigma)
        grad_b_rho = grad_b * (b_sample - self.b_mu) / (b_sigma ** 2)

        # Gradients with respect to mu (mean for weights)
        grad_W_mu = grad_W * W_sigma  # Gradients for mu (mean)
        grad_b_mu = grad_b * b_sigma

        # Step 3: Return the gradients for updating the parameters
        return grad_W_mu, grad_W_rho, grad_b_mu, grad_b_rho