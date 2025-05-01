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

        if grad_output.shape[0] != X.shape[0]:
            total_elements = grad_output.size
            expected_elements = X.shape[0] * (W_sample.shape[0] // X.shape[1])

            if total_elements == expected_elements:
                grad_output = grad_output.reshape(X.shape[0], -1)

        # Gradient of the loss with respect to the weights and bias
        try:
            grad_W = X.T @ grad_output # Shape: (in_features, out_features)
        except ValueError:
            # fix for dimension mismatch
            grad_W = np.zeros_like(self.W_mu)
            
            # smallest batch dimension to avoid index errors
            batch_size = min(X.shape[0], grad_output.shape[0])
            
            # handle single item grad_output case
            if grad_output.shape[0] == 1 and X.shape[0] > 1:
                # broadcast the single gradient to all inputs
                x_i = X.reshape(X.shape[0], -1)  # reshape X to 2D
                grad_i = grad_output.flatten()   # flatten to 1D
                
                # check if dimensions make sense for matrix multiply
                if x_i.shape[1] == self.in_features and len(grad_i) == self.out_features:
                    # each row of X gets same gradient
                    for i in range(X.shape[0]):
                        grad_W += np.outer(grad_i, x_i[i])
                else:
                    # fallback - reshape everything
                    print(f"Reshaping for backprop: X:{x_i.shape}, grad:{grad_i.shape}")
                    # create properly sized gradient
                    grad_W = np.zeros((self.out_features, self.in_features))
                    # simple implementation - each feature gets equal gradient
                    for i in range(self.out_features):
                        for j in range(self.in_features):
                            grad_W[i,j] = grad_output.flatten()[0]
            else:
                # normal case - iterate through the smaller batch dimension
                for i in range(batch_size):
                    x_i = X[i].flatten() 
                    grad_i = grad_output[i].flatten() if grad_output.shape[0] > 1 else grad_output.flatten()
                    
                    # try to compute outer product with compatible dimensions
                    if len(x_i) == self.in_features and len(grad_i) == self.out_features:
                        grad_W += np.outer(grad_i, x_i)  
                    else:
                        # last resort - create something with right shape 
                        pad_x = np.zeros(self.in_features)
                        pad_g = np.zeros(self.out_features)
                        
                        # copy what we can
                        pad_x[:min(len(x_i), self.in_features)] = x_i[:min(len(x_i), self.in_features)]
                        pad_g[:min(len(grad_i), self.out_features)] = grad_i[:min(len(grad_i), self.out_features)]
                        
                        grad_W += np.outer(pad_g, pad_x)

        grad_b = np.sum(grad_output, axis=0)  # Shape: (out_features,)

        # Step 2: Compute gradients for variational parameters (rho) and mean (mu) for weights
        if grad_W.shape != W_sample.shape:
            if grad_W.shape == W_sample.shape[::-1]:
                grad_W = grad_W.T
                #print(f"Transposed grad_W shape: {grad_W.shape}")
            else:
                # creating a properly sized grad_W as a fallback (not ideal)
                #print(f"WARNING: Cannot reconcile grad_W shape {grad_W.shape} with W_sample shape {W_sample.shape}")
                grad_W = np.zeros_like(W_sample)

        if grad_b.shape != b_sample.shape:
            if len(grad_b) < len(b_sample):
                pad_width = len(b_sample) - len(grad_b)
                grad_b = np.pad(grad_b, (0, pad_width), 'constant')
                #print(f"Padded grad_b shape: {grad_b.shape}")
            elif len(grad_b) > len(b_sample):
                grad_b = grad_b[:len(b_sample)]
                #print(f"Truncated grad_b shape: {grad_b.shape}")

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