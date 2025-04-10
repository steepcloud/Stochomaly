import numpy as np
from feature_engineering.base import FeatureTransformer


class PCA(FeatureTransformer):
    """Principal Component Analysis implemented from scratch"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """Fit PCA to data"""
        # center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # store explained variance
        self.explained_variance = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_var

        return self

    def transform(self, X):
        """Transform data to principal components"""
        # center the data
        X_centered = X - self.mean

        # project data onto principal components
        X_transformed = np.dot(X_centered, self.components)

        return X_transformed

    def inverse_transform(self, X_transformed):
        """Reconstruct data from principal components"""
        # project back to original space
        X_reconstructed = np.dot(X_transformed, self.components.T) + self.mean

        return X_reconstructed

    def get_params(self):
        """Get parameters for saving"""
        return {
            'n_components': self.n_components,
            'components': self.components,
            'mean': self.mean,
            'explained_variance': self.explained_variance,
            'explained_variance_ratio': self.explained_variance_ratio
        }

    def set_params(self, params):
        """Set parameters when loading"""
        self.n_components = params.get('n_components', self.n_components)
        self.components = params.get('components', self.components)
        self.mean = params.get('mean', self.mean)
        self.explained_variance = params.get('explained_variance', self.explained_variance)
        self.explained_variance_ratio = params.get('explained_variance_ratio', self.explained_variance_ratio)