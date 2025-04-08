class FeatureTransformer:
    """Base class for all feature engineering methods"""

    def fit(self, X):
        """Learn parameters from data"""
        raise NotImplementedError

    def transform(self, X):
        """Transform data using learned parameters"""
        raise NotImplementedError

    def fit_transform(self, X):
        """Convenience method"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Reconstruct original features if supported"""
        raise NotImplementedError

    def get_params(self):
        """Get parameters for saving/loading"""
        return {}

    def set_params(self, params):
        """Set parameters when loading"""
        pass