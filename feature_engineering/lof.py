from sklearn.neighbors import LocalOutlierFactor
from feature_engineering.base import FeatureTransformer


class LOFFeatures(FeatureTransformer):
    """Extract Local Outlier Factor scores as features for anomaly detection"""

    def __init__(self, n_neighbors=20, algorithm='auto', contamination='auto', novelty=True):
        """
        Args:
            n_neighbors: Number of neighbors to consider
            algorithm: Algorithm for nearest neighbors search
            contamination: Expected proportion of outliers in the data
            novelty: Whether to use LOF for novelty detection
        """
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.contamination = contamination
        self.novelty = novelty
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            contamination=contamination,
            novelty=novelty
        )

    def fit(self, X):
        """Fit the LOF model"""
        if self.novelty:
            self.model.fit(X)
        else:
            # in non-novelty mode, LOF computes scores during fit
            self._fit_scores = -self.model.fit_predict(X)
        return self

    def transform(self, X):
        """Extract LOF scores as features
        Returns negative score as feature (higher value = more anomalous)
        """
        if self.novelty:
            # for novelty mode, we use the opposite of decision function
            scores = -self.model.decision_function(X).reshape(-1, 1)
            return scores
        else:
            # for non-novelty mode, we need to warn that this only works
            # on training data that was used in fit
            if hasattr(self, '_fit_scores'):
                return self._fit_scores.reshape(-1, 1)
            else:
                raise ValueError("In non-novelty mode, transform can only be called on training data")

    def get_params(self):
        """Get parameters for saving/loading"""
        return {
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'contamination': self.contamination,
            'novelty': self.novelty,
            'model': self.model
        }

    def set_params(self, params):
        """Set parameters when loading"""
        if 'model' in params:
            self.model = params['model']
            self.n_neighbors = params['n_neighbors']
            self.algorithm = params['algorithm']
            self.contamination = params['contamination']
            self.novelty = params['novelty']