from sklearn.ensemble import IsolationForest
from feature_engineering.base import FeatureTransformer


class IsolationForestFeatures(FeatureTransformer):
    """Extract isolation forest scores as features for anomaly detection"""

    def __init__(self, n_estimators=100, contamination='auto', random_state=42):
        """
        Args:
            n_estimators: Number of estimators in the ensemble
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        """Fit the isolation forest model"""
        self.model.fit(X)
        return self

    def transform(self, X):
        """Extract anomaly scores as a feature
        Returns negative anomaly score as feature (higher value = more anomalous)
        """
        # decision_function returns the opposite of anomaly score, so we negate it
        scores = -self.model.decision_function(X).reshape(-1, 1)
        return scores

    def get_params(self):
        """Get parameters for saving/loading"""
        return {
            'n_estimators': self.n_estimators,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'model': self.model
        }

    def set_params(self, params):
        """Set parameters when loading"""
        if 'model' in params:
            self.model = params['model']
            self.n_estimators = params['n_estimators']
            self.contamination = params['contamination']
            self.random_state = params['random_state']