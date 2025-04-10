import numpy as np
from feature_engineering.autoencoder import Autoencoder
from feature_engineering.isolation_forest import IsolationForestFeatures
from feature_engineering.lof import LOFFeatures

class FeatureEngineeringPipeline:
    def __init__(self, autoencoder_params=None, if_params=None, lof_params=None):
        """Feature engineering pipeline combining multiple techniques"""

        self.autoencoder_params = autoencoder_params or {}
        self.if_params = if_params or {}
        self.lof_params = lof_params or {}

        self.transformers = {}

    def fit(self, X):
        """Fit all feature transformers"""
        input_dim = X.shape[1]

        # init and fit autoencoder
        autoencoder = Autoencoder(
            input_dim=input_dim,
            hidden_dim=self.autoencoder_params.get('hidden_dim', 32),
            latent_dim=self.autoencoder_params.get('latent_dim', 10),
            **{k: v for k, v in self.autoencoder_params.items()
               if k not in ['input_dim', 'hidden_dim', 'latent_dim']}
        )
        autoencoder.fit(X)

        # init and fit isolation forest
        if_extractor = IsolationForestFeatures(**self.if_params)
        if_extractor.fit(X)

        # init and fit LOF
        lof_extractor = LOFFeatures(**self.lof_params)
        lof_extractor.fit(X)

        self.transformers['autoencoder'] = autoencoder
        self.transformers['isolation_forest'] = if_extractor
        self.transformers['lof'] = lof_extractor

        return self

    def transform(self, X):
        """Transform data using all feature extractors"""
        # check if transformers exist
        if not self.transformers:
            raise ValueError("Pipeline has not been fitted yet")

        # extract features from each transformer
        ae_features = self.transformers['autoencoder'].transform(X)
        if_features = self.transformers['isolation_forest'].transform(X)
        lof_features = self.transformers['lof'].transform(X)

        # combine all features
        return np.hstack([X, ae_features, if_features, lof_features])