import unittest
import numpy as np
from feature_engineering.autoencoder import Autoencoder
from feature_engineering.manifold import UMAP
from feature_engineering.isolation_forest import IsolationForestFeatures
from feature_engineering.lof import LOFFeatures
from feature_engineering.pipeline import FeatureEngineeringPipeline


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # create synthetic data for testing
        np.random.seed(42)
        self.X = np.random.randn(100, 10)  # 100 samples, 10 features

    def test_autoencoder_initialization(self):
        ae = Autoencoder(input_dim=10, hidden_dim=8, latent_dim=2)
        self.assertEqual(ae.input_dim, 10)
        self.assertEqual(ae.hidden_dim, 8)
        self.assertEqual(ae.latent_dim, 2)

    def test_autoencoder_fit_transform(self):
        ae = Autoencoder(input_dim=10, hidden_dim=8, latent_dim=2,
                         epochs=5, batch_size=10)
        ae.fit(self.X)

        # test transform
        transformed = ae.transform(self.X)
        # check output shape
        self.assertEqual(transformed.shape, (100, 2))

        # test inverse transform
        reconstructed = ae.inverse_transform(transformed)
        # check output shape matches input
        self.assertEqual(reconstructed.shape, self.X.shape)

    def test_umap_initialization(self):
        umap = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, n_iter=10)
        self.assertEqual(umap.n_components, 2)
        self.assertEqual(umap.n_neighbors, 5)
        self.assertEqual(umap.min_dist, 0.1)

    def test_umap_fit_transform(self):
        # test UMAP with smaller dataset and fewer iterations for faster tests
        X_small = self.X[:20]  # using only 20 samples
        umap = UMAP(n_components=2, n_neighbors=5, n_iter=5)

        # test fit_transform
        embedding = umap.fit_transform(X_small)
        self.assertEqual(embedding.shape, (20, 2))

        # test transform with new data
        new_X = np.random.randn(5, 10)
        transformed = umap.transform(new_X)
        self.assertEqual(transformed.shape, (5, 2))

    def test_umap_get_set_params(self):
        # test parameter saving and loading
        umap = UMAP(n_components=3, n_neighbors=10)
        umap.fit(self.X[:20])

        # get parameters
        params = umap.get_params()
        self.assertEqual(params['n_components'], 3)
        self.assertEqual(params['n_neighbors'], 10)

        # create new instance and set parameters
        new_umap = UMAP()
        new_umap.set_params(params)
        self.assertEqual(new_umap.n_components, 3)
        self.assertEqual(new_umap.n_neighbors, 10)
        self.assertIsNotNone(new_umap.embedding_)

    def test_distance_computation(self):
        # test pairwise distance computation
        umap = UMAP()
        X_small = np.array([[0, 0], [1, 0], [0, 1]])
        distances = umap._compute_pairwise_distances(X_small)

        # expected distances: [[0, 1, 1], [1, 0, 2], [1, 2, 0]]
        expected = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]])
        np.testing.assert_almost_equal(distances, expected)

    def test_isolation_forest_features(self):
        # test IsolationForest feature extractor
        if_extractor = IsolationForestFeatures(n_estimators=10)
        if_extractor.fit(self.X)

        # transform data
        scores = if_extractor.transform(self.X)

        # check output shape (should be n_samples x 1)
        self.assertEqual(scores.shape, (100, 1))

        # check if values are in expected range for anomaly scores
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_lof_features(self):
        # test LOF feature extractor
        lof_extractor = LOFFeatures(n_neighbors=5, novelty=True)
        lof_extractor.fit(self.X)

        # transform data
        scores = lof_extractor.transform(self.X)

        # check output shape (should be n_samples x 1)
        self.assertEqual(scores.shape, (100, 1))

        # check if values are in expected range
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_feature_engineering_pipeline(self):
        # test the complete pipeline
        pipeline = FeatureEngineeringPipeline(
            autoencoder_params={
                'input_dim': 10,
                'hidden_dim': 8,
                'latent_dim': 3,
                'epochs': 2
            },
            if_params={
                'n_estimators': 10
            },
            lof_params={
                'n_neighbors': 5
            }
        )

        # fit pipeline
        pipeline.fit(self.X)

        # transform data
        transformed = pipeline.transform(self.X)

        # expected shape: original features (10) + autoencoder latent (3) + IF (1) + LOF (1)
        expected_features = 10 + 3 + 1 + 1
        self.assertEqual(transformed.shape, (100, expected_features))

        # check all transformers were created
        self.assertIn('autoencoder', pipeline.transformers)
        self.assertIn('isolation_forest', pipeline.transformers)
        self.assertIn('lof', pipeline.transformers)


if __name__ == '__main__':
    unittest.main()