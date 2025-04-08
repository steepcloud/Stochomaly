import unittest
import numpy as np
from feature_engineering.base import FeatureTransformer
from feature_engineering.autoencoder import Autoencoder
from feature_engineering.manifold import UMAP


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


if __name__ == '__main__':
    unittest.main()