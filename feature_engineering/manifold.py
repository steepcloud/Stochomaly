import numpy as np
from feature_engineering.base import FeatureTransformer


class TSNE(FeatureTransformer):
    """t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation from scratch"""

    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding_ = None

    def _compute_pairwise_distances(self, X):
        """Compute squared Euclidean distance matrix"""
        sum_X = np.sum(X ** 2, axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return np.maximum(D, 0)

    def _compute_probabilities(self, D, perplexity):
        """Compute conditional probabilities p_j|i from distances"""
        n_samples = D.shape[0]
        P = np.zeros((n_samples, n_samples))
        target_entropy = np.log(perplexity)

        for i in range(n_samples):
            # set diagonal to infinity to avoid self-neighbor
            D_i = D[i].copy()
            D_i[i] = np.inf

            # binary search for sigma that gives desired perplexity
            sigma_i = 1.0
            min_sigma = 1e-20
            max_sigma = 1e20

            for _ in range(50):  # max 50 binary search steps
                P_i = np.exp(-D_i / (2 * sigma_i ** 2))
                P_i /= np.sum(P_i)

                # compute entropy
                entropy = -np.sum(P_i * np.log2(np.maximum(P_i, 1e-12)))

                # check if we've reached target entropy
                if np.abs(entropy - target_entropy) < 1e-5:
                    break

                # adjust sigma based on entropy
                if entropy < target_entropy:
                    min_sigma = sigma_i
                    sigma_i = sigma_i * 2 if max_sigma == 1e20 else (sigma_i + max_sigma) / 2
                else:
                    max_sigma = sigma_i
                    sigma_i = sigma_i / 2 if min_sigma == 1e-20 else (sigma_i + min_sigma) / 2

            P[i] = P_i

        # symmetrize and normalize
        P = (P + P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-12)

        return P

    def fit_transform(self, X):
        """Fit t-SNE and return embedded coordinates"""
        n_samples = X.shape[0]

        # compute pairwise distances and probabilities
        D = self._compute_pairwise_distances(X)
        P = self._compute_probabilities(D, self.perplexity)

        # initialize embedding randomly
        Y = np.random.normal(0, 1e-4, size=(n_samples, self.n_components))

        # gradient descent parameters
        gains = np.ones_like(Y)
        update = np.zeros_like(Y)

        # perform gradient descent
        for iter in range(self.n_iter):
            # compute Student-t distribution in embedding space
            sum_Y = np.sum(Y ** 2, axis=1)
            D_Y = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
            Q = 1 / (1 + D_Y)
            np.fill_diagonal(Q, 0)
            Q = Q / np.sum(Q)
            Q = np.maximum(Q, 1e-12)

            # compute gradient
            PQ_diff = P - Q
            for i in range(n_samples):
                grad = 4 * np.sum(np.multiply(PQ_diff[i:i + 1].T,
                                              Q[i:i + 1].T * (1 + D_Y[i:i + 1].T)) *
                                  (Y[i:i + 1] - Y), axis=0)

                # apply gradient with momentum and gains
                gains[i] = (gains[i] + 0.2) * ((grad * update[i]) <= 0) + \
                           (gains[i] * 0.8) * ((grad * update[i]) > 0)
                gains[i] = np.maximum(gains[i], 0.01)

                update[i] = 0.9 * update[i] - self.learning_rate * gains[i] * grad
                Y[i] = Y[i] + update[i]

            # center to avoid drifting
            Y = Y - np.mean(Y, axis=0)

            # report progress
            if (iter + 1) % 100 == 0:
                cost = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))
                print(f"Iteration {iter + 1}/{self.n_iter}, cost: {cost:.6f}")

        self.embedding_ = Y
        return Y

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        """Note: t-SNE doesn't support transform of new data well"""
        if self.embedding_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.embedding_

    def inverse_transform(self, X_transformed):
        raise NotImplementedError("t-SNE does not support inverse transform")


class UMAP(FeatureTransformer):
    """Simplified Uniform Manifold Approximation and Projection (UMAP)"""

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, n_iter=200):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_iter = n_iter
        self.embedding_ = None

    def _compute_pairwise_distances(self, X):
        """Compute squared Euclidean distance matrix"""
        sum_X = np.sum(X ** 2, axis=1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return np.maximum(D, 0)

    def _find_nearest_neighbors(self, X):
        """Find k nearest neighbors for each point"""
        n_samples = X.shape[0]
        D = self._compute_pairwise_distances(X)

        # for each point, find k nearest neighbors
        indices = np.zeros((n_samples, self.n_neighbors), dtype=int)
        distances = np.zeros((n_samples, self.n_neighbors))

        for i in range(n_samples):
            # sort distances and get indices of nearest neighbors
            idx = np.argsort(D[i])
            # skipping first one (self)
            indices[i] = idx[1:self.n_neighbors + 1]
            distances[i] = D[i, indices[i]]

        return indices, distances

    def _compute_graph(self, X):
        """Compute fuzzy simplicial set (graph)"""
        n_samples = X.shape[0]

        # find nearest neighbors
        indices, distances = self._find_nearest_neighbors(X)

        # compute fuzzy graph
        graph = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # compute sigma for this point (adaptive bandwidth)
            sigma = np.mean(distances[i]) * 1.0

            # set values for this point's neighbors
            for j, idx in enumerate(indices[i]):
                graph[i, idx] = np.exp(-distances[i, j] / sigma)

        # symmetrize the graph
        graph = (graph + graph.T) / 2

        return graph

    def _optimize_embedding(self, graph):
        """Optimize the low-dimensional embedding"""
        n_samples = graph.shape[0]

        # initialize embedding with random noise
        embedding = np.random.normal(scale=0.1, size=(n_samples, self.n_components))

        # parameters for optimization
        alpha = 1.0

        # creating repulsion and attraction functions based on min_dist
        a = 1.0
        b = 1.0
        if self.min_dist > 0:
            b = np.log2(2) / self.min_dist

        # optimize embedding using gradient descent
        for iteration in range(self.n_iter):
            # updating step size (learning rate)
            alpha = 1.0 - (iteration / self.n_iter)

            # computing attractive and repulsive forces
            attractive_force = np.zeros_like(embedding)
            repulsive_force = np.zeros_like(embedding)

            # computing pairwise distances in low-dimension space
            embedding_distances = self._compute_pairwise_distances(embedding)

            # computing attractive forces (based on graph)
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j and graph[i, j] > 0:
                        # attractive force
                        force = graph[i, j] * (
                            1 / (1 + a * embedding_distances[i, j] ** (2 * b))
                        )
                        direction = embedding[j] - embedding[i]
                        attractive_force[i] += force * direction

            # apply forces
            embedding = embedding + alpha * attractive_force

            # center the embedding
            embedding = embedding - np.mean(embedding, axis=0)

            if (iteration + 1) % 50 == 0:
                print(f"UMAP iteration {iteration + 1}/{self.n_iter}")

        return embedding

    def fit_transform(self, X):
        """Fit UMAP and return embedded coordinates"""

        # storing original data for transform method
        self.original_data = X.copy()

        # compute graph representation
        graph = self._compute_graph(X)

        # optimize embedding
        self.embedding_ = self._optimize_embedding(graph)

        return self.embedding_

    def fit(self, X):
        """Fit UMAP to data"""
        self.original_data = X.copy()
        self.fit_transform(X)
        return self

    def transform(self, X):
        """Transform new data by projecting into the existing embedding"""
        if self.embedding_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # store original data if not already stored
        if not hasattr(self, 'original_data'):
            raise ValueError("Original training data not stored. Refit the model.")

        n_samples = X.shape[0]
        k = min(self.n_neighbors, self.embedding_.shape[0])
        transformed = np.zeros((n_samples, self.n_components))

        for i in range(n_samples):
            # calculate distances to all points in original data
            distances = np.sum((self.original_data - X[i]) ** 2, axis=1)

            # find k nearest neighbors
            indices = np.argsort(distances)[:k]

            # calculate weights (inverse distance)
            weights = 1.0 / (distances[indices] + 1e-10)
            weights /= np.sum(weights)

            # weighted average of neighbor embeddings
            transformed[i] = np.sum(self.embedding_[indices] * weights[:, np.newaxis], axis=0)

        return transformed

    def inverse_transform(self, X_transformed):
        """UMAP does not support inverse transform"""
        raise NotImplementedError("UMAP does not support inverse transform")

    def get_params(self):
        """Get parameters for saving"""
        return {
            'n_components': self.n_components,
            'n_neighbors': self.n_neighbors,
            'min_dist': self.min_dist,
            'embedding_': self.embedding_
        }

    def set_params(self, params):
        """Set parameters when loading"""
        self.n_components = params.get('n_components', self.n_components)
        self.n_neighbors = params.get('n_neighbors', self.n_neighbors)
        self.min_dist = params.get('min_dist', self.min_dist)
        self.embedding_ = params.get('embedding_', self.embedding_)