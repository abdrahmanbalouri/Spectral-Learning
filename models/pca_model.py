import numpy as np
from utils.matrix_operations import eig_desc


class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        n_samples, n_features = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        covariance = (X_centered.T @ X_centered) / (n_samples - 1)

        values, vectors = eig_desc(covariance)

        values = np.clip(values, 0, None)

        k = self.n_components or n_features

        self.components_ = vectors[:, :k].T.copy()

        total = np.sum(values)

        self.explained_variance_ = values[:k]
        self.explained_variance_ratio_ = values[:k] / total
        self.cumulative_variance_ratio_ = np.cumsum(
            self.explained_variance_ratio_
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X @ self.components_ + self.mean_