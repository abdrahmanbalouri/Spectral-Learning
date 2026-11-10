"""PCA from scratch: covariance -> eigenvalues -> eigenvectors -> projection."""

import numpy as np

from utils.matrix_operations import (
    check_matrix,
    eig_desc,
    n_for_variance,
    variance_ratio,
)


class PCA:
    """Principal Component Analysis using NumPy only.

    n_components: None (all), an int, or a float in (0, 1] for a variance target.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = check_matrix(X)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0)
        covariance = np.cov(X, rowvar=False, ddof=1)
        values, vectors = eig_desc(covariance)
        values = np.clip(values, 0, None)

        self.full_ratios_ = variance_ratio(values)
        tolerance = np.finfo(float).eps * max(n_samples, n_features) * values[0]
        self.max_components_ = max(1, int((values > tolerance).sum()))

        k = self._n_components()
        self.n_components_ = k
        self.components_ = vectors[:, :k].T.copy()
        self._fix_signs()

        self.explained_variance_ = values[:k] / (n_samples - 1)
        self.explained_variance_ratio_ = self.full_ratios_[:k]
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        self.singular_values_ = np.sqrt(values[:k] * (n_samples - 1))
        self.full_singular_values_ = np.sqrt(values * (n_samples - 1))
        return self

    def _n_components(self):
        if self.n_components is None:
            return self.max_components_
        if isinstance(self.n_components, float):
            return min(n_for_variance(self.full_ratios_, self.n_components), self.max_components_)
        if not 1 <= self.n_components <= self.max_components_:
            raise ValueError(f"n_components must be between 1 and {self.max_components_}")
        return int(self.n_components)

    def _fix_signs(self):
        """Make the largest loading of every component positive."""
        largest = np.argmax(np.abs(self.components_), axis=1)
        signs = np.sign(self.components_[np.arange(self.n_components_), largest])
        signs[signs == 0] = 1
        self.components_ *= signs[:, None]

    def transform(self, X):
        X = check_matrix(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, scores):
        return np.asarray(scores) @ self.components_ + self.mean_

    def select_n_components(self, threshold=0.95):
        return n_for_variance(self.full_ratios_, threshold)

    def top_features(self, component, names, n=3):
        indices = np.argsort(np.abs(self.components_[component]))[::-1][:n]
        return [names[i] for i in indices]
