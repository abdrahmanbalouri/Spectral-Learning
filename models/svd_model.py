"""SVD from scratch: A.T @ A = V S^2 V.T, then U = A V / S."""

import numpy as np

from utils.matrix_operations import (
    check_matrix,
    eig_desc,
    n_for_variance,
    variance_ratio,
)


class SVD:
    """Thin SVD of a (centered) data matrix using NumPy only.

    n_components: None (full rank), an int, or a float in (0, 1] for a
    variance target.  Set center=False to decompose the raw matrix.
    """

    def __init__(self, n_components=None, center=True):
        self.n_components = n_components
        self.center = center

    def fit(self, X):
        X = check_matrix(X)
        n_samples, n_features = X.shape

        self.mean_ = X.mean(axis=0) if self.center else np.zeros(n_features)
        A = X - self.mean_

        values, vectors = eig_desc(A.T @ A)
        values = np.clip(values, 0, None)

        self.full_ratios_ = variance_ratio(values)
        tolerance = np.finfo(float).eps * max(n_samples, n_features) * values[0]
        self.max_components_ = max(1, int((values > tolerance).sum()))

        k = self._n_components()
        self.n_components_ = k
        self.singular_values_ = np.sqrt(values[:k])
        self.Vt_ = vectors[:, :k].T.copy()

        # U = A V / S
        self.U_ = (A @ self.Vt_.T) / self.singular_values_
        self._fix_signs()

        self.components_ = self.Vt_
        self.explained_variance_ = self.singular_values_ ** 2 / (
            n_samples - 1 if self.center else n_samples
        )
        self.explained_variance_ratio_ = self.full_ratios_[:k]
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        self.full_singular_values_ = np.sqrt(values)
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
        """Make the largest loading of every right singular vector positive."""
        largest = np.argmax(np.abs(self.Vt_), axis=1)
        signs = np.sign(self.Vt_[np.arange(self.n_components_), largest])
        signs[signs == 0] = 1
        self.Vt_ *= signs[:, None]
        self.U_ *= signs[None, :]

    def transform(self, X):
        X = check_matrix(X)
        return (X - self.mean_) @ self.Vt_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, scores):
        return np.asarray(scores) @ self.Vt_ + self.mean_

    def select_n_components(self, threshold=0.95):
        return n_for_variance(self.full_ratios_, threshold)

    def top_features(self, component, names, n=3):
        indices = np.argsort(np.abs(self.Vt_[component]))[::-1][:n]
        return [names[i] for i in indices]
