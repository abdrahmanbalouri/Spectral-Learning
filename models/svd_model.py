"""SVD dimensionality reduction implemented with NumPy."""

import numpy as np

from utils.matrix_operations import check_matrix, component_count


class SVD:
    """Reduce data using its largest singular values and vectors.

    ``n_components`` can be a component count or a variance target such as 0.95.
    """

    def __init__(self, n_components=0.95):
        self.n_components = n_components

    def fit(self, X):
        """Fit SVD to matrix X and return the fitted model."""
        X = check_matrix(X)
        if len(X) < 2:
            raise ValueError("SVD needs at least two rows")

        self.mean_ = np.mean(X, axis=0)
        centered = X - self.mean_
        self.U_, self.singular_values_, self.Vt_ = np.linalg.svd(
            centered, full_matrices=False
        )

        squared = self.singular_values_**2
        total_variance = np.sum(squared)
        if total_variance == 0:
            raise ValueError("SVD needs data with non-zero variance")
        self.explained_variance_ratio_ = squared / total_variance
        self.n_components_ = component_count(
            self.explained_variance_ratio_, self.n_components
        )
        self.components_ = self.Vt_[: self.n_components_, :]
        return self

    def transform(self, X):
        """Return X projected onto the selected right-singular vectors."""
        return (check_matrix(X) - self.mean_) @ self.components_.T

    def reconstruct(self, reduced_X):
        """Return an approximate reconstruction of reduced_X."""
        return np.asarray(reduced_X) @ self.components_ + self.mean_
