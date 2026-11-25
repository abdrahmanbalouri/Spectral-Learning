"""PCA implemented from scratch with NumPy."""

import numpy as np

from utils.matrix_operations import check_matrix, component_count


class PCA:
    """Reduce data using eigenvectors of its covariance matrix.

    ``n_components`` can be a component count or a variance target such as 0.95.
    """

    def __init__(self, n_components=0.95):
        self.n_components = n_components

    def fit(self, X):
        """Fit PCA to matrix X and return the fitted model."""
        X = check_matrix(X)
        if len(X) < 2:
            raise ValueError("PCA needs at least two rows")

        self.mean_ = np.mean(X, axis=0)
        centered = X - self.mean_
        self.covariance_matrix_ = centered.T @ centered / (len(centered) - 1)

        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_matrix_)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(np.real(eigenvalues[order]), 0, None)
        eigenvectors = np.real(eigenvectors[:, order])

        total_variance = np.sum(eigenvalues)
        if total_variance == 0:
            raise ValueError("PCA needs data with non-zero variance")
        self.eigenvalues_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.n_components_ = component_count(
            self.explained_variance_ratio_, self.n_components
        )
        self.components_ = eigenvectors[:, : self.n_components_].T
        return self

    def transform(self, X):
        """Return X projected onto the selected principal components."""
        return (check_matrix(X) - self.mean_) @ self.components_.T

    def inverse_transform(self, reduced_X):
        """Return an approximate reconstruction of reduced_X."""
        return np.asarray(reduced_X) @ self.components_ + self.mean_
