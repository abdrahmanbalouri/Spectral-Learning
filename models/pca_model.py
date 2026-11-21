"""PCA implemented from scratch with NumPy."""

import numpy as np

from utils.matrix_operations import check_matrix, component_count


class PCA:
    def __init__(self, n_components=0.95):
        self.n_components = n_components

    def fit(self, X):
        """Center data and find the eigenvectors of its covariance matrix."""
        X = check_matrix(X)
        if len(X) < 2:
            raise ValueError("PCA needs at least two rows")

        self.mean_ = np.mean(X, axis=0)
        centered = X - self.mean_
        covariance = (centered.T @ centered) / (len(X) - 1)

        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(np.real(eigenvalues[order]), 0, None)
        eigenvectors = np.real(eigenvectors[:, order])

        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
        self.n_components_ = component_count(
            self.explained_variance_ratio_, self.n_components
        )
        self.components_ = eigenvectors[:, : self.n_components_].T
        return self

    def transform(self, X):
        """Project data onto the selected principal components."""
        return (check_matrix(X) - self.mean_) @ self.components_.T

    def inverse_transform(self, reduced_X):
        """Reconstruct data from principal components."""
        return np.asarray(reduced_X) @ self.components_ + self.mean_
