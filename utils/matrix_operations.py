"""Small NumPy helpers shared by PCA and SVD."""

import numpy as np


def check_matrix(X):
    """Return X as a finite 2-D float array."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.size == 0:
        raise ValueError("X must be a non-empty 2-D matrix")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def standardize(X):
    """Center each column and scale it to unit variance."""
    X = check_matrix(X)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - X.mean(axis=0)) / std


def eig_desc(A):
    """Eigenvalues/vectors of a symmetric matrix, largest first."""
    values, vectors = np.linalg.eigh(A)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def variance_ratio(values):
    """Turn eigenvalues or squared singular values into variance ratios."""
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    total = values.sum()
    if total == 0:
        raise ValueError("Total variance is zero; the data is constant")
    return values / total


def n_for_variance(ratios, threshold=0.95):
    """Smallest number of components needed to reach threshold variance."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    return int(np.searchsorted(np.cumsum(ratios), threshold) + 1)


def reconstruction_errors(singular_values, n_samples, n_features):
    """MSE for keeping 0, 1, ..., all components (centered data)."""
    squared = np.asarray(singular_values, dtype=float) ** 2
    tail = np.cumsum(squared[::-1])[::-1]
    return np.r_[tail, 0.0] / (n_samples * n_features)
