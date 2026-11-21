"""Shared NumPy operations."""

import numpy as np


def check_matrix(X):
    """Convert input to a valid numerical matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D matrix")
    return X


def standardize(X):
    """Scale each feature to mean 0 and standard deviation 1."""
    X = check_matrix(X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std


def component_count(ratios, target):
    """Choose a component count from an integer or variance target."""
    if isinstance(target, float):
        if not 0 < target <= 1:
            raise ValueError("variance target must be in (0, 1]")
        count = np.searchsorted(np.cumsum(ratios), target) + 1
        return int(min(count, len(ratios)))
    if not 1 <= target <= len(ratios):
        raise ValueError("invalid number of components")
    return int(target)
