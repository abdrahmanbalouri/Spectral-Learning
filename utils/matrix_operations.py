"""Shared NumPy operations."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def check_matrix(X):
    """Convert input to a valid numerical matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D matrix")
    return X


def standardize(X):
    """Scale each feature to mean 0 and standard deviation 1.""" 
    return StandardScaler().fit_transform(check_matrix(X))


def component_count(ratios, target):
        return int(np.argmax(np.cumsum(ratios) >= target) + 1)
  
