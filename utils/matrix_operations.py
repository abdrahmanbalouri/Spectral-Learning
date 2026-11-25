"""Small NumPy operations shared by PCA and SVD."""

import numpy as np


def check_matrix(X):
    """Return X as a finite, two-dimensional float array."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D matrix")
    return X


def standardize(X):
    """Return X with every feature scaled to mean 0 and deviation 1."""
    X = check_matrix(X)
    means = np.mean(X, axis=0)
    standard_deviations = np.std(X, axis=0)

    # A constant feature is already centered and should not cause division by zero.
    standard_deviations[standard_deviations == 0] = 1
    return (X - means) / standard_deviations


def component_count(ratios, target):
    """Return a fixed component count or the count needed for a variance target."""
    if isinstance(target, (int, np.integer)):
        if not 1 <= target <= len(ratios):
            raise ValueError(f"component count must be between 1 and {len(ratios)}")
        return int(target)

    if not isinstance(target, (float, np.floating)) or not 0 < target <= 1:
        raise ValueError("variance target must be between 0 and 1")
    return int(np.searchsorted(np.cumsum(ratios), target) + 1)
