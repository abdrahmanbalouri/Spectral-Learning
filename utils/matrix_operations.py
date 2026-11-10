import numpy as np

def compute_covariance(X):
    """Compute covariance matrix of standardized data."""
    n_samples = X.shape[0]
    return (X.T @ X) / (n_samples - 1)

def eig_desc(cov_matrix):
    """Compute eigenvalues and eigenvectors sorted in descending order."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]

def reconstruction_errors(singular_values, n_samples, n_features):
    """Calculate reconstruction error curves based on singular values."""
    errors = []
    total_energy = np.sum(singular_values ** 2)
    for k in range(1, len(singular_values) + 1):
        retained_energy = np.sum(singular_values[:k] ** 2)
        err = 1.0 - (retained_energy / (total_energy + 1e-10))
        errors.append(max(0.0, err))
    return errors