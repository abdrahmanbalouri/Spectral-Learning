import numpy as np
from utils.matrix_operations import eig_desc


class SVD:
    def fit(self, X):
        A = np.asarray(X, dtype=float)

        m, n = A.shape

        if m >= n:
            values, V = eig_desc(A.T @ A)

            values = np.clip(values, 0, None)
            singular_values = np.sqrt(values)

            mask = singular_values > 1e-12

            self.singular_values_ = singular_values[mask]
            V = V[:, mask]

            self.U_ = (
                A @ V
            ) / self.singular_values_[None, :]

            self.Vt_ = V.T

        else:
            values, U = eig_desc(A @ A.T)

            values = np.clip(values, 0, None)
            singular_values = np.sqrt(values)

            mask = singular_values > 1e-12

            self.singular_values_ = singular_values[mask]
            U = U[:, mask]

            self.U_ = U

            self.Vt_ = (
                U.T @ A
            ) / self.singular_values_[:, None]

        self.components_ = self.Vt_

        return self

    def reconstruct(self, n_components=None):
        k = n_components or len(self.singular_values_)

        return (
            self.U_[:, :k] * self.singular_values_[None, :k]
        ) @ self.Vt_[:k, :]