"""Clustering and downstream classification helpers."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import KNeighborsClassifier


def cluster_accuracy(y_true, y_pred):
    """Match arbitrary cluster IDs to true labels and return accuracy."""
    table = contingency_matrix(y_true, y_pred)
    rows, cols = linear_sum_assignment(-table)
    return table[rows, cols].sum() / len(y_true)


def cluster_metrics(X, y, n_clusters, seed=42):
    """Fit K-means and return labels, centers and quality metrics."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(X)
    labels = model.labels_
    return {
        "labels": labels,
        "centers": model.cluster_centers_,
        "inertia": float(model.inertia_),
        "accuracy": float(cluster_accuracy(y, labels)),
        "silhouette": float(silhouette_score(X, labels)) if n_clusters > 1 else 0.0,
        "ari": float(adjusted_rand_score(y, labels)),
    }


def knn_accuracy(X_train, y_train, X_test, y_test, k=5):
    """Accuracy of a k-NN classifier on a reduced representation."""
    model = KNeighborsClassifier(n_neighbors=min(k, len(X_train)))
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))
