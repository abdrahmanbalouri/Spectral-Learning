"""K-means evaluation helpers."""

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def evaluate_clusters(X, y, seed=42):
    """Return K-means labels plus silhouette and adjusted Rand scores."""
    n_clusters = len(set(y))
    if not 2 <= n_clusters < len(X):
        raise ValueError("clustering needs at least two classes and samples")

    labels = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit_predict(X)
    scores = {
        "silhouette": silhouette_score(X, labels),
        "ari": adjusted_rand_score(y, labels),
    }
    return labels, scores
