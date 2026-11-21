"""K-means evaluation helpers."""

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


def evaluate_clusters(X, y, n_clusters, seed=42):
    """Run K-means and return labels, silhouette, and ARI scores."""
    labels = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit_predict(X)
    scores = {
        "silhouette": silhouette_score(X, labels),
        "ari": adjusted_rand_score(y, labels),
    }
    return labels, scores
