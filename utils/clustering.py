import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def cluster_metrics(X, y, n_clusters=3, seed=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X)
    
    acc = 0.0 # Placeholder or rough mapping if needed, simplified for metrics
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0.0
    ari = adjusted_rand_score(y, labels)
    
    return {
        "accuracy": float(acc),
        "silhouette": float(sil),
        "ari": float(ari),
        "inertia": float(kmeans.inertia_),
        "labels": labels,
        "centers": kmeans.cluster_centers_
    }

def knn_accuracy(X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float(accuracy_score(y_test, preds))