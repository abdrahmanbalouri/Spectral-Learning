"""Run the required PCA and SVD workflow."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np

from models.pca_model import PCA
from models.svd_model import SVD
from utils.clustering import evaluate_clusters
from utils.data_loader import load_data
from utils.visualization import (
    plot_2d,
    plot_3d,
    plot_clusters,
    plot_loadings,
    plot_variance,
)


def main():
    variance_target = 0.95
    output = Path("outputs")
    output.mkdir(exist_ok=True)

    X, y, features = load_data()
    pca = PCA(variance_target).fit(X)
    svd = SVD(variance_target).fit(X)
    X_pca = pca.transform(X)
    X_svd = svd.transform(X)

    n_clusters = len(np.unique(y))
    pca_labels, pca_scores = evaluate_clusters(X_pca[:, :2], y, n_clusters)
    svd_labels, svd_scores = evaluate_clusters(X_svd[:, :2], y, n_clusters)

    svd_error = np.mean((X - svd.reconstruct(X_svd)) ** 2)
    print(f"Data shape: {X.shape}")
    print(f"PCA components: {pca.n_components_}")
    print(f"SVD components: {svd.n_components_}")
    print(f"SVD reconstruction MSE: {svd_error:.4f}")
    print(f"PCA scores: {pca_scores}")
    print(f"SVD scores: {svd_scores}")

    plot_variance(
        pca.explained_variance_ratio_,
        svd.explained_variance_ratio_,
        variance_target,
        output / "variance.png",
    )
    plot_2d(X_pca, y, "PCA 2D Projection", output / "pca_2d.png")
    plot_2d(X_svd, y, "SVD 2D Projection", output / "svd_2d.png")
    if X_pca.shape[1] >= 3:
        plot_3d(X_pca, y, output / "pca_3d.png")
    plot_clusters(
        X_pca[:, :2], X_svd[:, :2], pca_labels, svd_labels,
        output / "clusters.png",
    )
    plot_loadings(pca, features, output / "pca_loadings.png")
    print(f"Plots saved in {output.resolve()}")


if __name__ == "__main__":
    main()
