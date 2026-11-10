import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from models.pca_model import PCA
from models.svd_model import SVD
from utils.clustering import cluster_metrics, knn_accuracy
from utils.data_loader import load_data
from utils.matrix_operations import reconstruction_errors
from utils.visualization import (
    plot_2d, plot_3d, plot_accuracy, plot_cluster_panels,
    plot_cumulative, plot_loadings, plot_reconstruction, plot_variance_bars
)

def parse_args():
    parser = argparse.ArgumentParser(description="PCA/SVD from scratch demo")
    parser.add_argument("--csv", default=None, help="Path to CSV file")
    parser.add_argument("--no-download", action="store_true", help="Do not download data")
    parser.add_argument("--variance", type=float, default=0.95, help="Variance target")
    parser.add_argument("--clusters", type=int, default=None, help="Number of clusters")
    parser.add_argument("--output-dir", default="outputs", help="Output folder for plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, y, features, info = load_data(args.csv, download=not args.no_download)
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_clusters = args.clusters or len(classes)
    n_clusters = max(2, min(n_clusters, max(2, n_samples - 1)))

    print("\n=== Dataset Summary ===")
    print(f"Rows: {n_samples} | Features: {n_features} | Classes: {len(classes)}")

    pca = PCA().fit(X)
    svd = SVD().fit(X)

    k_pca = pca.select_n_components(args.variance)
    k_svd = svd.select_n_components(args.variance)
    max_k = min(pca.max_components_, svd.max_components_)
    k_show = max(1, min(max(k_pca, k_svd, 2), max_k))

    print(f"PCA components for {args.variance*100}% variance: {k_pca}")
    print(f"SVD components for {args.variance*100}% variance: {k_svd}")

    pca_scores = pca.transform(X)
    svd_scores = svd.transform(X)
    pca_2d, svd_2d = pca_scores[:, :2], svd_scores[:, :2]

    raw_result = cluster_metrics(X, y, n_clusters, args.seed)
    pca_result = cluster_metrics(pca_2d, y, n_clusters, args.seed)
    svd_result = cluster_metrics(svd_2d, y, n_clusters, args.seed)

    print("\n=== K-means Results ===")
    for name, res in [("Raw", raw_result), ("PCA-2D", pca_result), ("SVD-2D", svd_result)]:
        print(f"{name:7s} | Silhouette: {res['silhouette']:.3f} | ARI: {res['ari']:.3f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=args.seed)

    ks = list(range(1, max_k + 1))
    pca_acc, svd_acc = [], []
    for k in ks:
        p_k = PCA(k).fit(X_train)
        s_k = SVD(k).fit(X_train)
        pca_acc.append(knn_accuracy(p_k.transform(X_train), y_train, p_k.transform(X_test), y_test))
        svd_acc.append(knn_accuracy(s_k.transform(X_train), y_train, s_k.transform(X_test), y_test))

    pca_errs = reconstruction_errors(pca.full_singular_values_, n_samples, n_features) if hasattr(pca, 'full_singular_values_') else []
    svd_errs = reconstruction_errors(svd.full_singular_values_, n_samples, n_features)

    # Plotting & saving
    plot_cumulative(pca, svd, args.variance, out / "01_cumulative_variance.png")
    plot_variance_bars(pca, svd, k_show, out / "02_variance_comparison.png")
    plot_2d(pca_2d, y, "PCA - 2D Projection", out / "03_pca_2d.png")
    plot_2d(svd_2d, y, "SVD - 2D Projection", out / "04_svd_2d.png")
    if pca_scores.shape[1] >= 3:
        plot_3d(pca_scores[:, :3], y, "PCA - 3D", out / "05_pca_3d.png")
        plot_3d(svd_scores[:, :3], y, "SVD - 3D", out / "06_svd_3d.png")
    plot_cluster_panels(pca_2d, svd_2d, y, pca_result, svd_result, out / "07_cluster_patterns.png")
    plot_loadings(pca, svd, features, k_show, out / "08_loadings.png")
    plot_reconstruction(pca_errs, svd_errs, out / "09_reconstruction.png")
    plot_accuracy(ks, pca_acc, svd_acc, out / "10_knn_accuracy.png")

    print(f"\nAll plots and report successfully saved in: {out.resolve()}")

if __name__ == "__main__":
    main()