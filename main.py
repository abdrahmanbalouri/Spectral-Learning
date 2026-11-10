"""Run PCA and SVD from scratch on the Wine Quality dataset."""

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
from utils.matrix_operations import n_for_variance, reconstruction_errors
from utils.visualization import (
    plot_2d,
    plot_3d,
    plot_accuracy,
    plot_cluster_panels,
    plot_cumulative,
    plot_loadings,
    plot_reconstruction,
    plot_variance_bars,
)


def parse_args():
    parser = argparse.ArgumentParser(description="PCA/SVD from scratch demo")
    parser.add_argument("--csv", default=None, help="Path to a CSV file")
    parser.add_argument("--no-download", action="store_true", help="Do not download data")
    parser.add_argument("--variance", type=float, default=0.95, help="Variance target")
    parser.add_argument("--clusters", type=int, default=None, help="Number of K-means clusters")
    parser.add_argument("--output-dir", default="outputs", help="Folder for plots and report")
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

    print("\n=== Dataset ===")
    print(f"source={info['source']} rows={n_samples} features={n_features} classes={len(classes)}")
    print(f"loaded={info['rows_loaded']} duplicates_removed={info['duplicates_removed']} "
          f"missing_imputed={info['missing_imputed']}")

    # Full decompositions: all components are available for analysis.
    pca = PCA().fit(X)
    svd = SVD().fit(X)

    k_pca = pca.select_n_components(args.variance)
    k_svd = svd.select_n_components(args.variance)
    max_k = min(pca.max_components_, svd.max_components_)
    k_show = max(1, min(max(k_pca, k_svd, 2), max_k))

    print("\n=== Variance explained ===")
    print(f"PCA: {k_pca} components reach {pca.full_ratios_[:k_pca].sum():.2%}")
    print(f"SVD: {k_svd} components reach {svd.full_ratios_[:k_svd].sum():.2%}")
    print(f"PCA vs SVD component agreement: "
          f"{np.mean(np.abs(np.sum(pca.components_[:k_show] * svd.Vt_[:k_show], axis=1))):.4f}")

    print("\n=== Top features per component (PCA) ===")
    for index in range(min(3, k_show)):
        print(f"PC{index + 1}: {', '.join(pca.top_features(index, features))}")

    # Reduce to 2-D / 3-D for visualization and clustering.
    pca_scores = pca.transform(X)
    svd_scores = svd.transform(X)
    pca_2d, svd_2d = pca_scores[:, :2], svd_scores[:, :2]

    raw_result = cluster_metrics(X, y, n_clusters, args.seed)
    pca_result = cluster_metrics(pca_2d, y, n_clusters, args.seed)
    svd_result = cluster_metrics(svd_2d, y, n_clusters, args.seed)

    print("\n=== K-means quality ===")
    for name, result in [("raw", raw_result), ("PCA-2D", pca_result), ("SVD-2D", svd_result)]:
        print(f"{name:7s} accuracy={result['accuracy']:.3f} "
              f"silhouette={result['silhouette']:.3f} "
              f"ARI={result['ari']:.3f} inertia={result['inertia']:.1f}")

    # Downstream classification: does the reduced space keep useful information?
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=args.seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=args.seed
        )

    ks = list(range(1, max_k + 1))
    pca_accuracy, svd_accuracy = [], []
    for k in ks:
        pca_k = PCA(k).fit(X_train)
        svd_k = SVD(k).fit(X_train)
        pca_accuracy.append(
            knn_accuracy(pca_k.transform(X_train), y_train, pca_k.transform(X_test), y_test)
        )
        svd_accuracy.append(
            knn_accuracy(svd_k.transform(X_train), y_train, svd_k.transform(X_test), y_test)
        )

    # Reconstruction error curves come directly from the singular values.
    pca_errors = reconstruction_errors(pca.full_singular_values_, n_samples, n_features)
    svd_errors = reconstruction_errors(svd.full_singular_values_, n_samples, n_features)

    # Figures.
    plot_cumulative(pca, svd, args.variance, out / "01_cumulative_variance.png")
    plot_variance_bars(pca, svd, k_show, out / "02_variance_comparison.png")
    plot_2d(pca_2d, y, "PCA - true labels", out / "03_pca_2d.png")
    plot_2d(svd_2d, y, "SVD - true labels", out / "04_svd_2d.png")
    if pca_scores.shape[1] >= 3:
        plot_3d(pca_scores[:, :3], y, "PCA - 3D", out / "05_pca_3d.png")
        plot_3d(svd_scores[:, :3], y, "SVD - 3D", out / "06_svd_3d.png")
    plot_cluster_panels(pca_2d, svd_2d, y, pca_result, svd_result, out / "07_cluster_patterns.png")
    plot_loadings(pca, svd, features, k_show, out / "08_loadings.png")
    plot_reconstruction(pca_errors, svd_errors, out / "09_reconstruction.png")
    plot_accuracy(ks, pca_accuracy, svd_accuracy, out / "10_knn_accuracy.png")

    report = {
        "dataset": {
            "source": info["source"],
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "classes": [int(value) for value in classes],
            "cleaning": info,
        },
        "variance": {
            "target": args.variance,
            "pca_components": int(k_pca),
            "svd_components": int(k_svd),
            "pca_cumulative": float(pca.full_ratios_[:k_pca].sum()),
            "svd_cumulative": float(svd.full_ratios_[:k_svd].sum()),
            "component_agreement": float(
                np.mean(np.abs(np.sum(pca.components_[:k_show] * svd.Vt_[:k_show], axis=1)))
            ),
        },
        "clustering": {
            name: {key: value for key, value in result.items() if key not in ("labels", "centers")}
            for name, result in [("raw", raw_result), ("pca_2d", pca_result), ("svd_2d", svd_result)]
        },
        "classification": {
            "components": ks,
            "pca_accuracy": pca_accuracy,
            "svd_accuracy": svd_accuracy,
            "best_pca": max(zip(pca_accuracy, ks)),
            "best_svd": max(zip(svd_accuracy, ks)),
        },
        "top_features": {
            f"PC{index + 1}": pca.top_features(index, features, n=5)
            for index in range(min(3, k_show))
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved plots and report to {out.resolve()}")


if __name__ == "__main__":
    main()
