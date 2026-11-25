"""Run the required PCA and SVD workflow."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

from models import PCA, SVD
from utils import visualization
from utils.clustering import evaluate_clusters
from utils.data_loader import load_data


def main():
    variance_target = 0.95
    project_dir = Path(__file__).resolve().parent
    output = project_dir / "outputs"
    output.mkdir(exist_ok=True)

    X, y, features = load_data(project_dir / "data" / "wine_quality.csv")
    pca = PCA(variance_target).fit(X)
    svd = SVD(variance_target).fit(X)
    X_pca = pca.transform(X)
    X_svd = svd.transform(X)

    pca_labels, pca_scores = evaluate_clusters(X_pca, y)
    svd_labels, svd_scores = evaluate_clusters(X_svd, y)

    pca_error = ((X - pca.inverse_transform(X_pca)) ** 2).mean()
    svd_error = ((X - svd.reconstruct(X_svd)) ** 2).mean()
    pca_variance = pca.explained_variance_ratio_[: pca.n_components_].sum()
    svd_variance = svd.explained_variance_ratio_[: svd.n_components_].sum()
    print(f"Data shape: {X.shape}")
    print(f"PCA components: {pca.n_components_} ({pca_variance:.1%} variance)")
    print(f"SVD components: {svd.n_components_} ({svd_variance:.1%} variance)")
    print(f"PCA reconstruction MSE: {pca_error:.4f}")
    print(f"SVD reconstruction MSE: {svd_error:.4f}")
    print(f"PCA scores: {pca_scores}")
    print(f"SVD scores: {svd_scores}")

    for number, component in enumerate(pca.components_[:3], start=1):
        top_feature = features[abs(component).argmax()]
        print(f"Strongest feature in PC{number}: {top_feature}")

    visualization.plot_variance(
        pca.explained_variance_ratio_,
        svd.explained_variance_ratio_,
        variance_target,
        output / "variance.png",
    )
    visualization.plot_2d(X_pca, y, "PCA 2D Projection", output / "pca_2d.png")
    visualization.plot_2d(X_svd, y, "SVD 2D Projection", output / "svd_2d.png")
    if X_pca.shape[1] >= 3:
        visualization.plot_3d(X_pca, y, output / "pca_3d.png")
    visualization.plot_clusters(
        X_pca[:, :2],
        X_svd[:, :2],
        pca_labels,
        svd_labels,
        output / "clusters.png",
    )
    visualization.plot_loadings(pca, features, output / "pca_loadings.png")
    print(f"Plots saved in {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error
