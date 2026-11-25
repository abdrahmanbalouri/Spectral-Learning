"""Required project visualizations."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_variance(pca_ratios, svd_ratios, target, path):
    """Compare cumulative variance for PCA and SVD."""
    plt.figure(figsize=(7, 4))
    plt.plot(
        np.arange(1, len(pca_ratios) + 1), np.cumsum(pca_ratios), label="PCA"
    )
    plt.plot(
        np.arange(1, len(svd_ratios) + 1),
        np.cumsum(svd_ratios),
        "--",
        label="SVD",
    )
    plt.axhline(target, color="red", linestyle=":", label="Target")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance")
    plt.legend()
    _save(path)


def plot_2d(X, colors, title, path):
    """Plot a two-dimensional representation."""
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=colors, palette="viridis")
    plt.title(title)
    _save(path)


def plot_3d(X, colors, path):
    """Plot a three-dimensional PCA representation."""
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(projection="3d")
    axis.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap="viridis", s=12)
    axis.set_title("PCA 3D Projection")
    _save(path)


def plot_clusters(pca_2d, svd_2d, pca_labels, svd_labels, path):
    """Show K-means patterns after PCA and SVD."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=pca_labels, s=10)
    axes[0].set_title("PCA clusters")
    axes[1].scatter(svd_2d[:, 0], svd_2d[:, 1], c=svd_labels, s=10)
    axes[1].set_title("SVD clusters")
    _save(path)


def plot_loadings(pca, feature_names, path):
    """Visualize the first three principal components."""
    count = min(3, pca.n_components_)
    sns.heatmap(
        pca.components_[:count,:],
        cmap="coolwarm",
        center=0,
        xticklabels=feature_names,
        yticklabels=[f"PC{i}" for i in range(1, count + 1)],
    )
    plt.title("PCA Feature Loadings")
    _save(path)
