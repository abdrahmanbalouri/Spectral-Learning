"""Required project visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _check_plot_data(X, dimensions, labels):
    """Validate only the data needed by projection plots."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < dimensions:
        raise ValueError(f"plot needs at least {dimensions} dimensions")
    if len(labels) != len(X):
        raise ValueError("plot needs one label for each row")
    return X


def _save(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    X = _check_plot_data(X, 2, colors)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=colors, palette="viridis")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    _save(path)


def plot_3d(X, colors, path):
    """Plot a three-dimensional PCA representation."""
    X = _check_plot_data(X, 3, colors)
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(projection="3d")
    axis.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap="viridis", s=12)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("PC3")
    axis.set_title("PCA 3D Projection")
    _save(path)


def plot_clusters(pca_2d, svd_2d, pca_labels, svd_labels, path):
    """Show K-means patterns after PCA and SVD."""
    pca_2d = _check_plot_data(pca_2d, 2, pca_labels)
    svd_2d = _check_plot_data(svd_2d, 2, svd_labels)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=pca_labels, s=10)
    axes[0].set_title("PCA clusters")
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")
    axes[1].scatter(svd_2d[:, 0], svd_2d[:, 1], c=svd_labels, s=10)
    axes[1].set_title("SVD clusters")
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")
    _save(path)


def plot_cluster_subsets(subset_1, subset_2, labels_1, labels_2, path):
    """Compare PCA clustering patterns in two random data subsets."""
    subset_1 = _check_plot_data(subset_1, 2, labels_1)
    subset_2 = _check_plot_data(subset_2, 2, labels_2)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(subset_1[:, 0], subset_1[:, 1], c=labels_1, s=10)
    axes[0].set_title("PCA clusters - subset 1")
    axes[1].scatter(subset_2[:, 0], subset_2[:, 1], c=labels_2, s=10)
    axes[1].set_title("PCA clusters - subset 2")

    for axis in axes:
        axis.set_xlabel("Component 1")
        axis.set_ylabel("Component 2")
    _save(path)


def plot_loadings(pca, feature_names, path):
    """Visualize the first three principal components."""
    count = min(3, pca.n_components_)
    plt.figure(figsize=(10, 3))
    sns.heatmap(
        pca.components_[:count, :],
        cmap="coolwarm",
        center=0,
        xticklabels=feature_names,
        yticklabels=[f"PC{i}" for i in range(1, count + 1)],
    )
    plt.title("PCA Feature Loadings")
    _save(path)
