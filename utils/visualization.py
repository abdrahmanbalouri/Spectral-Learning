"""Matplotlib/Seaborn plots for the spectral-learning workflow."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative(pca, svd, threshold, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(pca.full_ratios_) + 1), np.cumsum(pca.full_ratios_), "o-", label="PCA")
    ax.plot(np.arange(1, len(svd.full_ratios_) + 1), np.cumsum(svd.full_ratios_), "s--", label="SVD")
    ax.axhline(threshold, color="gray", linestyle=":", label=f"{threshold:.0%} target")
    ax.set(xlabel="Components", ylabel="Cumulative variance", title="Cumulative explained variance")
    ax.legend()
    _save(fig, path)


def plot_variance_bars(pca, svd, k, path):
    x = np.arange(1, k + 1)
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, pca.explained_variance_ratio_[:k], width, label="PCA")
    ax.bar(x + width / 2, svd.explained_variance_ratio_[:k], width, label="SVD")
    ax.set(xlabel="Component", ylabel="Variance ratio", title="PCA vs SVD variance per component")
    ax.set_xticks(x)
    ax.legend()
    _save(fig, path)


def plot_2d(scores, labels, title, path, xlabel="Component 1", ylabel="Component 2"):
    fig, ax = plt.subplots(figsize=(6, 5))
    points = ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap="tab10", s=20, alpha=0.85)
    fig.colorbar(points, ax=ax, label="Label")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    _save(fig, path)


def plot_3d(scores, labels, title, path):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(scores[:, 0], scores[:, 1], scores[:, 2], c=labels, cmap="tab10", s=16)
    ax.set(xlabel="Component 1", ylabel="Component 2", zlabel="Component 3", title=title)
    _save(fig, path)


def plot_cluster_panels(pca_scores, svd_scores, y, pca_result, svd_result, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (pca_scores, y, "PCA - true labels"),
        (pca_scores, pca_result["labels"], "PCA - K-means clusters"),
        (svd_scores, y, "SVD - true labels"),
        (svd_scores, svd_result["labels"], "SVD - K-means clusters"),
    ]
    for ax, (scores, labels, title) in zip(axes.ravel(), panels):
        ax.scatter(scores[:, 0], scores[:, 1], c=labels, cmap="tab10", s=14)
        ax.set(title=title, xlabel="Component 1", ylabel="Component 2")
    fig.tight_layout()
    _save(fig, path)


def plot_loadings(pca, svd, feature_names, k, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.heatmap(
        pca.components_[:k], cmap="coolwarm", center=0, xticklabels=feature_names,
        yticklabels=[f"PC{i + 1}" for i in range(k)], ax=axes[0],
    )
    sns.heatmap(
        svd.Vt_[:k], cmap="coolwarm", center=0, xticklabels=feature_names,
        yticklabels=[f"SV{i + 1}" for i in range(k)], ax=axes[1],
    )
    axes[0].set_title("PCA loadings")
    axes[1].set_title("SVD right singular vectors")
    fig.tight_layout()
    _save(fig, path)


def plot_reconstruction(pca_errors, svd_errors, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(pca_errors)), pca_errors, "o-", label="PCA")
    ax.plot(range(len(svd_errors)), svd_errors, "s--", label="SVD")
    ax.set(xlabel="Components kept", ylabel="Reconstruction MSE", title="Reconstruction error")
    ax.legend()
    _save(fig, path)


def plot_accuracy(ks, pca_acc, svd_acc, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, pca_acc, "o-", label="PCA + k-NN")
    ax.plot(ks, svd_acc, "s--", label="SVD + k-NN")
    ax.set(xlabel="Components kept", ylabel="k-NN accuracy", title="Downstream classification vs components")
    ax.legend()
    _save(fig, path)
