import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_cumulative(pca, svd, variance_target, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.full_ratios_), label='PCA Cumulative Variance', lw=2)
    plt.plot(np.cumsum(svd.full_ratios_), label='SVD Cumulative Variance', lw=2, linestyle='--')
    plt.axhline(y=variance_target, color='r', linestyle=':', label=f'Target ({variance_target})')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Variance Explained')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_variance_bars(pca, svd, k_show, filename):
    plt.figure(figsize=(8, 5))
    x = np.arange(1, k_show + 1)
    width = 0.35
    plt.bar(x - width/2, pca.full_ratios_[:k_show], width, label='PCA')
    plt.bar(x + width/2, svd.full_ratios_[:k_show], width, label='SVD')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.title('Variance Ratio per Component')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_2d(X_2d, y, title, filename):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette='viridis', alpha=0.8, legend='full')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_3d(X_3d, y, title, filename):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, cmap='viridis', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_cluster_panels(pca_2d, svd_2d, y, pca_res, svd_res, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(ax=axes[0], x=pca_2d[:, 0], y=pca_2d[:, 1], hue=pca_res['labels'], palette='Set1')
    axes[0].setTitle('PCA K-means Clusters')
    sns.scatterplot(ax=axes[1], x=svd_2d[:, 0], y=svd_2d[:, 1], hue=svd_res['labels'], palette='Set1')
    axes[1].setTitle('SVD K-means Clusters')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_loadings(pca, svd, features, k_show, filename):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pca.components_[:k_show, :], annot=True, fmt='.2f', cmap='coolwarm', xticklabels=features)
    plt.title('PCA Feature Loadings')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_reconstruction(pca_errors, svd_errors, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(pca_errors, label='PCA Reconstruction Error')
    plt.plot(svd_errors, label='SVD Reconstruction Error', linestyle='--')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Error Rate')
    plt.title('Reconstruction Error vs. Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_accuracy(ks, pca_acc, svd_acc, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(ks, pca_acc, label='PCA + KNN Accuracy')
    plt.plot(ks, svd_acc, label='SVD + KNN Accuracy', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('KNN Accuracy')
    plt.title('Classification Accuracy vs Components')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()