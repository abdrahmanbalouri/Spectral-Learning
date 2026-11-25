# Spectral Learning

A small, from-scratch implementation of PCA and SVD on the UCI red Wine Quality dataset. The goal is to understand every main step of dimensionality reduction instead of hiding it inside a ready-made estimator.

## Objectives

- Clean and standardize a numerical dataset.
- Implement PCA with a covariance matrix and `numpy.linalg.eig`.
- Implement SVD reduction and reconstruction with `numpy.linalg.svd`.
- Choose a useful number of components from cumulative explained variance.
- Compare PCA and SVD with reconstruction, clustering, and visualizations.

No `sklearn.decomposition` class is used. Scikit-learn is used only for K-means and evaluation metrics.

## Project Structure

```text
spectral-learning/
├── data/wine_quality.csv
├── models/
│   ├── pca_model.py
│   ├── svd_model.py
│   └── __init__.py
├── utils/
│   ├── data_loader.py
│   ├── matrix_operations.py
│   ├── clustering.py
│   ├── visualization.py
│   └── __init__.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The included CSV is the red-wine file from the UCI Wine Quality dataset. It is semicolon-delimited, as distributed by UCI, so no download is needed to run the project.

## Workflow

1. Load the semicolon-separated Wine Quality CSV.
2. Remove duplicates, fill missing feature values, and standardize every feature with NumPy.
3. Apply PCA and SVD while retaining at least 95% of the variance.
4. Run K-means on all retained dimensions and on two random data subsets.
5. Evaluate clustering with silhouette score and adjusted Rand index (ARI).
6. Save variance, projection, clustering, and loading plots in `outputs/`.

The program reports clear errors for missing or invalid data, invalid component choices, insufficient clustering classes, and projection data with too few dimensions.

## PCA

PCA centers the data, calculates the covariance matrix as `X_centered.T @ X_centered / (n - 1)`, and calls `numpy.linalg.eig`. Eigenvalues and eigenvectors are sorted from largest to smallest. An eigenvector is a direction in the data and its eigenvalue is the variance along that direction. The top eigenvectors therefore give the directions that preserve the most information.

## SVD

SVD centers the same matrix and factors it into `U`, singular values, and `Vt` with `numpy.linalg.svd`. The first rows of `Vt` are used as components. Squared singular values give the explained-variance ratios. Multiplying reduced data by the selected components reconstructs an approximation of the original data.

PCA and centered SVD should retain the same variance. Component signs can differ without changing their meaning.

## Component Choice

The smallest number of components reaching 95% cumulative variance is kept. This avoids unnecessary dimensions and limits the amount of low-variance noise passed to K-means. Since this is an unsupervised workflow, the cumulative-variance curve is used instead of a supervised learning curve to judge whether enough components were kept.

Before looking at the plot, a reasonable prediction for this 11-feature dataset is that roughly 8-10 components will be needed for 95%. The result is 9 components. Choosing this threshold before evaluating clusters also avoids selecting dimensions only because they happen to improve one result.

## Results

With the included dataset, cleaning leaves 1,359 rows and 11 features. Both methods select 9 components and retain 97.8% of the variance. Their reconstruction MSE is about 0.0218. Their results match because PCA on a covariance matrix and SVD on the same centered matrix find the same component directions; component signs may still differ.

The largest absolute loading is fixed acidity for PC1, total sulfur dioxide for PC2, and alcohol for PC3. These features have the strongest influence on those component directions, not necessarily on wine quality itself. K-means gives a silhouette score near 0.197 and an ARI near 0.062, so the quality labels do not form strongly separated clusters in this linear representation.

## Broader Applications

- **Image compression:** SVD can keep only the largest singular values of an image matrix. Keeping fewer values creates a smaller but less detailed reconstruction.
- **Signal processing:** PCA or SVD can separate strong shared patterns from low-variance noise, which is useful for denoising sensor or audio signals.

## Outputs

- `variance.png`: PCA and SVD cumulative variance
- `pca_2d.png` and `svd_2d.png`: 2D projections
- `pca_3d.png`: 3D PCA projection
- `clusters.png`: PCA and SVD K-means patterns
- `cluster_subsets.png`: PCA clustering patterns across two random data subsets
- `pca_loadings.png`: contribution of features to the first components
