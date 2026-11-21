# Spectral Learning

Simple implementation of PCA and SVD on the UCI red Wine Quality dataset. PCA uses `numpy.linalg.eig`, SVD uses `numpy.linalg.svd`, and no ready-made decomposition model is used.

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

## Workflow

1. Load the semicolon-separated Wine Quality CSV.
2. Remove duplicates, fill missing feature values, and standardize every feature.
3. Apply PCA and SVD while retaining at least 95% of the variance.
4. Run K-means on all retained dimensions and use the first two for plotting.
5. Evaluate clustering with silhouette score and adjusted Rand index (ARI).
6. Save variance, projection, clustering, and loading plots in `outputs/`.

## PCA

PCA centers the data, calculates the covariance matrix, and calls `numpy.linalg.eig`. Eigenvalues and eigenvectors are sorted from largest to smallest. The largest eigenvalues represent the directions with the most variance, so their eigenvectors become the selected principal components.

## SVD

SVD centers the same matrix and factors it into `U`, singular values, and `Vt` with `numpy.linalg.svd`. The first rows of `Vt` are used as components. Squared singular values give the explained-variance ratios. Multiplying reduced data by the selected components reconstructs an approximation of the original data.

PCA and centered SVD should retain the same variance. Component signs can differ without changing their meaning.

## Component Choice

The smallest number of components reaching 95% cumulative variance is kept. This avoids unnecessary dimensions and limits the amount of low-variance noise passed to K-means. Since this is an unsupervised workflow, the cumulative-variance curve is used instead of a supervised learning curve to judge whether enough components were kept.

## Results

With the included dataset, cleaning leaves 1,359 rows and 11 features. Both methods select 9 components and retain 97.8% of the variance. Their results match because PCA on a covariance matrix and SVD on the same centered matrix find the same component directions; component signs may still differ.

The largest absolute loading is fixed acidity for PC1, total sulfur dioxide for PC2, and alcohol for PC3. These features have the strongest influence on those component directions, not necessarily on wine quality itself. K-means gives a silhouette score near 0.197 and an ARI near 0.062, so the quality labels do not form strongly separated clusters in this linear representation.

## Outputs

- `variance.png`: PCA and SVD cumulative variance
- `pca_2d.png` and `svd_2d.png`: 2D projections
- `pca_3d.png`: 3D PCA projection
- `clusters.png`: PCA and SVD K-means patterns
- `pca_loadings.png`: contribution of features to the first components
