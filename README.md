# Spectral Learning

Simple implementation of PCA and SVD on the UCI Wine Quality dataset. PCA uses `numpy.linalg.eig`, SVD uses `numpy.linalg.svd`, and no ready-made decomposition model is used.

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
4. Run K-means on the first two reduced dimensions.
5. Evaluate clustering with silhouette score and adjusted Rand index (ARI).
6. Save variance, projection, clustering, and loading plots in `outputs/`.

## PCA

PCA centers the data, calculates the covariance matrix, and calls `numpy.linalg.eig`. Eigenvalues and eigenvectors are sorted from largest to smallest. The largest eigenvalues represent the directions with the most variance, so their eigenvectors become the selected principal components.

## SVD

SVD centers the same matrix and factors it into `U`, singular values, and `Vt` with `numpy.linalg.svd`. The first rows of `Vt` are used as components. Squared singular values give the explained-variance ratios. Multiplying reduced data by the selected components reconstructs an approximation of the original data.

PCA and centered SVD should retain the same variance. Component signs can differ without changing their meaning.

## Component Choice

The smallest number of components reaching 95% cumulative variance is kept. This removes low-variance noise, keeps the model simple, and reduces overfitting. The cumulative-variance plot shows whether this choice is sufficient.

## Outputs

- `variance.png`: PCA and SVD cumulative variance
- `pca_2d.png` and `svd_2d.png`: 2D projections
- `pca_3d.png`: 3D PCA projection
- `clusters.png`: PCA and SVD K-means patterns
- `pca_loadings.png`: contribution of features to the first components
