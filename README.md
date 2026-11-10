# Spectral Learning

A small, readable project that implements **PCA** and **SVD from scratch** with NumPy and applies them to the UCI **Wine Quality** dataset. The goal is to show how high-dimensional data can be compressed, visualized and clustered while keeping the most important patterns.

## What the project does

1. Loads and cleans the Wine Quality data (duplicates, missing values, standardization).
2. Runs PCA using the covariance matrix and eigenvalue decomposition.
3. Runs SVD using `A.T @ A = V S² V.T` and `U = A V / S`.
4. Compares variance explained, reconstruction error and component agreement.
5. Clusters the raw data and the 2-D PCA/SVD projections with K-means.
6. Measures how reduced dimensions affect a k-NN classifier.
7. Saves 2-D/3-D plots, loadings heatmaps and a JSON report.

PCA and SVD never call `sklearn.decomposition`, `numpy.linalg.svd` or SciPy's SVD. Scikit-learn is only used for K-means, metrics, k-NN and the offline fallback dataset.

## Project structure

```text
spectral-learning-project/
├── data/wine_quality.csv
├── models/
│   ├── pca_model.py        # PCA from scratch
│   ├── svd_model.py        # SVD from scratch
│   └── __init__.py
├── utils/
│   ├── data_loader.py      # loading and cleaning
│   ├── matrix_operations.py# shared NumPy helpers
│   ├── clustering.py       # K-means metrics and k-NN
│   ├── visualization.py    # plots
│   └── __init__.py
├── main.py                 # runs the full workflow
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Useful options:

```bash
python main.py --variance 0.90          # keep enough components for 90% variance
python main.py --clusters 3             # set the number of K-means clusters
python main.py --csv path/to/data.csv   # use another CSV
python main.py --no-download            # never download; use sklearn fallback if needed
python main.py --output-dir results     # choose the output folder
```

## Outputs

All results are written to `outputs/`:

| File | Content |
| --- | --- |
| `01_cumulative_variance.png` | PCA vs SVD cumulative variance |
| `02_variance_comparison.png` | Variance per component |
| `03_pca_2d.png`, `04_svd_2d.png` | 2-D projections colored by label |
| `05_pca_3d.png`, `06_svd_3d.png` | 3-D projections |
| `07_cluster_patterns.png` | True labels vs K-means clusters |
| `08_loadings.png` | PCA loadings and SVD right singular vectors |
| `09_reconstruction.png` | Reconstruction error vs components |
| `10_knn_accuracy.png` | k-NN accuracy vs components |
| `report.json` | Dataset, variance and clustering summary |

## How it works

**PCA** centers the data, estimates the covariance matrix, computes its eigenvalues/eigenvectors with `numpy.linalg.eigh`, sorts them from largest to smallest and keeps the leading directions.

**SVD** computes the Gram matrix `A.T @ A`, eigendecomposes it to obtain `V` and the squared singular values, then builds `U = A V / S`. Centering is enabled by default, which makes SVD equivalent to PCA and makes variance comparisons meaningful.

**K-means** is fitted in the original space and in the 2-D PCA/SVD spaces. Cluster IDs are arbitrary, so accuracy is computed with Hungarian matching against the true labels. Silhouette score and adjusted Rand index are also reported.

## Notes

- The Wine Quality dataset is included in `data/`. If it is missing, the loader tries the UCI URL and then falls back to `sklearn.datasets.load_wine`.
- The dataset is standardized before decomposition, so total variance equals the number of features.
- `n_components` can be `None`, an integer, or a float such as `0.95`.
- Classification uses a single stratified train/test split. PCA/SVD are fitted on the training split for that evaluation.
- The workflow is sized for the wine data (about 1,400 rows); very large datasets may need sampling or a faster silhouette implementation.
