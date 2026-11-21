"""Load and preprocess Wine Quality data."""

from pathlib import Path

import pandas as pd

from utils.matrix_operations import standardize


def load_data(path="data/wine_quality.csv"):
    """Read, clean, and standardize the numerical dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    data = pd.read_csv(path, sep=";").drop_duplicates()
    target = "quality"
    data = data.apply(pd.to_numeric, errors="coerce").dropna(subset=[target])

    features = data.drop(columns=target)
    features = features.fillna(features.mean())
    if features.isna().any().any():
        raise ValueError("all feature columns must be numerical")

    X = standardize(features.to_numpy())
    y = data[target].to_numpy()
    return X, y, list(features.columns)
