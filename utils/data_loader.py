"""Load and clean the Wine Quality dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from utils.matrix_operations import standardize

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "wine_quality.csv"


def load_data(csv_path=None, download=True):
    """Return X (standardized), y, feature names and a small info dict."""
    path = Path(csv_path) if csv_path else DEFAULT_PATH
    source = "local"

    if not path.exists() and download:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(URL, path)
            source = "uci"
        except Exception as error:
            print(f"Download failed ({error}); using sklearn wine dataset.")
            path = None
    elif not path.exists():
        path = None

    df = None
    if path is not None and path.exists():
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception as error:
            print(f"Could not read {path} ({error}); using sklearn wine dataset.")

    if df is None or df.empty:
        from sklearn.datasets import load_wine

        data = load_wine()
        columns = [name.replace(" ", "_") for name in data.feature_names]
        df = pd.DataFrame(data.data, columns=columns)
        df["quality"] = data.target
        source = "sklearn"

    target = "quality" if "quality" in df.columns else df.columns[-1]
    rows_loaded = len(df)

    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])
    before = len(df)
    df = df.drop_duplicates()
    duplicates = before - len(df)

    features = [column for column in df.columns if column != target]
    all_nan = [column for column in features if df[column].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)
        features = [column for column in features if column not in all_nan]

    missing = int(df[features].isna().sum().sum())
    df[features] = df[features].fillna(df[features].median())

    constant = [column for column in features if df[column].nunique() <= 1]
    if constant:
        df = df.drop(columns=constant)
        features = [column for column in features if column not in constant]

    if df.empty or not features:
        raise ValueError("No usable data left after cleaning")

    X = standardize(df[features].to_numpy(dtype=float))
    y = df[target].to_numpy()
    if np.allclose(y, np.round(y)):
        y = np.round(y).astype(int)

    info = {
        "source": source,
        "rows_loaded": rows_loaded,
        "rows_clean": len(df),
        "duplicates_removed": duplicates,
        "missing_imputed": missing,
        "features_removed": all_nan + constant,
    }
    return X, y, features, info
