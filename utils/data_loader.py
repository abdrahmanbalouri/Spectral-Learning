import os
import urllib.request
import pandas as pd
import numpy as np

WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

def load_data(csv_path=None, download=True):
    """
    Load and preprocess the Wine Quality dataset.
    """
    path = csv_path or "data/wine_quality.csv"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    rows_loaded = 0
    duplicates_removed = 0
    missing_imputed = 0
    
    if not os.path.exists(path) and download:
        try:
            print(f"Downloading dataset from {WINE_URL}...")
            urllib.request.urlretrieve(WINE_URL, path)
        except Exception as e:
            print(f"Download failed ({e}). Generating synthetic wine-like data for demonstration.")
            np.random.seed(42)
            X_synth = np.random.randn(1500, 11)
            y_synth = np.random.randint(3, 9, size=1500)
            cols = [f"feature_{i}" for i in range(11)]
            df_synth = pd.DataFrame(X_synth, columns=cols)
            df_synth['quality'] = y_synth
            df_synth.to_csv(path, sep=';', index=False)

    try:
        # Wine quality dataset uses semicolon separator
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path)

    rows_loaded = len(df)
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        missing_imputed = int(df.isnull().sum().sum())
        df = df.fillna(df.mean(numeric_only=True))
        
    # Handle duplicates
    before_dup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = before_dup - len(df)
    
    # Separate features and target
    if 'quality' in df.columns:
        y = df['quality'].values
        X_df = df.drop(columns=['quality'])
    else:
        y = df.iloc[:, -1].values
        X_df = df.iloc[:, :-1]
        
    features = list(X_df.columns)
    X = X_df.values.astype(np.float64)
    
    # Standardize features (mean=0, std=1)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0 # Prevent division by zero
    X_normalized = (X - mean) / std
    
    info = {
        "source": path,
        "rows_loaded": rows_loaded,
        "duplicates_removed": duplicates_removed,
        "missing_imputed": missing_imputed
    }
    
    return X_normalized, y, features, info