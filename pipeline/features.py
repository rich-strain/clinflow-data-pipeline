"""
pipeline/features.py
---------------------
Stage 2 of the clinical data pipeline.
Takes the cleaned DataFrame and produces a feature matrix
ready for scikit-learn model training.

Responsibilities:
  - Encode the target label (diagnosis → integer)
  - Derive BMI from height + weight
  - Scale numeric features with StandardScaler
  - Return X (features), y (labels), and the fitted scaler
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Features the model will train on
FEATURE_COLS = [
    "age",
    "gender",
    "height",
    "weight",
    "bmi",           # derived
    "temperature",
    "blood_pressure",
    "heart_rate",
    "glucose",
    "spo2",
]

TARGET_COL = "diagnosis"


def derive_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """BMI = weight(kg) / height(m)^2"""
    df = df.copy()
    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / (height_m ** 2)
    # Clip to a clinically plausible range
    df["bmi"] = df["bmi"].clip(10, 70)
    return df


def encode_target(df: pd.DataFrame):
    """
    Encode the diagnosis string column to integer labels.
    Returns (encoded_series, fitted LabelEncoder).
    """
    le = LabelEncoder()
    y  = le.fit_transform(df[TARGET_COL].astype(str))
    return pd.Series(y, index=df.index), le


def build_features(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
):
    """
    Build and scale the feature matrix.

    Parameters
    ----------
    df          : cleaned DataFrame (output of cleaner.clean + cleaner.impute)
    scaler      : pass a pre-fitted scaler to transform without re-fitting
                  (use this for the test set)
    fit_scaler  : if True and scaler is None, fit a new StandardScaler

    Returns
    -------
    X           : scaled feature DataFrame
    y           : integer-encoded label Series
    le          : fitted LabelEncoder
    scaler      : fitted StandardScaler
    """
    df = derive_bmi(df)

    # Ensure all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after cleaning: {missing}")

    X_raw = df[FEATURE_COLS].copy()

    # Encode target
    y, le = encode_target(df)

    # Scale
    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
    elif scaler is not None:
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = X_raw.values

    X = pd.DataFrame(X_scaled, columns=FEATURE_COLS, index=df.index)

    return X, y, le, scaler
