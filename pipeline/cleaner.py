"""
pipeline/cleaner.py
-------------------
Stage 1 of the clinical data pipeline.
Handles parsing, unit standardization, and format normalization
of the raw messy patient records.

All values are converted to a canonical unit:
  height      → cm (float)
  weight      → kg (float)
  temperature → Celsius (float)
  blood_pressure → systolic_bp + diastolic_bp mmHg (float)
  heart_rate  → bpm (int)
  glucose     → mg/dL (float)
  spo2        → percentage 0–100 (float)
  age         → years (int)
  gender      → 0 = Female, 1 = Male
"""

import re
import pandas as pd
import numpy as np


# ── Age ───────────────────────────────────────────────────────────

def parse_age(val) -> float:
    """Extract numeric age from strings like '35', '35 years', '35 years old'."""
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else np.nan


# ── Gender ────────────────────────────────────────────────────────

_MALE_TOKENS   = {"male", "m", "1"}
_FEMALE_TOKENS = {"female", "f", "0"}

def parse_gender(val) -> float:
    """Encode gender as 1 = Male, 0 = Female, NaN = unknown."""
    if pd.isna(val):
        return np.nan
    token = str(val).strip().lower()
    if token in _MALE_TOKENS:
        return 1.0
    if token in _FEMALE_TOKENS:
        return 0.0
    return np.nan


# ── Height → cm ───────────────────────────────────────────────────

def parse_height(val) -> float:
    """
    Convert height to centimetres.
    Handles: '175.0 cm', '68.9 in', "5'9\"", "5'9"
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    # feet-and-inches: 5'9" or 5'9
    ft_in = re.match(r"""(\d+)'(\d+(?:\.\d+)?)"?""", s)
    if ft_in:
        feet  = float(ft_in.group(1))
        inches = float(ft_in.group(2))
        return (feet * 12 + inches) * 2.54

    # numeric + unit
    num_match = re.search(r"([\d.]+)\s*(cm|in|inch|inches)?", s, re.IGNORECASE)
    if num_match:
        num  = float(num_match.group(1))
        unit = (num_match.group(2) or "cm").lower()
        if unit in ("in", "inch", "inches"):
            return num * 2.54
        return num  # already cm

    return np.nan


# ── Weight → kg ───────────────────────────────────────────────────

def parse_weight(val) -> float:
    """Convert weight to kilograms. Handles 'kg' and 'lbs'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    num_match = re.search(r"([\d.]+)\s*(kg|lbs?)?", s, re.IGNORECASE)
    if num_match:
        num  = float(num_match.group(1))
        unit = (num_match.group(2) or "kg").lower()
        if unit.startswith("lb"):
            return num / 2.20462
        return num
    return np.nan


# ── Temperature → Celsius ─────────────────────────────────────────

def parse_temperature(val) -> float:
    """Convert temperature to Celsius. Handles C and F."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    num_match = re.search(r"([\d.]+)\s*([CF])?", s, re.IGNORECASE)
    if num_match:
        num  = float(num_match.group(1))
        unit = (num_match.group(2) or "C").upper()
        if unit == "F":
            return (num - 32) * 5 / 9
        return num
    return np.nan


# ── Blood Pressure → systolic mmHg ───────────────────────────────

def parse_blood_pressure(val) -> tuple:
    """
    Extract (systolic, diastolic) BP in mmHg.
    Handles: '140/90', 'BP 140 over 90', '140' (systolic only → diastolic NaN)
    """
    if pd.isna(val):
        return (np.nan, np.nan)
    s = str(val).strip()

    # "140/90" or "140 / 90"
    slash = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if slash:
        return (float(slash.group(1)), float(slash.group(2)))

    # "BP 140 over 90" or "140 over 90"
    over = re.search(r"(\d+)\s+over\s+(\d+)", s, re.IGNORECASE)
    if over:
        return (float(over.group(1)), float(over.group(2)))

    # bare number — systolic only, diastolic was never recorded
    bare = re.search(r"(\d+)", s)
    if bare:
        return (float(bare.group(1)), np.nan)

    return (np.nan, np.nan)


# ── Heart Rate → bpm ──────────────────────────────────────────────

def parse_heart_rate(val) -> float:
    """Extract numeric heart rate from '72', '72 bpm', 'HR: 72'."""
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else np.nan


# ── Glucose → mg/dL ───────────────────────────────────────────────

def parse_glucose(val) -> float:
    """
    Convert glucose to mg/dL.
    Handles 'mg/dL' (no conversion) and 'mmol/L' (multiply by 18).
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    num_match = re.search(r"([\d.]+)\s*(mg|mmol)?", s, re.IGNORECASE)
    if num_match:
        num  = float(num_match.group(1))
        unit = (num_match.group(2) or "mg").lower()
        if unit == "mmol":
            return num * 18.0
        return num
    return np.nan


# ── SpO2 → percentage ─────────────────────────────────────────────

def parse_spo2(val) -> float:
    """
    Convert SpO2 to a 0–100 percentage.
    Handles '97%' and decimal '0.97'.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()

    # percentage: "97%"
    pct = re.search(r"([\d.]+)\s*%", s)
    if pct:
        return float(pct.group(1))

    # decimal: "0.97"
    dec = re.match(r"^0\.([\d]+)$", s)
    if dec:
        return float(s) * 100

    # bare number — assume percentage if > 1
    bare = re.search(r"([\d.]+)", s)
    if bare:
        num = float(bare.group(1))
        return num if num > 1 else num * 100

    return np.nan


# ── Main cleaner ──────────────────────────────────────────────────

# Scalar parsers applied column-by-column. Blood pressure is handled
# separately in clean() because it splits one field into two columns.
PARSERS = {
    "age":            parse_age,
    "gender":         parse_gender,
    "height":         parse_height,
    "weight":         parse_weight,
    "temperature":    parse_temperature,
    "heart_rate":     parse_heart_rate,
    "glucose":        parse_glucose,
    "spo2":           parse_spo2,
}

# Numeric columns present after clean() — used for median imputation.
NUMERIC_COLS = [
    "age", "gender", "height", "weight", "temperature",
    "systolic_bp", "diastolic_bp", "heart_rate", "glucose", "spo2",
]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all parsers to the raw DataFrame.
    Returns a new DataFrame with cleaned, numeric columns.
    Original messy columns are preserved with a '_raw' suffix for traceability.
    """
    cleaned = df.copy()

    for col, parser in PARSERS.items():
        if col in cleaned.columns:
            cleaned[f"{col}_raw"] = cleaned[col]   # preserve original
            cleaned[col] = cleaned[col].apply(parser)

    # Blood pressure splits one messy field into two numeric columns.
    # Bare-number records (systolic only) leave diastolic_bp as NaN,
    # which the imputation step then fills.
    if "blood_pressure" in cleaned.columns:
        cleaned["blood_pressure_raw"] = cleaned["blood_pressure"]
        bp = cleaned["blood_pressure"].apply(parse_blood_pressure)
        cleaned["systolic_bp"]  = bp.apply(lambda pair: pair[0])
        cleaned["diastolic_bp"] = bp.apply(lambda pair: pair[1])
        cleaned = cleaned.drop(columns=["blood_pressure"])

    return cleaned


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple median imputation for remaining NaN values in numeric columns.
    In a production pipeline this would be a more sophisticated strategy.
    """
    for col in NUMERIC_COLS:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df
