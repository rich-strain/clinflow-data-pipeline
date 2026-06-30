"""
generate_dataset.py
-------------------
Generates a synthetic dataset of messy patient records for the
ClinFlow clinical pipeline demo. Intentionally introduces realistic
data quality issues: unit inconsistencies, format variations,
mixed encodings, and values requiring translation/normalization.
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

# ── Diagnosis categories (target variable) ────────────────────────
DIAGNOSES = {
    "Cardiovascular":   0,
    "Respiratory":      1,
    "Metabolic":        2,
    "Musculoskeletal":  3,
    "Neurological":     4,
}

# ── Typical vital ranges per diagnosis ────────────────────────────
DIAGNOSIS_PROFILES = {
    "Cardiovascular":  {"sbp": (140, 20), "hr": (88, 15),  "bmi": (28, 4),  "glucose": (105, 15), "spo2": (96, 2)},
    "Respiratory":     {"sbp": (118, 12), "hr": (95, 18),  "bmi": (24, 4),  "glucose": (98,  12), "spo2": (91, 4)},
    "Metabolic":       {"sbp": (130, 15), "hr": (80, 12),  "bmi": (34, 5),  "glucose": (180, 40), "spo2": (97, 1)},
    "Musculoskeletal": {"sbp": (120, 10), "hr": (75, 10),  "bmi": (27, 5),  "glucose": (95,  10), "spo2": (98, 1)},
    "Neurological":    {"sbp": (125, 18), "hr": (82, 14),  "bmi": (25, 4),  "glucose": (100, 15), "spo2": (97, 2)},
}

# ── Messy format helpers ───────────────────────────────────────────

def messy_gender(gender: str) -> str:
    """Return gender in one of several inconsistent formats."""
    variants = {
        "Male":   ["Male", "M", "male", "MALE", "1", "m"],
        "Female": ["Female", "F", "female", "FEMALE", "0", "f"],
    }
    return random.choice(variants[gender])

def messy_height(height_cm: float) -> str:
    """Return height in cm, inches, or feet-and-inches string."""
    fmt = random.choice(["cm", "in", "ft"])
    if fmt == "cm":
        return f"{height_cm:.1f} cm"
    elif fmt == "in":
        inches = height_cm / 2.54
        return f"{inches:.1f} in"
    else:
        total_in = height_cm / 2.54
        feet = int(total_in // 12)
        inch = total_in % 12
        return f"{feet}'{inch:.0f}\""

def messy_weight(weight_kg: float) -> str:
    """Return weight in kg or lbs."""
    if random.random() < 0.5:
        return f"{weight_kg:.1f} kg"
    else:
        return f"{weight_kg * 2.20462:.1f} lbs"

def messy_temp(temp_c: float) -> str:
    """Return temperature in Celsius or Fahrenheit."""
    if random.random() < 0.5:
        return f"{temp_c:.1f} C"
    else:
        return f"{temp_c * 9/5 + 32:.1f} F"

def messy_bp(sbp: int) -> str:
    """Return systolic BP in various formats."""
    dbp = sbp - random.randint(30, 50)
    fmt = random.choice(["slash", "text", "raw"])
    if fmt == "slash":
        return f"{sbp}/{dbp}"
    elif fmt == "text":
        return f"BP {sbp} over {dbp}"
    else:
        return f"{sbp}"

def messy_hr(hr: int) -> str:
    """Return heart rate in various formats."""
    fmt = random.choice(["raw", "bpm", "labeled"])
    if fmt == "raw":
        return str(hr)
    elif fmt == "bpm":
        return f"{hr} bpm"
    else:
        return f"HR: {hr}"

def messy_glucose(glucose: float) -> str:
    """Return glucose in mg/dL or mmol/L."""
    if random.random() < 0.6:
        return f"{glucose:.0f} mg/dL"
    else:
        return f"{glucose / 18.0:.1f} mmol/L"

def messy_age(age: int) -> str:
    """Return age as integer string, 'X years', or 'X years old'."""
    fmt = random.choice(["raw", "years", "years old"])
    if fmt == "raw":
        return str(age)
    elif fmt == "years":
        return f"{age} years"
    else:
        return f"{age} years old"

def messy_spo2(spo2: float) -> str:
    """Return SpO2 as percentage or decimal."""
    if random.random() < 0.7:
        return f"{spo2:.0f}%"
    else:
        return f"{spo2 / 100:.2f}"

# ── Row generator ─────────────────────────────────────────────────

def generate_patient(patient_id: int) -> dict:
    diagnosis = random.choice(list(DIAGNOSIS_PROFILES.keys()))
    profile   = DIAGNOSIS_PROFILES[diagnosis]

    gender     = random.choice(["Male", "Female"])
    age        = random.randint(25, 80)
    height_cm  = np.random.normal(170 if gender == "Male" else 162, 8)
    weight_kg  = np.random.normal(82 if gender == "Male" else 68, 14)
    temp_c     = np.random.normal(37.0, 0.4)

    sbp     = max(80,  int(np.random.normal(*profile["sbp"])))
    hr      = max(40,  int(np.random.normal(*profile["hr"])))
    glucose = max(50,  np.random.normal(*profile["glucose"]))
    spo2    = min(100, max(80, np.random.normal(*profile["spo2"])))

    # Occasionally inject nulls to simulate missing data
    def maybe_null(val, p=0.05):
        return None if random.random() < p else val

    return {
        "patient_id": f"PT-{patient_id:05d}",
        "age":        messy_age(age),
        "gender":     messy_gender(gender),
        "height":     maybe_null(messy_height(height_cm)),
        "weight":     maybe_null(messy_weight(weight_kg)),
        "temperature":maybe_null(messy_temp(temp_c)),
        "blood_pressure": maybe_null(messy_bp(sbp)),
        "heart_rate": maybe_null(messy_hr(hr)),
        "glucose":    maybe_null(messy_glucose(glucose)),
        "spo2":       maybe_null(messy_spo2(spo2)),
        "diagnosis":  diagnosis,
    }

# ── Main ──────────────────────────────────────────────────────────

def generate(n: int = 1000, output_path: str = None) -> pd.DataFrame:
    records = [generate_patient(i + 1) for i in range(n)]
    df = pd.DataFrame(records)
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Generated {n} records → {output_path}")
    return df

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "raw_patients.csv")
    generate(n=1000, output_path=out)
