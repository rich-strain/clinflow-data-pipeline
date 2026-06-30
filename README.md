# ClinFlow Clinical Data Pipeline

A demonstration of a clinical data pipeline that takes messy synthetic patient records,
cleans and normalises them, trains a scikit-learn classifier to predict diagnosis category,
and visualises model performance.

---

## Project Structure

```
clinflow-clinical-pipeline/
├── data/
│   └── generate_dataset.py    # Synthetic messy patient record generator
├── pipeline/
│   ├── cleaner.py             # Parsing, unit conversion, normalisation
│   └── features.py            # Feature engineering, scaling, label encoding
├── model/
│   └── trainer.py             # sklearn model training and evaluation
├── app.py                     # Streamlit UI
├── environment.yml            # Conda environment spec
└── README.md
```

---

## Setup

### 1. Create and activate the Conda environment

```bash
conda env create -f environment.yml
conda activate clinflow-pipeline
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Pipeline Stages

### Stage 1 — Data Generation
`data/generate_dataset.py` produces a synthetic CSV of 200–2000 patient records with
intentionally messy data:
- **Unit inconsistencies** — height in cm / inches / feet-inches; weight in kg / lbs;
  temperature in °C / °F; glucose in mg/dL / mmol/L; SpO2 as % or decimal
- **Format variations** — HR as `"72"`, `"72 bpm"`, or `"HR: 72"`; BP as `"140/90"`,
  `"BP 140 over 90"`, or `"140"`; age as `"35"`, `"35 years"`, `"35 years old"`
- **Missing values** — ~5% random nulls across fields
- **Categorical encoding** — gender as `"Male"`, `"M"`, `"male"`, `"1"`, etc.

### Stage 2 — Cleaning (`pipeline/cleaner.py`)
Each field is parsed by a dedicated function using regex to handle all format variants,
then converted to a canonical unit. Original raw values are preserved with a `_raw`
suffix for traceability. Remaining nulls are median-imputed.

### Stage 3 — Feature Engineering (`pipeline/features.py`)
- BMI is derived from height and weight
- The diagnosis string column is integer-encoded with `LabelEncoder`
- All numeric features are scaled with `StandardScaler`

### Stage 4 — Model Training (`model/trainer.py`)
Supports two classifiers:
- **Random Forest** — robust ensemble, provides feature importances
- **Logistic Regression** — linear baseline, interpretable coefficients

Evaluation includes:
- Train/test split accuracy
- 5-fold cross-validation
- Per-class classification report (precision, recall, F1)
- Confusion matrix
- ROC curves (one-vs-rest per diagnosis class)

### Stage 5 — Visualisation (`app.py`)
Streamlit UI with four tabs:
1. **Data Preview** — raw vs cleaned records side by side
2. **Data Quality** — null counts before/after, class distribution
3. **Model Performance** — accuracy, CV scores, confusion matrix, ROC curves
4. **Feature Analysis** — feature importances, per-diagnosis distributions

---

## Diagnosis Categories (Target Variable)

| Class | Description |
|---|---|
| Cardiovascular | Elevated BP, higher HR |
| Respiratory | Lower SpO2, higher HR |
| Metabolic | Higher BMI, elevated glucose |
| Musculoskeletal | Near-normal vitals |
| Neurological | Mild BP and HR variation |

---

## Notes

- All data is **synthetic** — no real patient data or PHI is used anywhere.
- The dataset generator uses diagnosis-specific vital sign profiles with
  realistic noise to give the classifier meaningful signal.
- Raw values are preserved alongside cleaned values to demonstrate traceability,
  a core requirement in clinical data pipelines.
