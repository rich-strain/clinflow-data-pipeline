"""
model/trainer.py
----------------
Trains and evaluates scikit-learn classifiers on the prepared
clinical feature matrix.

Models available:
  - Random Forest  (default — robust, interpretable feature importance)
  - Logistic Regression (linear baseline)

Returns performance metrics and artefacts needed for the
Streamlit visualisation layer.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ── Model registry ────────────────────────────────────────────────

def get_model(name: str = "Random Forest"):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name]


# ── Train / evaluate ──────────────────────────────────────────────

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    le,
    model_name: str = "Random Forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split data, train the chosen model, and return a results dict
    containing everything the visualisation layer needs.

    Returns
    -------
    dict with keys:
      model           : fitted sklearn model
      X_train, X_test : feature splits
      y_train, y_test : label splits
      y_pred          : predictions on test set
      y_proba         : predicted probabilities (test set)
      accuracy        : overall accuracy (float)
      cv_scores       : 5-fold cross-validation accuracy scores
      report          : classification_report as dict
      conf_matrix     : confusion matrix (ndarray)
      feature_names   : list of feature column names
      feature_importances : array of importances (RF only, else None)
      class_names     : list of string class labels
      roc_data        : dict of per-class ROC curve data
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    model = get_model(model_name)
    model.fit(X_train, y_train)

    # Predict
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Metrics
    accuracy   = accuracy_score(y_test, y_pred)
    cv_scores  = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    report     = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    conf_mat   = confusion_matrix(y_test, y_pred)

    # Feature importances (Random Forest only)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Use absolute coefficient magnitudes for LogReg (averaged across classes)
        importances = np.abs(model.coef_).mean(axis=0)

    # Per-class ROC curves (one-vs-rest)
    n_classes = len(le.classes_)
    roc_data  = {}
    for i, cls in enumerate(le.classes_):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        auc = roc_auc_score(y_bin, y_proba[:, i])
        roc_data[cls] = {"fpr": fpr, "tpr": tpr, "auc": auc}

    return {
        "model":               model,
        "model_name":          model_name,
        "X_train":             X_train,
        "X_test":              X_test,
        "y_train":             y_train,
        "y_test":              y_test,
        "y_pred":              y_pred,
        "y_proba":             y_proba,
        "accuracy":            accuracy,
        "cv_scores":           cv_scores,
        "report":              report,
        "conf_matrix":         conf_mat,
        "feature_names":       list(X.columns),
        "feature_importances": importances,
        "class_names":         list(le.classes_),
        "roc_data":            roc_data,
    }
