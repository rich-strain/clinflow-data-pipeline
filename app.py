"""
app.py
------
Streamlit UI for the ClinFlow Clinical Data Pipeline demo.

Stages shown:
  1. Raw data preview  (messy)
  2. Cleaned data preview + quality metrics
  3. Model training controls
  4. Performance charts:
       - Accuracy + CV scores
       - Confusion matrix
       - ROC curves (per class)
       - Feature importances
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Path setup ────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from data.generate_dataset import generate
from pipeline.cleaner       import clean, impute
from pipeline.features      import build_features, derive_bmi
from model.trainer          import train_and_evaluate

# ── Colorblind-friendly palette ───────────────────────────────────
PALETTE = {
    "primary":   "#1B6CA8",   # blue
    "secondary": "#E87722",   # amber
    "teal":      "#008080",
    "coral":     "#E05C5C",
    "gray":      "#7F8C8D",
    "classes": [
        "#1B6CA8",  # blue        — Cardiovascular
        "#E87722",  # amber       — Respiratory
        "#008080",  # teal        — Metabolic
        "#9B59B6",  # purple/violet — Musculoskeletal  (not adjacent to blue here)
        "#E05C5C",  # coral       — Neurological
    ],
}

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinFlow Clinical Pipeline",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 ClinFlow Clinical Data Pipeline")
st.markdown(
    "A demonstration of **messy EHR data → cleaning pipeline → ML model training → performance analysis**."
)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — controls
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Pipeline Controls")

    n_records = st.slider("Number of patient records", 200, 2000, 1000, step=100)
    model_choice = st.selectbox("Classifier", ["Random Forest", "Logistic Regression"])
    test_size    = st.slider("Test set fraction", 0.10, 0.40, 0.20, step=0.05)
    run_btn      = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE — persist results across rerenders
# ═══════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

# ═══════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════
if run_btn:
    with st.spinner("Generating synthetic patient records…"):
        raw_df = generate(n=n_records)
        st.session_state.raw_df = raw_df

    with st.spinner("Cleaning and normalising data…"):
        cleaned_df = clean(raw_df)
        cleaned_df = impute(cleaned_df)
        cleaned_df = derive_bmi(cleaned_df)
        st.session_state.clean_df = cleaned_df

    with st.spinner("Engineering features and training model…"):
        X, y, le, scaler = build_features(cleaned_df)
        results = train_and_evaluate(
            X, y, le,
            model_name=model_choice,
            test_size=test_size,
        )
        st.session_state.results = results

    st.success("Pipeline complete!")

# ═══════════════════════════════════════════════════════════════════
# DISPLAY — only shown after pipeline runs
# ═══════════════════════════════════════════════════════════════════
if st.session_state.results is None:
    st.info("👈 Configure the pipeline in the sidebar and click **Run Pipeline** to begin.")
    st.stop()

raw_df   = st.session_state.raw_df
clean_df = st.session_state.clean_df
results  = st.session_state.results

# ── Tab layout ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data Preview",
    "🔬 Data Quality",
    "📊 Model Performance",
    "🔍 Feature Analysis",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Data Preview
# ═══════════════════════════════════════════════════════════════════
with tab1:
    n_preview = st.slider("Rows to preview", min_value=10, max_value=min(200, len(raw_df)), value=20, step=10)

    st.subheader("Raw (Messy) Patient Records")
    st.dataframe(raw_df.head(n_preview), use_container_width=True)

    st.subheader("After Cleaning & Normalisation")
    display_cols = [
        "patient_id", "age", "gender", "height", "weight", "bmi",
        "temperature", "blood_pressure", "heart_rate", "glucose", "spo2", "diagnosis"
    ]
    display_cols = [c for c in display_cols if c in clean_df.columns]
    st.dataframe(clean_df[display_cols].head(n_preview).round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Data Quality
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Data Quality Improvements")

    feature_cols = ["age", "gender", "height", "weight",
                    "temperature", "blood_pressure", "heart_rate", "glucose", "spo2"]

    # Null counts before and after
    raw_nulls   = raw_df[feature_cols].isna().sum()
    clean_nulls = clean_df[feature_cols].isna().sum()

    quality_df = pd.DataFrame({
        "Raw Nulls":     raw_nulls,
        "Cleaned Nulls": clean_nulls,
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=[
            go.Bar(name="Before Cleaning", x=quality_df.index, y=quality_df["Raw Nulls"],
                   marker_color=PALETTE["coral"]),
            go.Bar(name="After Cleaning",  x=quality_df.index, y=quality_df["Cleaned Nulls"],
                   marker_color=PALETTE["teal"]),
        ])
        fig.update_layout(
            title="Missing Values: Before vs After",
            barmode="group",
            xaxis_title="Feature",
            yaxis_title="Null Count",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Class distribution
        class_counts = pd.Series(results["class_names"])[results["y_test"]].value_counts()

        # Get value counts properly
        y_series = pd.Series(results["y_test"])
        label_map = {i: name for i, name in enumerate(results["class_names"])}
        named_counts = y_series.map(label_map).value_counts()

        fig2 = px.pie(
            values=named_counts.values,
            names=named_counts.index,
            title="Diagnosis Distribution (Test Set)",
            color_discrete_sequence=PALETTE["classes"],
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Summary stats comparison
    st.subheader("Cleaned Feature Statistics")
    st.dataframe(
        clean_df[feature_cols].describe().round(2),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Model: {results['model_name']}")

    # ── KPI metrics ───────────────────────────────────────────────
    acc      = results["accuracy"]
    cv_mean  = results["cv_scores"].mean()
    cv_std   = results["cv_scores"].std()
    avg_auc  = np.mean([v["auc"] for v in results["roc_data"].values()])

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Accuracy",       f"{acc:.1%}")
    m2.metric("CV Accuracy (5-fold)", f"{cv_mean:.1%}", f"± {cv_std:.1%}")
    m3.metric("Avg. AUC (OvR)",      f"{avg_auc:.3f}")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Confusion Matrix ──────────────────────────────────────────
    with col_left:
        cm     = results["conf_matrix"]
        labels = results["class_names"]

        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=[[0, "#EBF4FB"], [1, PALETTE["primary"]]],
            title="Confusion Matrix",
        )
        fig_cm.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curves ────────────────────────────────────────────────
    with col_right:
        fig_roc = go.Figure()
        for i, (cls, data) in enumerate(results["roc_data"].items()):
            fig_roc.add_trace(go.Scatter(
                x=data["fpr"],
                y=data["tpr"],
                name=f"{cls} (AUC={data['auc']:.2f})",
                line=dict(color=PALETTE["classes"][i % len(PALETTE["classes"])], width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color=PALETTE["gray"], dash="dash", width=1),
            showlegend=False,
            name="Chance",
        ))
        fig_roc.update_layout(
            title="ROC Curves (One-vs-Rest per Class)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.6, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Normalized confusion matrix ───────────────────────────────
    st.subheader("Confusion Matrix — Normalised (Row = Recall per Class)")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig_cm_norm = px.imshow(
        cm_norm,
        labels=dict(x="Predicted", y="Actual", color="Recall"),
        x=labels,
        y=labels,
        text_auto=".0%",
        color_continuous_scale=[[0, "#EBF4FB"], [1, PALETTE["primary"]]],
        zmin=0, zmax=1,
        title="Row-Normalised Confusion Matrix (each row sums to 100%)",
    )
    fig_cm_norm.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm_norm, use_container_width=True)

    # ── Training set class balance ────────────────────────────────
    st.subheader("Class Balance — Training Set")
    y_train_series = pd.Series(results["y_train"])
    label_map = {i: name for i, name in enumerate(results["class_names"])}
    train_counts = y_train_series.map(label_map).value_counts().sort_index()
    fig_balance = go.Figure(go.Bar(
        x=train_counts.index,
        y=train_counts.values,
        marker_color=PALETTE["classes"],
        text=train_counts.values,
        textposition="outside",
    ))
    fig_balance.update_layout(
        yaxis_title="Training samples",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig_balance, use_container_width=True)

    # ── Cross-validation scores ───────────────────────────────────
    st.subheader("Cross-Validation Accuracy (5 Folds)")
    cv_df = pd.DataFrame({
        "Fold":     [f"Fold {i+1}" for i in range(len(results["cv_scores"]))],
        "Accuracy": results["cv_scores"],
    })
    fig_cv = go.Figure(data=[
        go.Bar(
            x=cv_df["Fold"],
            y=cv_df["Accuracy"],
            marker_color=PALETTE["primary"],
            text=[f"{v:.1%}" for v in cv_df["Accuracy"]],
            textposition="outside",
        )
    ])
    fig_cv.add_hline(
        y=cv_mean,
        line_dash="dash",
        line_color=PALETTE["secondary"],
        annotation_text=f"Mean: {cv_mean:.1%}",
    )
    fig_cv.update_layout(
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # ── Per-class report ──────────────────────────────────────────
    st.subheader("Per-Class Classification Report")
    report_df = pd.DataFrame(results["report"]).T
    report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    report_df = report_df[["precision", "recall", "f1-score", "support"]].round(3)
    st.dataframe(report_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — Feature Analysis
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Feature Importances")

    importances   = results["feature_importances"]
    feature_names = results["feature_names"]

    importance_df = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Feature"],
        orientation="h",
        marker_color=PALETTE["secondary"],
    ))
    fig_imp.update_layout(
        title=f"Feature Importances — {results['model_name']}",
        xaxis_title="Importance Score",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Feature distributions by diagnosis ───────────────────────
    st.subheader("Feature Distributions by Diagnosis")

    top_feature = importance_df.iloc[-1]["Feature"]  # most important
    selected_feature = st.selectbox(
        "Select feature to explore",
        options=feature_names,
        index=feature_names.index(top_feature),
    )

    plot_df = clean_df[["diagnosis", selected_feature]].dropna()

    fig_box = px.box(
        plot_df,
        x="diagnosis",
        y=selected_feature,
        color="diagnosis",
        color_discrete_sequence=PALETTE["classes"],
        title=f"{selected_feature} Distribution by Diagnosis",
        points="outliers",
    )
    fig_box.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Diagnosis",
        yaxis_title=selected_feature,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Class feature profile heatmap ─────────────────────────────
    st.subheader("Class Feature Profiles (Z-Scored Means)")
    st.markdown(
        "Each cell shows how far above/below the global mean a class sits for that feature. "
        "A class with **no distinctive pattern** (all cells near zero) will be hard to separate."
    )

    numeric_features = [c for c in feature_names if c in clean_df.columns]
    profile_df = clean_df.groupby("diagnosis")[numeric_features].mean()
    profile_z  = (profile_df - profile_df.mean()) / profile_df.std()
    profile_z  = profile_z.reindex(sorted(profile_z.index))

    fig_profile = px.imshow(
        profile_z,
        labels=dict(x="Feature", y="Diagnosis", color="Z-score"),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        text_auto=".2f",
        title="Z-Scored Feature Means by Class — classes with weak signal cluster near 0",
        aspect="auto",
    )
    fig_profile.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    st.plotly_chart(fig_profile, use_container_width=True)

    # ── ANOVA F-statistic per feature for Neurological ────────────
    st.subheader("Feature Discriminability — ANOVA F-Statistic")
    st.markdown(
        "Higher F means a feature separates classes well. "
        "Low F for every feature confirms the signal problem is in the **data**, not the model."
    )

    from scipy.stats import f_oneway

    f_rows = []
    for feat in numeric_features:
        groups = [
            clean_df.loc[clean_df["diagnosis"] == cls, feat].dropna().values
            for cls in clean_df["diagnosis"].unique()
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) >= 2:
            f_stat, p_val = f_oneway(*groups)
            f_rows.append({"Feature": feat, "F-statistic": round(f_stat, 2), "p-value": round(p_val, 4)})

    f_df = pd.DataFrame(f_rows).sort_values("F-statistic", ascending=True)

    fig_f = go.Figure(go.Bar(
        x=f_df["F-statistic"],
        y=f_df["Feature"],
        orientation="h",
        marker_color=[
            PALETTE["coral"] if v < 20 else PALETTE["teal"]
            for v in f_df["F-statistic"]
        ],
        text=[f"F={v:.1f}" for v in f_df["F-statistic"]],
        textposition="outside",
    ))
    fig_f.update_layout(
        title="ANOVA F-Statistic per Feature (red = weak separation across all 5 classes)",
        xaxis_title="F-statistic",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # ── Per-class model weights / importances ─────────────────────
    st.subheader("Per-Class Model Weights")
    model = results["model"]

    if hasattr(model, "coef_"):
        # Logistic Regression: coef_ shape (n_classes, n_features)
        coef_df = pd.DataFrame(
            model.coef_,
            index=results["class_names"],
            columns=feature_names,
        )
        fig_coef = px.imshow(
            coef_df,
            labels=dict(x="Feature", y="Class", color="Coefficient"),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            text_auto=".2f",
            title="Logistic Regression Coefficients per Class — large |values| = strong signal",
            aspect="auto",
        )
        fig_coef.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption(
            "Neurological row near-zero across all features → model has no discriminating coefficients to assign."
        )
    else:
        # Random Forest: show mean predicted probability per true class (calibration heatmap)
        st.markdown("**Mean predicted class probabilities for each true class (Random Forest)**")
        y_test   = results["y_test"]
        y_proba  = results["y_proba"]
        classes  = results["class_names"]
        prob_rows = []
        for i, cls in enumerate(classes):
            mask = (y_test == i)
            if mask.sum() > 0:
                mean_probs = y_proba[mask].mean(axis=0)
                prob_rows.append(dict(zip(["True Class"] + classes, [cls] + list(mean_probs))))
        prob_df = pd.DataFrame(prob_rows).set_index("True Class")

        fig_prob = px.imshow(
            prob_df.astype(float),
            labels=dict(x="Predicted Class", y="True Class", color="Mean P"),
            color_continuous_scale=[[0, "#EBF4FB"], [1, PALETTE["primary"]]],
            text_auto=".2f",
            zmin=0, zmax=1,
            title="Mean RF Predicted Probabilities by True Class — diagonal = confidence; off-diagonal = confusion",
            aspect="auto",
        )
        fig_prob.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.caption(
            "Low diagonal probability for Neurological row = model is uncertain about every Neurological case."
        )
