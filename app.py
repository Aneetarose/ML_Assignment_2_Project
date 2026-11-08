import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="ML Classification Model Evaluator", page_icon="📊", layout="wide")
st.title("Machine Learning Classification Model Evaluator")
st.caption("M.Tech (AIML) • Machine Learning Assignment 2")
st.write(
    "Upload a labelled CSV containing the 30 Wisconsin Breast Cancer Diagnostic features "
    "and a **target** column. The app evaluates saved models without retraining them."
)
with st.expander("Dataset and target details"):
    st.write("Dataset: Wisconsin Diagnostic Breast Cancer (WDBC), 569 rows and 30 numeric features.")
    st.write("Target encoding: 0 = malignant; 1 = benign. The target column is required for evaluation.")
    st.write("Use the included `test_data.csv` as a sample upload.")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
default_path = BASE_DIR / "test_data.csv"
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
else:
    st.info("No file uploaded. The included sample test_data.csv will be used.")
    df = pd.read_csv(default_path)

missing = [col for col in FEATURES + ["target"] if col not in df.columns]
if missing:
    st.error("Missing required column(s): " + ", ".join(missing))
    st.stop()
if df.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()

try:
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y_true = pd.to_numeric(df["target"], errors="raise").astype(int)
except Exception as exc:
    st.error(f"Features and target must be numeric. Details: {exc}")
    st.stop()

if not set(y_true.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 (malignant) and 1 (benign).")
    st.stop()

selected = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
show_all = st.checkbox("Show comparison of all five models", value=True)

def evaluate(model, X, y):
    pred = model.predict(X)
    try:
        score = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1 Score": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }, pred

def load_model(label):
    path = MODEL_DIR / MODEL_FILES[label]
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return joblib.load(path)

try:
    model = load_model(selected)
    result, y_pred = evaluate(model, X, y_true)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.subheader(f"Metrics — {selected}")
cols = st.columns(6)
for col, (label, value) in zip(cols, result.items()):
    col.metric(label, f"{value:.4f}" if np.isfinite(value) else "N/A")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual malignant (0)", "Actual benign (1)"],
                              columns=["Predicted malignant (0)", "Predicted benign (1)"]),
                 use_container_width=True)
with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["malignant", "benign"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

if show_all:
    st.subheader("Comparison of all five saved models")
    rows = []
    for label in MODEL_FILES:
        m = load_model(label)
        r, _ = evaluate(m, X, y_true)
        rows.append({"ML Model Name": label, **r})
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison.round(4), use_container_width=True)
    st.download_button(
        "Download metrics as CSV",
        comparison.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_metrics.csv",
        mime="text/csv"
    )

st.caption("Models were trained before deployment. Uploaded data is used only for evaluation.")
