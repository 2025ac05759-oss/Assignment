import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.features import CLASS_NAMES, FEATURE_COLUMNS, TARGET_COLUMN

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(ROOT, "model", "saved")
SAMPLE_TEST_PATH = os.path.join(ROOT, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(os.path.join(SAVED_DIR, file_name))


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_TEST_PATH)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def build_classification_report(y_true, y_pred, target_names) -> pd.DataFrame:
    """classification_report(output_dict=True) stores 'accuracy' as a bare
    float instead of a {precision, recall, f1, support} dict. Converting
    that straight to a DataFrame broadcasts the single accuracy value into
    every column on that row, which reads as nonsense (e.g. a 'precision of
    accuracy'). This rebuilds the row the way scikit-learn's own plain-text
    report shows it: precision/recall blank, accuracy under f1-score, and
    the true total sample count under support."""
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    accuracy = report.pop("accuracy")

    df = pd.DataFrame(report).transpose().round(3)
    total_support = int(df.loc[target_names, "support"].sum())

    df.loc["accuracy"] = [None, None, round(accuracy, 3), total_support]
    return df.loc[target_names + ["accuracy", "macro avg", "weighted avg"]]


st.title("Breast Cancer Diagnosis - Classifier Comparison")
st.markdown(
    "Predicts whether a breast tumor is **malignant** or **benign** from "
    "30 diagnostic measurements (Breast Cancer Wisconsin Diagnostic dataset). "
    "Upload test data, pick a model, and inspect its performance."
)

st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
use_sample = uploaded_file is None and st.sidebar.checkbox(
    "Use the bundled sample test_data.csv", value=True
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    source_label = uploaded_file.name
elif use_sample:
    data = load_sample_data()
    source_label = "test_data.csv (bundled sample)"
else:
    data = None
    source_label = None

if data is None:
    st.info("Upload a CSV in the sidebar, or tick the sample-data checkbox, to see results.")
    st.stop()

missing_cols = [c for c in FEATURE_COLUMNS if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded file is missing {len(missing_cols)} required feature column(s): {missing_cols[:5]}...")
    st.stop()

st.subheader(f"Data preview — {source_label}")
st.dataframe(data.head(10), width="stretch")
st.caption(f"{data.shape[0]} rows x {data.shape[1]} columns")

X = data[FEATURE_COLUMNS]
has_labels = TARGET_COLUMN in data.columns

pipe = load_model(MODEL_FILES[model_name])
predictions = pipe.predict(X)
probabilities = pipe.predict_proba(X)[:, 1]

results = data.copy()
results["predicted_diagnosis"] = pd.Series(predictions).map(CLASS_NAMES)
results["benign_probability"] = probabilities.round(4)

st.subheader(f"Predictions — {model_name}")
preview_cols = ["id"] if "id" in results.columns else []
preview_cols += ["predicted_diagnosis", "benign_probability"]
if has_labels:
    results["actual_diagnosis"] = results[TARGET_COLUMN].map(CLASS_NAMES)
    preview_cols.append("actual_diagnosis")
st.dataframe(results[preview_cols], width="stretch")

if has_labels:
    y_true = data[TARGET_COLUMN]
    metrics = compute_metrics(y_true, predictions, probabilities)

    st.subheader("Evaluation metrics")
    metric_cols = st.columns(6)
    for col, (metric_name, value) in zip(metric_cols, metrics.items()):
        col.metric(metric_name, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(CLASS_NAMES.values()),
            yticklabels=list(CLASS_NAMES.values()),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification report**")
        report_df = build_classification_report(
            y_true, predictions, target_names=list(CLASS_NAMES.values())
        )
        st.dataframe(report_df, width="stretch")

    with st.expander("Compare all 5 models on this data"):
        rows = []
        for name, file_name in MODEL_FILES.items():
            m = load_model(file_name)
            p = m.predict(X)
            pr = m.predict_proba(X)[:, 1]
            row = compute_metrics(y_true, p, pr)
            row["Model"] = name
            rows.append(row)
        comparison = pd.DataFrame(rows)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
        st.dataframe(comparison.round(4), width="stretch")
else:
    st.warning(
        f"No '{TARGET_COLUMN}' column found in the uploaded data, so evaluation metrics "
        "and the confusion matrix can't be computed — only predictions are shown."
    )
