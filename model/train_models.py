"""
Trains all 5 classification models on the Breast Cancer Wisconsin
(Diagnostic) dataset, evaluates each with the required metric set,
and saves:
  - the full raw dataset             -> data/breast_cancer_raw.csv
  - one fitted pipeline per model    -> model/saved/*.joblib
  - a metrics comparison table       -> model/metrics_comparison.csv
  - the held-out test split (with true labels) -> test_data.csv

Run from the project root:
    python model/train_models.py
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(ROOT, "data", "breast_cancer_raw.csv")
SAVED_DIR = os.path.join(ROOT, "model", "saved")
METRICS_PATH = os.path.join(ROOT, "model", "metrics_comparison.csv")
TEST_DATA_PATH = os.path.join(ROOT, "test_data.csv")

RANDOM_STATE = 42


def load_raw_dataset() -> pd.DataFrame:
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=FEATURE_COLUMNS)
    df.insert(0, "id", range(1, len(df) + 1))
    df[TARGET_COLUMN] = bunch.target  # 0 = malignant, 1 = benign
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved raw dataset ({df.shape[0]} rows, {len(FEATURE_COLUMNS)} features) -> {RAW_DATA_PATH}")
    return df


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE)),
        ]),
        "kNN": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=9)),
        ]),
        "Naive Bayes": Pipeline([
            ("classifier", GaussianNB()),
        ]),
        "Random Forest (Ensemble)": Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    os.makedirs(SAVED_DIR, exist_ok=True)

    df = load_raw_dataset()

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET_COLUMN], random_state=RANDOM_STATE
    )
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved held-out test data ({len(test_df)} rows) -> {TEST_DATA_PATH}")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    results = []
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

        file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
        joblib.dump(pipe, os.path.join(SAVED_DIR, file_name))
        print(f"{name:30s} -> {metrics}")

    results_df = pd.DataFrame(results)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    results_df.to_csv(METRICS_PATH, index=False)
    print("\nComparison table:")
    print(results_df.to_string(index=False))
    print(f"\nSaved metrics -> {METRICS_PATH}")
    print(f"Saved models  -> {SAVED_DIR}")


if __name__ == "__main__":
    main()
