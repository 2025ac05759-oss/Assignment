"""
Shared feature definitions for the Breast Cancer Wisconsin (Diagnostic)
dataset, used by both the training script and the Streamlit app so the
two never drift out of sync.
"""

from sklearn.datasets import load_breast_cancer

_bunch = load_breast_cancer()

FEATURE_COLUMNS = [name.replace(" ", "_") for name in _bunch.feature_names]
TARGET_COLUMN = "diagnosis"
CLASS_NAMES = {0: "malignant", 1: "benign"}
