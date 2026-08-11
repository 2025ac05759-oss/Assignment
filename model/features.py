"""
Column names for the Breast Cancer Wisconsin dataset, kept in one place
so the training script and the Streamlit app always agree on them.
"""

from sklearn.datasets import load_breast_cancer

_bunch = load_breast_cancer()

FEATURE_COLUMNS = [name.replace(" ", "_") for name in _bunch.feature_names]
TARGET_COLUMN = "diagnosis"
CLASS_NAMES = {0: "malignant", 1: "benign"}
