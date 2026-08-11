# Breast Cancer Diagnosis — Classifier Comparison

## a. Problem statement

Breast cancer diagnosis from fine needle aspirate (FNA) images is traditionally
done by a pathologist inspecting cell nuclei under a microscope. This project
frames diagnosis as a binary classification problem: given a set of numeric
measurements computed from a digitized FNA image of a breast mass, predict
whether the mass is **malignant** or **benign**. Five classification models are
trained on the same dataset, compared on a common set of evaluation metrics,
and made available through an interactive Streamlit app so a user can upload
test data, pick a model, and inspect its predictions and performance.

## b. Dataset description

**Source:** Breast Cancer Wisconsin (Diagnostic) Data Set, originally from the
UCI Machine Learning Repository (also distributed via `scikit-learn.datasets`
and mirrored on Kaggle). The full raw copy used for this project is saved at
[`data/breast_cancer_raw.csv`](data/breast_cancer_raw.csv).

| Property | Value |
|---|---|
| Instances | 569 |
| Features | 30 numeric features (mean, standard-error, and "worst" value of 10 measurements per cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Target | `diagnosis` — 0 = malignant, 1 = benign |
| Class balance | 212 malignant / 357 benign |
| Missing values | None |

The dataset is split 80/20 (stratified on the target) into a training set used
to fit the models and a held-out test set. The held-out test set — with true
labels included — is saved as [`test_data.csv`](test_data.csv) and is what the
Streamlit app uses for its bundled demo / what should be uploaded to see live
evaluation metrics.

## c. GitHub Repository Link

`<PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE_AFTER_PUSHING>`

## d. Models used

All 5 models were trained and evaluated on the identical train/test split
described above (`random_state=42`, 20% held out, stratified).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9368 | 0.9437 | 0.9306 | 0.9371 | 0.8313 |
| kNN | 0.9737 | 0.9944 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

*(Regenerate this table any time with `python model/train_models.py` — it
recomputes the split, retrains every model, and rewrites
`model/metrics_comparison.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best performer on every single metric (Accuracy 0.9825, AUC 0.9954, MCC 0.9623). The 30 features are strongly correlated with the target in a roughly linear/monotonic way after standard scaling, which plays directly to a linear decision boundary and gives it an edge even over the ensemble methods. |
| Decision Tree | Weakest model overall (Accuracy 0.9211, MCC 0.8313). A single tree, even depth-limited (`max_depth=5`) to control overfitting, produces axis-aligned splits that cut across the smooth, correlated feature space less efficiently than a linear or distance-based boundary — it also has the widest gap between AUC (0.9368) and the other models, showing weaker probability calibration. |
| kNN | Very close second (Accuracy 0.9737, Recall 1.0000 — it caught every malignant case in this split). With features standardized before distance computation, the two diagnosis classes turn out to be well separated in the 30-dimensional feature space, which favors a local, distance-based method. |
| Naive Bayes | Middle of the pack (Accuracy 0.9386, MCC 0.8676). The Gaussian Naive Bayes independence assumption is violated here — many of the 30 features are direct mathematical transforms of each other (e.g. mean radius, mean perimeter, and mean area are all measuring the same underlying size), and that correlation is exactly what hurts Naive Bayes most. |
| Random Forest (Ensemble) | Solid, well-balanced performance (Accuracy 0.9474, AUC 0.9937) and a clear improvement over the single Decision Tree it's built from, confirming that bagging many trees reduces the variance that hurt the standalone tree. It still trails Logistic Regression and kNN on this dataset, likely because 569 samples is a modest amount of data for 300 trees to each learn something meaningfully different from. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it leads on Accuracy, AUC, Precision, Recall, F1, and MCC simultaneously, and its strength lines up with what's known about this dataset: the features are engineered, continuous, well-scaled, and largely linearly separable, so a linear model captures nearly all of the signal without the variance risk that hurts the tree-based methods on a dataset this size. |

## Project structure

```
project-folder/
├── app.py                       # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv                # held-out test split (with true labels)
├── data/
│   └── breast_cancer_raw.csv    # full raw dataset
└── model/
    ├── features.py               # shared feature/column definitions
    ├── train_models.py           # trains all 5 models, computes metrics
    ├── metrics_comparison.csv    # generated comparison table
    └── saved/                    # fitted model pipelines (*.joblib)
```

## How to run locally

```bash
pip install -r requirements.txt
python model/train_models.py   # trains all 5 models, writes test_data.csv
streamlit run app.py
```

## Live Streamlit App Link

`<PASTE_YOUR_STREAMLIT_COMMUNITY_CLOUD_APP_URL_HERE_AFTER_DEPLOYING>`

## App features

- CSV upload of test data (sidebar)
- Model selection dropdown (Logistic Regression, Decision Tree, kNN, Naive
  Bayes, Random Forest)
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC) displayed for
  the selected model when the uploaded/sample data includes true labels
- Confusion matrix heatmap and full classification report
- Expandable side-by-side comparison of all 5 models on the same uploaded data
