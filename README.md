# Breast Cancer Diagnosis — Classifier Comparison

## a. Problem statement

Breast cancer is usually diagnosed by a doctor looking at cell samples under a
microscope and deciding whether a lump is cancerous (malignant) or not
(benign). For this project I built a classifier that makes that same
malignant-vs-benign call automatically, using numeric measurements taken from
the cell images instead of a human judgment call. It's a binary classification
problem, and I trained five different ML models on the same dataset so I
could compare how each one actually performs instead of just picking one and
hoping for the best. Everything is wrapped in a Streamlit app where you can
upload test data, pick which model to use, and see how well it does.

## b. Dataset description

I used the Breast Cancer Wisconsin (Diagnostic) dataset a well-known
dataset originally from the UCI Machine Learning Repository (it also ships
built into `scikit-learn` and is mirrored on Kaggle). The raw copy I used is
saved at [`data/breast_cancer_raw.csv`](data/breast_cancer_raw.csv).

Each row is one tumor sample. The columns are 30 numbers calculated from a
digitized image of the cell nuclei things like radius, texture, and
smoothness and each of those 10 underlying measurements is given three
ways: its mean, its standard error, and its "worst" (largest) value. That's
why there are 30 columns instead of 10.

| Property | Value |
|---|---|
| Instances | 569 |
| Features | 30 numeric columns (mean / error / worst of 10 cell-nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) |
| Target | `diagnosis` — 0 = malignant, 1 = benign |
| Class balance | 212 malignant, 357 benign |
| Missing values | None |

I split the data 80/20 into a training set and a test set, keeping the same
malignant/benign ratio in both halves (a "stratified" split) so the test set
is a fair, representative sample and not accidentally skewed toward one
class. The test set with the correct answers included is saved as
[`test_data.csv`](test_data.csv). That's the file the Streamlit app uses by
default, and it's what you should upload if you want to see the metrics
update live.

## c. GitHub Repository Link

`<PASTE_YOUR_GITHUB_REPOSITORY_URL_HERE_AFTER_PUSHING>`

## d. Models used

All 5 models were trained and tested on the exact same split described above
(same random seed, same 80/20 split), so the comparison below is apples to
apples.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9368 | 0.9437 | 0.9306 | 0.9371 | 0.8313 |
| kNN | 0.9737 | 0.9944 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

*(You can regenerate this table any time by running `python model/train_models.py` —
it redoes the split, retrains every model, and rewrites `model/metrics_comparison.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Came out on top on every single metric (accuracy 0.9825, AUC 0.9954, MCC 0.9623). I didn't expect the simplest model here to win, but it makes sense once you look at the features — most of the 30 columns are really just different ways of measuring "how big and irregular is this cell", so they move in a pretty straight line with the diagnosis. That's exactly the kind of pattern a linear model like this is built for. |
| Decision Tree | The weakest of the five (accuracy 0.9211, MCC 0.8313). I capped the tree depth at 5 so it wouldn't badly overfit on just 569 rows, but even a shallow tree can only cut the data into boxy, straight-edged regions, so it misses some of the smoother patterns the other models pick up. Its AUC (0.9368) was also the lowest by a clear margin, meaning its confidence scores are less trustworthy than the other models'. |
| kNN | A close second, and it got recall to a perfect 1.0 — it didn't miss a single actual malignant case in this test set. That works well here because I scaled the features first, so distance between points means something, and the malignant and benign cases turn out to sit in fairly separate clusters once you do that. |
| Naive Bayes | Landed in the middle (accuracy 0.9386, MCC 0.8676). Naive Bayes assumes all the features are independent of one another, and that's basically false here — mean radius, mean perimeter, and mean area are all just different ways of describing the size of the same cell, so they're heavily correlated. That broken assumption is probably the main reason it underperforms the rest. |
| Random Forest (Ensemble) | Did well (accuracy 0.9474, AUC 0.9937) and was a clear step up from the single Decision Tree it's built out of, which lines up with what you'd expect — averaging many trees smooths out the mistakes any one tree makes. It still didn't beat Logistic Regression or kNN here, probably because 569 rows isn't a huge amount of data for 300 trees to each find something different to specialize in. |
| **Overall Winner for your dataset?** | **Logistic Regression.** It wins on every metric at once, which doesn't happen very often. My best guess is that it comes down to the dataset itself — the features are clean, continuous, and pretty much linearly related to the outcome, so a simple linear model doesn't really lose anything by not being more complex. |

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

- Upload your own test CSV, or use the sample `test_data.csv` that's already bundled in
- Dropdown to switch between all 5 models
- Accuracy, AUC, Precision, Recall, F1, and MCC shown for whichever model is selected
- Confusion matrix and a full classification report
- An expandable section that runs all 5 models on the same uploaded data side by side, so you can compare them directly
