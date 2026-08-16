# Telco Customer Churn Prediction - ML Assignment 2

**Student:** Sai Prashanth Karanam  
**Course:** Machine Learning 

---

## 1. Problem Statement

Customer churn - customers discontinuing a subscription service - is a critical
business problem in the telecommunications industry.  This project builds and
compares five classification models that predict whether a customer is likely
to churn, based on demographic, contract, and service-usage features.  The
goal is to identify the best-performing classifier on a real-world imbalanced
dataset and deploy it in an interactive web application.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| **Name** | Telco Customer Churn |
| **Provider** | IBM Sample Datasets (distributed via Kaggle) |
| **Source link** |https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download |
| **File used** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Total rows** | 7 043 |
| **Original columns** | 21 |
| **Identifier column** | `customerID` (excluded from modelling) |
| **Target column** | `Churn` (Yes -> 1, No -> 0) |
| **Predictor columns** | 19 (4 numeric + 15 categorical) |
| **Class distribution** | No Churn: 5 174 (73.5 %) · Churn: 1 869 (26.5 %) |
| **Missing values** | None reported by pandas; 11 blank `TotalCharges` strings coerced to NaN |
| **Duplicate rows** | 0 |

### Feature categories

| Category | Features |
|---|---|
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account & contract | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## 3. Assignment Requirement Validation

| Requirement | Status |
|---|---|
| At least 500 instances | OK 7 043 rows |
| At least 12 input features | OK 19 predictor features |
| Classification task | OK Binary (Churn Yes/No) |


---

## 4. Data Preprocessing

Preprocessing is implemented in `preprocessing.py` using scikit-learn
`Pipeline` and `ColumnTransformer`.  Key decisions:

1. **Column-name normalisation** - leading/trailing whitespace stripped.
2. **TotalCharges conversion** - blank strings coerced to NaN via
   `pd.to_numeric(..., errors='coerce')`.  Rows are **not** dropped because
   blank values correspond to new customers (tenure = 0) who have not yet
   received a bill; dropping them without evidence would bias the dataset.
   The median imputer in the pipeline fills these NaN values.
3. **Identifier exclusion** - `customerID` is dropped from model features but
   optionally retained in prediction output for traceability.
4. **Numeric pipeline** - median imputation -> `StandardScaler`.
   Standardisation is applied to all five model pipelines for a clean,
   uniform implementation.  It has no effect on the split decisions of
   tree-based models.
5. **Categorical pipeline** - most-frequent imputation -> `OneHotEncoder`
   (`handle_unknown="ignore"`, `sparse_output=False`).
   `handle_unknown="ignore"` prevents crashes when unseen categories appear
   in uploaded test data.  `sparse_output=False` produces a dense array
   required by `GaussianNB`.
6. **Leakage prevention** - every transformer is fitted **only** on the
   training split.  Preprocessing is encapsulated inside each saved pipeline
   so training and inference always apply identical transformations.

---

## 5. Train / Test Split and Leakage Prevention

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

| Set | Rows | Churn=1 | Churn=0 |
|---|---|---|---|
| Train | 5 634 | 1 495 | 4 139 |
| Test  | 1 409 |   374 | 1 035 |

- The **same** split indices are used for all five models.
- `test_data.csv` contains only the held-out feature rows (no `Churn` column).
- True labels are saved separately in `artifacts/test_labels.csv`.
- `feature_schema.json` and `split_metadata.json` are saved for the app.

---

## 6. Models Used

| # | Model | Key parameters |
|---|---|---|
| 1 | `LogisticRegression` | `max_iter=2000` (increased for convergence), `random_state=42` |
| 2 | `DecisionTreeClassifier` | `random_state=42` |
| 3 | `KNeighborsClassifier` | defaults (`n_neighbors=5`); numeric features are standardised so distances are meaningful |
| 4 | `GaussianNB` | defaults; dense array output required by this model |
| 5 | `RandomForestClassifier` | `n_estimators=300` (for stable estimates), `random_state=42`, `n_jobs=-1` |

Each model is wrapped in a full preprocessing + classification `Pipeline` and
saved as a `.joblib` artifact.

---

## 7. Evaluation Metrics

All metrics are computed on the **same** held-out test set (1 409 rows).

| Metric | Definition |
|---|---|
| **Accuracy** | Proportion of all predictions that are correct. Misleading on imbalanced datasets. |
| **AUC (ROC)** | Area under the ROC curve.  Computed from probability scores, not hard predictions.  Measures discrimination across all thresholds. |
| **Precision** | Of all predicted positives, the fraction that are true positives.  High precision = few false alarms. |
| **Recall** | Of all actual positives, the fraction correctly identified.  High recall = few missed churners. |
| **F1** | Harmonic mean of Precision and Recall.  Primary metric for winner selection on this imbalanced dataset. |
| **MCC** | Matthews Correlation Coefficient.  Ranges from −1 to +1; considers all four confusion-matrix cells; robust to class imbalance. |

---

## 8. Model Comparison Table

Results on the held-out test set (1 409 rows, stratified 80/20 split):

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** * | **0.8055** | **0.8419** | **0.6572** | 0.5588 | **0.6040** | **0.4790** |
| Decision Tree | 0.7289 | 0.6573 | 0.4896 | 0.5053 | 0.4974 | 0.3119 |
| KNN | 0.7637 | 0.7896 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | **0.8369** | 0.5928 | 0.4245 |
| Random Forest | 0.7786 | 0.8171 | 0.6062 | 0.4733 | 0.5315 | 0.3945 |

---

## 9. Model Observations

**Logistic Regression (winner):** Achieves the highest F1 (0.6040), the
highest Accuracy (0.8055), and the best AUC (0.8419) and MCC (0.4790) among
all five models.  Its strong overall balance across all six metrics makes it
the clear winner under the documented selection rule.

**Decision Tree:** Lowest AUC (0.6573) and lowest F1 (0.4974), suggesting the
unpruned tree does not generalise as well as the other models on this dataset.
Accuracy at 0.7289 is the lowest among the five.

**KNN:** Moderate performance across all metrics (F1 = 0.5636, AUC = 0.7896).
With numeric features standardised, distance-based comparison is meaningful.
Slightly lower Precision (0.5527) than Logistic Regression and Random Forest.

**Naive Bayes:** The most extreme Recall (0.8369) at the cost of low Precision
(0.4589), reflecting a classifier that identifies most actual churners but
generates many false positives.  Despite low Accuracy (0.6948), AUC (0.8074)
is competitive, indicating reasonable probability discrimination.

**Random Forest (ensemble model):** Good Precision (0.6062) and competitive
AUC (0.8171) but lower Recall (0.4733) and F1 (0.5315) than Logistic
Regression, placing it second in Accuracy but fourth in F1.

---

## 10. Overall Winner

**Winner: Logistic Regression**

Selection rule (documented in `config.py`):
1. Primary: highest F1
2. Tie-break 1: highest MCC
3. Tie-break 2: highest AUC
4. Tie-break 3: highest Accuracy

The winner is determined dynamically at training time from `metrics.json`; no
model name is hard-coded.

---

## 11. Streamlit Application Features

- **Model comparison table** - all five models with six metrics, winner
  highlighted in green, visible before any file upload.
- **File upload** - drag-and-drop CSV upload; validates required columns
  against `feature_schema.json`.
- **Model selector** - dropdown for all five models.
- **Metric cards** - six metric cards (Accuracy, AUC, Precision, Recall, F1,
  MCC) labelled as held-out or uploaded-data metrics.
- **Confusion matrix** - heatmap with class labels `No Churn` / `Churn`.
- **Classification report** - per-class Precision, Recall, F1 table.
- **Predictions dataframe** - predicted label, Churn Probability, No Churn
  Probability; retains `customerID` if present.
- **CSV download** - download predictions named after the selected model.
- **Interpretation note** - expandable section explaining educational use.

---

## 12. Project Structure

```text
ML_Assignment_2/
├── app.py                    # Streamlit dashboard
├── train_models.py           # Training and evaluation script
├── preprocessing.py          # Reusable preprocessing utilities
├── config.py                 # Central configuration and model registry
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── test_data.csv             # Held-out features (no Churn column)
├── metrics.json              # Per-model metrics and winner metadata
├── model_comparison.csv      # Tabular comparison for reference
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest.joblib
```

The original dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is kept one
level above the project folder to avoid overwriting the downloaded file.

---

## 13. Local Setup and Execution

### Prerequisites

- Python 3.10 or later (tested with Python 3.14)
- pip

### Install dependencies

```bash
cd ML_Assignment_2
pip install -r requirements.txt
```

### Train models and generate artifacts

```bash
python train_models.py
```

This will:
- Load and validate the dataset
- Perform the 80/20 stratified split
- Train and evaluate all five pipelines
- Save model artifacts to `models/`
- Save `metrics.json`, `model_comparison.csv`, `test_data.csv`
- Save split metadata and feature schema to `artifacts/`
- Print a comparison table to the console

### Run the Streamlit app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in a browser.

---

## 14. How to Retrain Models

Re-run the training script at any time to regenerate all artifacts:

```bash
python train_models.py
```

The script uses `random_state=42` throughout for reproducibility.

---


## 15. Streamlit Community Cloud Deployment

1. Push all project files (including `models/`, `artifacts/`, `metrics.json`,
   `model_comparison.csv`, `test_data.csv`) to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect the repository and set the main file path to `ML_Assignment_2/app.py`.
4. Click **Deploy**.
5. Once live, test by uploading `test_data.csv` and verifying all five models.

---

## 16. Repository and Application Links

- **GitHub repository:** https://github.com/sai362-sky/bits-aiml/tree/master
- **Live Streamlit app:** https://bits-aiml-mee73bmsc8gkhp58u7dahc.streamlit.app/

---

## 17. Reproducibility Notes

- `random_state=42` is used in `train_test_split`, `LogisticRegression`,
  `DecisionTreeClassifier`, and `RandomForestClassifier`.
- `KNeighborsClassifier` and `GaussianNB` are deterministic given fixed input.
- All metric values in this README were generated by running `train_models.py`
  and reading from `metrics.json`; no values are fabricated.

---

## 18. Limitations

- The dataset is a static sample; model performance may differ on live
  production data from a real telecom operator.
- Hyperparameters were not tuned; cross-validated grid search could improve
  results, particularly for KNN and Decision Tree.
- Class imbalance (73.5 % / 26.5 %) was addressed only through stratified
  splitting; techniques such as SMOTE or class weighting may further improve
  Recall for the minority (Churn) class.
- `GaussianNB` assumes feature independence and Gaussian-distributed numeric
  features; these assumptions are likely violated in practice.
- The project uses scikit-learn pipelines that cannot be deployed to
  environments with incompatible scikit-learn versions without retraining.
