"""
config.py
---------
Central configuration for the Telco Customer Churn ML project.
All paths, constants, and the single model registry live here so that
train_models.py and app.py share identical settings without duplication.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to the project root, i.e. the ML_Assignment_2/ folder)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Original source dataset - kept one level up at the workspace root to avoid
# overwriting the downloaded file.  The path is made absolute so scripts work
# regardless of the current working directory.
DATA_CSV = os.path.join(PROJECT_ROOT, "..", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

METRICS_JSON = os.path.join(PROJECT_ROOT, "metrics.json")
MODEL_COMPARISON_CSV = os.path.join(PROJECT_ROOT, "model_comparison.csv")
TEST_DATA_CSV = os.path.join(PROJECT_ROOT, "test_data.csv")
TEST_LABELS_CSV = os.path.join(ARTIFACTS_DIR, "test_labels.csv")
FEATURE_SCHEMA_JSON = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
SPLIT_METADATA_JSON = os.path.join(ARTIFACTS_DIR, "split_metadata.json")

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Minimum dataset requirements
MIN_ROWS = 500
MIN_FEATURES = 12

# ---------------------------------------------------------------------------
# Model registry
# Maps display name -> artifact filename (without directory prefix).
# This is the single source of truth used by both training and the app.
# To add a sixth model later, add one entry here and implement the pipeline
# in train_models.py; no other file needs structural changes.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree":       "decision_tree.joblib",
    "KNN":                 "knn.joblib",
    "Naive Bayes":         "naive_bayes.joblib",
    "Random Forest":       "random_forest.joblib",
}

# Ordered list of model display names (used to maintain consistent ordering)
MODEL_NAMES = list(MODEL_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Evaluation metric fields (used for validation / display ordering)
# ---------------------------------------------------------------------------
METRIC_FIELDS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# ---------------------------------------------------------------------------
# Winner-selection rule (documented here for reference throughout the project)
# ---------------------------------------------------------------------------
# Primary:   Highest F1
# Tie-break 1: Highest MCC
# Tie-break 2: Highest AUC
# Tie-break 3: Highest Accuracy
WINNER_SORT_KEYS = ["F1", "MCC", "AUC", "Accuracy"]
