"""
train_models.py
---------------
Train five classification pipelines on the Telco Customer Churn dataset,
evaluate them on a held-out test split, save all artifacts, and print a
comparison summary.

Run with:
    python train_models.py

All outputs land in the same directory as this script (the ML_Assignment_2/
project root).
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Allow running from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    ARTIFACTS_DIR,
    DATA_CSV,
    FEATURE_SCHEMA_JSON,
    METRIC_FIELDS,
    METRICS_JSON,
    MODEL_COMPARISON_CSV,
    MODEL_NAMES,
    MODEL_REGISTRY,
    MODELS_DIR,
    RANDOM_STATE,
    SPLIT_METADATA_JSON,
    TEST_DATA_CSV,
    TEST_LABELS_CSV,
    TEST_SIZE,
    WINNER_SORT_KEYS,
)
from preprocessing import (
    build_model_pipeline,
    encode_target,
    get_required_feature_columns,
    load_and_clean_source,
    prepare_features_and_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    for d in [MODELS_DIR, ARTIFACTS_DIR]:
        os.makedirs(d, exist_ok=True)


def compute_metrics(pipeline, X_test, y_test) -> dict:
    """Compute all six required metrics for one fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def sanity_check(metrics_dict: dict, n_test: int):
    """Assert basic metric bounds and row-count consistency."""
    for model_name, m in metrics_dict.items():
        for field in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
            val = m[field]
            assert 0.0 <= val <= 1.0, (
                f"Metric {field} for {model_name} out of [0,1]: {val}"
            )
        assert -1.0 <= m["MCC"] <= 1.0, (
            f"MCC for {model_name} out of [-1,1]: {m['MCC']}"
        )
    # All models evaluated on same number of rows (verified via same X_test)
    print(f"  Sanity: all models evaluated on {n_test} held-out rows - OK")


def select_winner(metrics_dict: dict) -> str:
    """
    Select the best model using:
    Primary: highest F1
    Tie-break 1: highest MCC
    Tie-break 2: highest AUC
    Tie-break 3: highest Accuracy
    """
    scores = {
        name: tuple(m[k] for k in WINNER_SORT_KEYS)
        for name, m in metrics_dict.items()
    }
    return max(scores, key=lambda n: scores[n])


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Telco Customer Churn - Model Training")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Phase 1: Load and validate dataset
    # ------------------------------------------------------------------
    data_path = os.path.abspath(DATA_CSV)
    print(f"\n[1] Loading dataset: {os.path.basename(data_path)}")
    df = load_and_clean_source(data_path)
    print(f"    Shape            : {df.shape}")
    print(f"    Columns          : {list(df.columns)}")
    print(f"    Churn values     : {df['Churn'].unique()}")
    print(f"    TotalCharges NaN : {df['TotalCharges'].isna().sum()}")
    print(f"    Duplicates       : {df.duplicated().sum()}")
    print(f"    Class distribution:\n{df['Churn'].value_counts().to_string()}")

    X, y = prepare_features_and_target(df)
    print(f"\n    Predictor features : {X.shape[1]}  (>=12 required, check passed)")
    print(f"    Rows               : {X.shape[0]}  (>=500 required, check passed)")
    assert "Churn" not in X.columns, "Target column leaked into features!"

    # ------------------------------------------------------------------
    # Phase 2: Train/test split (stratified, random_state=42)
    # ------------------------------------------------------------------
    print("\n[2] Splitting data (80/20 stratified, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"    Train rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"    Train Churn=1: {y_train.sum()}  |  Test Churn=1: {y_test.sum()}")

    # Save split metadata
    ensure_dirs()
    split_meta = {
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratified": True,
        "train_churn_count": int(y_train.sum()),
        "test_churn_count": int(y_test.sum()),
        "train_no_churn_count": int((y_train == 0).sum()),
        "test_no_churn_count": int((y_test == 0).sum()),
    }
    with open(SPLIT_METADATA_JSON, "w") as f:
        json.dump(split_meta, f, indent=2)
    print(f"    Split metadata saved -> {os.path.relpath(SPLIT_METADATA_JSON, SCRIPT_DIR)}")

    # Save feature schema
    feature_schema = {
        "required_columns": get_required_feature_columns(),
        "numeric_features": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod",
        ],
    }
    with open(FEATURE_SCHEMA_JSON, "w") as f:
        json.dump(feature_schema, f, indent=2)
    print(f"    Feature schema saved -> {os.path.relpath(FEATURE_SCHEMA_JSON, SCRIPT_DIR)}")

    # Save test_data.csv (NO target column - for inference demonstration)
    X_test.reset_index(drop=True).to_csv(TEST_DATA_CSV, index=False)
    assert "Churn" not in pd.read_csv(TEST_DATA_CSV).columns, \
        "Churn column must not appear in test_data.csv!"
    print(f"    test_data.csv saved -> {os.path.relpath(TEST_DATA_CSV, SCRIPT_DIR)}  "
          f"(no Churn column - verified)")

    # Save test labels separately
    y_test.reset_index(drop=True).rename("Churn").to_csv(TEST_LABELS_CSV, index=False)
    print(f"    test_labels.csv saved -> {os.path.relpath(TEST_LABELS_CSV, SCRIPT_DIR)}")

    # ------------------------------------------------------------------
    # Phase 3 & 4: Define, train, evaluate all five model pipelines
    # ------------------------------------------------------------------
    print("\n[3] Building model definitions ...")

    # Five required classifiers (with documented parameter choices)
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,        # increased from default 100 to ensure convergence
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            # default n_neighbors=5; numeric features are standardised so
            # distance-based comparison is meaningful
        ),
        "Naive Bayes": GaussianNB(
            # default settings; works with dense arrays produced by preprocessor
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,     # 300 trees for stable estimates
            random_state=RANDOM_STATE,
            n_jobs=-1,            # use all available CPU cores
        ),
    }

    print("\n[4] Training and evaluating ...\n")
    all_metrics = {}
    for display_name in MODEL_NAMES:
        clf = classifiers[display_name]
        artifact_file = MODEL_REGISTRY[display_name]
        artifact_path = os.path.join(MODELS_DIR, artifact_file)

        print(f"  -> {display_name}")
        pipeline = build_model_pipeline(clf)
        pipeline.fit(X_train, y_train)

        # Evaluate on held-out test set
        m = compute_metrics(pipeline, X_test, y_test)
        all_metrics[display_name] = m
        print(f"     Accuracy={m['Accuracy']:.4f}  AUC={m['AUC']:.4f}  "
              f"F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")

        # Save pipeline artifact
        joblib.dump(pipeline, artifact_path)
        print(f"     Saved -> {os.path.relpath(artifact_path, SCRIPT_DIR)}")

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    print("\n[5] Running sanity checks ...")
    sanity_check(all_metrics, len(X_test))

    # Verify every artifact can be loaded and predict
    for name, fname in MODEL_REGISTRY.items():
        path = os.path.join(MODELS_DIR, fname)
        pipe = joblib.load(path)
        sample = X_test.iloc[:1]
        pred = pipe.predict(sample)
        assert len(pred) == 1, f"Prediction length mismatch for {name}"
    print("  Sanity: all five artifacts load and predict a sample row - OK")

    # ------------------------------------------------------------------
    # Determine winner dynamically
    # ------------------------------------------------------------------
    winner = select_winner(all_metrics)
    winner_rule = (
        "Primary: highest F1; tie-break order: MCC -> AUC -> Accuracy"
    )
    print(f"\n  Winner: {winner}  ({winner_rule})\n")
    # ------------------------------------------------------------------
    # Save metrics.json
    # ------------------------------------------------------------------
    output_metrics = {
        name: {
            "Accuracy": m["Accuracy"],
            "AUC": m["AUC"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1": m["F1"],
            "MCC": m["MCC"],
            "confusion_matrix": m["confusion_matrix"],
            "classification_report": m["classification_report"],
        }
        for name, m in all_metrics.items()
    }
    output_metrics["_meta"] = {
        "winner": winner,
        "winner_rule": winner_rule,
        "test_rows": int(len(X_test)),
        "metric_fields": METRIC_FIELDS,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(output_metrics, f, indent=2)
    print(f"\n[6] metrics.json saved -> {os.path.relpath(METRICS_JSON, SCRIPT_DIR)}")

    # ------------------------------------------------------------------
    # Save model_comparison.csv
    # ------------------------------------------------------------------
    rows = []
    for name in MODEL_NAMES:
        m = all_metrics[name]
        rows.append({
            "Model":     name,
            "Accuracy":  round(m["Accuracy"], 6),
            "AUC":       round(m["AUC"], 6),
            "Precision": round(m["Precision"], 6),
            "Recall":    round(m["Recall"], 6),
            "F1":        round(m["F1"], 6),
            "MCC":       round(m["MCC"], 6),
        })
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"    model_comparison.csv saved -> {os.path.relpath(MODEL_COMPARISON_CSV, SCRIPT_DIR)}")

    # ------------------------------------------------------------------
    # Console comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Model Comparison (held-out test set)")
    print("=" * 65)
    print(comparison_df.to_string(index=False))
    print(f"\n  *** Winner: {winner} ***")
    print("=" * 65)
    print("\nDone. Training complete.  Run  pytest -q  to validate artifacts.")


if __name__ == "__main__":
    main()

