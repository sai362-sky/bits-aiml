"""
app.py
------
Customer Churn Prediction Dashboard " Streamlit single-page application.

Compares five classifiers trained on the Telco Customer Churn dataset and
predicts whether customers are likely to churn.

Run:
    streamlit run app.py
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

# ---------------------------------------------------------------------------
# Allow imports from the project root regardless of working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from config import (
    FEATURE_SCHEMA_JSON,
    METRIC_FIELDS,
    METRICS_JSON,
    MODEL_NAMES,
    MODEL_REGISTRY,
    MODELS_DIR,
    SPLIT_METADATA_JSON,
)
from preprocessing import separate_id_and_target, validate_uploaded_dataframe


def _abs(path: str) -> str:
    """Return an absolute path anchored at SCRIPT_DIR if not already absolute."""
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    path = _abs(METRICS_JSON)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_split_metadata() -> dict:
    path = _abs(SPLIT_METADATA_JSON)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def load_feature_schema() -> dict:
    path = _abs(FEATURE_SCHEMA_JSON)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_model(display_name: str):
    fname = MODEL_REGISTRY.get(display_name)
    if not fname:
        return None
    path = os.path.join(_abs(MODELS_DIR), fname)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def artifacts_ready(metrics: dict) -> bool:
    return bool(metrics) and all(m in metrics for m in MODEL_NAMES)


def format_metric(val) -> str:
    """Return a 4-decimal string, or 'N/A' for None / non-numeric values."""
    try:
        if val is None:
            return "N/A"
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def metric_card(label: str, value: str):
    st.metric(label=label, value=value)


def _coerce_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast numeric feature columns to float, coercing errors to NaN."""
    df = df.copy()
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_live_metrics(pipeline, X_raw: pd.DataFrame, y_true: pd.Series) -> dict:
    """
    Compute metrics from uploaded labeled data.

    If the uploaded labels contain only one class, AUC cannot be computed;
    AUC is set to None and AUC_unavailable is set to True.
    """
    y_pred  = pipeline.predict(X_raw)
    probas  = pipeline.predict_proba(X_raw)
    y_proba = probas[:, 1]

    unique_classes   = set(int(v) for v in y_true.unique())
    auc              = None
    auc_unavailable  = False

    if len(unique_classes) < 2:
        auc_unavailable = True
    else:
        auc = float(roc_auc_score(y_true, y_proba))

    return {
        "Accuracy":         float(accuracy_score(y_true, y_pred)),
        "AUC":              auc,
        "AUC_unavailable":  auc_unavailable,
        "Precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC":              float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0,
            output_dict=True,
        ),
    }


def render_confusion_matrix(cm_list: list, title: str):
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_classification_report(report_dict: dict):
    rows = []
    for label in ["No Churn", "Churn"]:
        if label in report_dict:
            r = report_dict[label]
            rows.append({
                "Class":     label,
                "Precision": f"{r['precision']:.4f}",
                "Recall":    f"{r['recall']:.4f}",
                "F1-Score":  f"{r['f1-score']:.4f}",
                "Support":   int(r["support"]),
            })
    if "accuracy" in report_dict:
        rows.append({
            "Class": "accuracy", "Precision": "", "Recall": "",
            "F1-Score": f"{report_dict['accuracy']:.4f}",
            "Support": "",
        })
    for avg in ["macro avg", "weighted avg"]:
        if avg in report_dict:
            r = report_dict[avg]
            rows.append({
                "Class":     avg,
                "Precision": f"{r['precision']:.4f}",
                "Recall":    f"{r['recall']:.4f}",
                "F1-Score":  f"{r['f1-score']:.4f}",
                "Support":   int(r["support"]),
            })
    st.dataframe(pd.DataFrame(rows).astype(str), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    metrics = load_metrics()
    split_meta = load_split_metadata()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.title("Churn Prediction")
        st.markdown("**Dataset:** Telco Customer Churn")
        st.markdown("---")
        st.markdown("**Models evaluated:**")
        for name in MODEL_NAMES:
            st.markdown(f"  {name}")
        st.markdown("---")

        if split_meta:
            st.markdown("**Evaluation split:**")
            st.markdown(
                f"- Total rows: **{split_meta.get('total_rows', '"')}**\n"
                f"- Train: **{split_meta.get('train_rows', '"')}** rows\n"
                f"- Test: **{split_meta.get('test_rows', '"')}** rows\n"
                f"- Strategy: 80/20 stratified\n"
                f"- random_state: 42"
            )
        st.markdown("---")
        st.markdown(
            "**Usage guide:**\n"
            "1. Review the model comparison table.\n"
            "2. Upload a CSV to get predictions.\n"
            "3. Choose a model from the dropdown.\n"
            "4. View metrics, confusion matrix, and download results.\n\n"
            "*Upload `test_data.csv` to try a pre-built example.*"
        )
        st.markdown("---")
        st.caption("Developer: Sai Prashanth Karanam")

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        "This application compares **five classifiers** trained on the "
        "Telco Customer Churn dataset and predicts whether customers are "
        "likely to churn."
    )

    if not artifacts_ready(metrics):
        st.error(
            " Model artifacts or metrics not found.\n\n"
            "Please run the following command to generate them:\n"
            "```\npython train_models.py\n```"
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Section A: Model comparison table
    # -----------------------------------------------------------------------
    st.header("Model Comparison")
    st.caption(
        "Metrics computed on the held-out test set (20 % of the dataset, "
        "stratified, random_state=42). Different metrics capture different "
        "aspects of classifier behaviour - no single metric tells the full story."
    )

    meta = metrics.get("_meta", {})
    winner = meta.get("winner", "")
    winner_rule = meta.get("winner_rule", "")

    comp_rows = []
    for name in MODEL_NAMES:
        m = metrics[name]
        row = {"Model": name}
        for field in METRIC_FIELDS:
            row[field] = format_metric(m.get(field))
        comp_rows.append(row)
    comp_df = pd.DataFrame(comp_rows)

    # Highlight winner row using explicit HTML via st.markdown table fallback
    # to avoid Arrow serialisation issues with Styler on Python 3.14.
    def _html_table(df: pd.DataFrame, winner_name: str) -> str:
        cols = list(df.columns)
        header = "".join(f"<th>{c}</th>" for c in cols)
        rows_html = ""
        for _, row in df.iterrows():
            bg = ' style="background-color:#d4edda;font-weight:bold"' if row["Model"] == winner_name else ""
            cells = "".join(f"<td>{row[c]}</td>" for c in cols)
            rows_html += f"<tr{bg}>{cells}</tr>"
        return (
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="background:#f0f2f6">{header}</tr></thead>'
            f'<tbody>{rows_html}</tbody></table>'
        )

    st.markdown(_html_table(comp_df, winner), unsafe_allow_html=True)

    if winner:
        st.success(f"Winner: {winner} -- selected by: {winner_rule}")

    # -----------------------------------------------------------------------
    # Section B: Upload test CSV
    # -----------------------------------------------------------------------
    st.header("Upload Test Data")
    uploaded_file = st.file_uploader("Upload test data CSV", type=["csv"])

    schema = load_feature_schema()
    required_cols = schema.get("required_columns", [])

    uploaded_df = None
    X_upload = None
    y_upload = None
    id_upload = None

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            st.stop()

        if raw_df.empty:
            st.error("The uploaded file is empty. Please upload a CSV with data rows.")
            st.stop()

        st.write(f"**Uploaded:** {raw_df.shape[0]} rows -- {raw_df.shape[1]} columns")
        st.dataframe(raw_df.head(5), use_container_width=True)

        if required_cols:
            missing_cols = validate_uploaded_dataframe(raw_df, required_cols)
            if missing_cols:
                st.error(
                    "The uploaded CSV is missing these required feature columns:\n\n"
                    + ", ".join(f"`{c}`" for c in missing_cols)
                    + "\n\nUse `test_data.csv` as a template for the expected format."
                )
                st.stop()

        try:
            X_upload, y_upload, id_upload = separate_id_and_target(raw_df)
        except ValueError as e:
            st.error(f"Label encoding error in uploaded file: {e}")
            st.stop()

        X_upload = _coerce_feature_dtypes(X_upload)

        if X_upload.shape[0] == 0:
            st.error("No valid rows remain after processing. Check the file format.")
            st.stop()

        nan_ratio = X_upload.isnull().mean()
        high_nan = nan_ratio[nan_ratio > 0.5]
        if not high_nan.empty:
            st.warning(
                "These columns have >50 % missing values and may affect predictions: "
                + ", ".join(f"`{c}`" for c in high_nan.index)
            )

        uploaded_df = raw_df
        if y_upload is not None:
            st.info(
                f"'Churn' column detected ({len(y_upload)} labelled rows). "
                "Metrics will be recalculated from the uploaded labels."
            )
        else:
            st.info(
                "No 'Churn' column detected. Showing stored held-out metrics. "
                "Predictions and probabilities will be generated for all uploaded rows."
            )

    # -----------------------------------------------------------------------
    # Section C: Model selection
    # -----------------------------------------------------------------------
    st.header("Select Model")
    selected_model = st.selectbox("Choose a model:", MODEL_NAMES)
    pipeline = load_model(selected_model)

    if pipeline is None:
        st.error(
            f"Artifact for **{selected_model}** not found. "
            "Run `python train_models.py` to regenerate model files."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # Determine metric source
    # -----------------------------------------------------------------------
    live_metrics = None
    using_live = False
    auc_unavailable = False

    if X_upload is not None and y_upload is not None:
        try:
            live_metrics = compute_live_metrics(pipeline, X_upload, y_upload)
            using_live = True
            auc_unavailable = live_metrics.get("AUC_unavailable", False)
            if auc_unavailable:
                st.warning(
                    "AUC cannot be computed because the uploaded 'Churn' column "
                    "contains only one class. All other metrics are shown."
                )
        except Exception as e:
            st.warning(
                f"Could not compute live metrics: {e}\n\n"
                "Showing stored held-out test metrics instead."
            )

    display_metrics = live_metrics if using_live else metrics[selected_model]
    metrics_source_label = (
        "Metrics recalculated from uploaded labelled data"
        if using_live
        else "Metrics from held-out test split (20 %)"
    )

    # -----------------------------------------------------------------------
    # Section D: Six metric cards
    # -----------------------------------------------------------------------
    st.header(f"Metrics -- {selected_model}")
    st.caption(metrics_source_label)

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    cols = [col1, col2, col3, col4, col5, col6]
    for i, field in enumerate(METRIC_FIELDS):
        with cols[i]:
            if field == "AUC" and auc_unavailable:
                st.metric(label="AUC", value="N/A")
                st.caption("One class only")
            else:
                val = display_metrics.get(field) if isinstance(display_metrics, dict) else display_metrics[field]
                metric_card(field, format_metric(val))

    # -----------------------------------------------------------------------
    # Section E: Confusion matrix and classification report
    # -----------------------------------------------------------------------
    st.header("Confusion Matrix & Classification Report")
    st.caption(metrics_source_label)

    cm_col, report_col = st.columns([1, 2])
    with cm_col:
        render_confusion_matrix(
            display_metrics["confusion_matrix"],
            title=f"{selected_model}",
        )
    with report_col:
        st.markdown("**Classification Report**")
        render_classification_report(display_metrics["classification_report"])

    # -----------------------------------------------------------------------
    # Section F: Predictions for uploaded data
    # -----------------------------------------------------------------------
    if X_upload is not None:
        st.header("Predictions")
        try:
            preds = pipeline.predict(X_upload)
            probas = pipeline.predict_proba(X_upload)

            result_df = X_upload.copy()

            # Prepend customerID if present " retained in output only, never used as feature
            if id_upload is not None:
                result_df.insert(0, "customerID", id_upload.reset_index(drop=True).values)

            result_df["Predicted Churn"]      = np.where(preds == 1, "Churn", "No Churn")
            result_df["Churn Probability"]    = probas[:, 1].round(4)
            result_df["No Churn Probability"] = probas[:, 0].round(4)

            churn_count    = int((preds == 1).sum())
            no_churn_count = int((preds == 0).sum())

            cnt1, cnt2 = st.columns(2)
            cnt1.metric("Predicted Churn",    churn_count)
            cnt2.metric("Predicted No Churn", no_churn_count)

            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            safe_name = selected_model.lower().replace(" ", "_")
            st.download_button(
                label="Download predictions as CSV",
                data=csv_bytes,
                file_name=f"churn_predictions_{safe_name}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(
                f"Prediction failed: {e}\n\n"
                "Ensure the uploaded file has the correct feature columns."
            )

    # -----------------------------------------------------------------------
    # Section G: Interpretation note
    # -----------------------------------------------------------------------
    with st.expander("Model Interpretation Note"):
        st.markdown(
            """
            - Predictions shown here are **educational outputs** produced by
              models trained on a publicly available telco churn sample dataset.
            - A **churn probability** is a model estimate, not a guarantee of
              future customer behaviour.
            - Metrics labelled *held-out test split* are based on the 20 % of
              rows withheld during training (1 409 rows).
              Metrics labelled *uploaded labelled data* are recalculated on the
              file you provided.
            - **AUC** is shown as N/A when the uploaded file contains only one
              class label because ROC AUC requires both classes to be present.
            - These results should not be used for real business decisions
              without further independent validation.
            """
        )


if __name__ == "__main__":
    main()

