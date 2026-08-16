"""
preprocessing.py
----------------
Reusable preprocessing utilities for the Telco Customer Churn project.

Design decisions
~~~~~~~~~~~~~~~~
* All preprocessing is encapsulated inside each saved sklearn Pipeline so
  training and inference always apply identical transformations.
* Transformers are fitted **only** on training data; this module never fits
  on the full dataset or on uploaded test data.
* Numeric features are standardised for all five pipelines.  This is
  acceptable because StandardScaler is a no-op for tree-based models in
  terms of split decisions, and using a uniform approach keeps the code
  clean and avoids model-specific preprocessing branches.
* OneHotEncoder uses sparse_output=False (sklearn >= 1.2) so the output is
  a dense array, which is required by GaussianNB.
* handle_unknown="ignore" ensures unseen categories in uploaded test data
  do not crash the app.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column definitions (derived from dataset inspection)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Dataset loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean_source(csv_path: str) -> pd.DataFrame:
    """
    Load the raw Telco CSV and apply minimal, documented cleaning steps.

    Steps
    -----
    1. Strip leading/trailing whitespace from column names.
    2. Convert TotalCharges to float, coercing blank strings to NaN.
       Rows with blank TotalCharges are **not** dropped - the median imputer
       in the pipeline will fill them.  Dropping them would be unjustified
       because the blank value pattern likely corresponds to customers with
       zero tenure (new customers who have not yet been billed).
    3. Validate and encode the Churn target column.
    4. Drop the customerID identifier column from the feature set.
    """
    df = pd.read_csv(csv_path, dtype=str)

    # 1. Normalise column name whitespace
    df.columns = [c.strip() for c in df.columns]

    # 2. TotalCharges: coerce blanks/spaces to NaN then cast to float
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(), errors="coerce"
    )

    # 3. Cast numeric columns that should be numeric
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    return df


def encode_target(series: pd.Series) -> pd.Series:
    """
    Map Churn Yes/No to 1/0.

    - Strips surrounding whitespace from each value before mapping.
    - Raises ValueError on unexpected labels.
    - Raises ValueError if any NaN values remain after mapping
      (used during training; upload path tolerates missing via app checks).
    """
    # Convert to plain Python strings so .strip() works on ArrowStringArray too
    cleaned = series.astype(str).str.strip()
    allowed = {"Yes", "No"}
    unique_vals = set(cleaned.dropna().unique()) - {"nan", "None", ""}
    unexpected = unique_vals - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected Churn values found: {unexpected}. "
            f"Expected only 'Yes' and 'No'."
        )
    result = cleaned.map({"Yes": 1, "No": 0})
    if result.isna().any():
        raise ValueError(
            "Churn column contains missing or unrecognised values after mapping. "
            "Ensure every row has 'Yes' or 'No'."
        )
    return result.astype(int)


def prepare_features_and_target(df: pd.DataFrame):
    """
    Return (X, y) after validating and encoding.

    X does not contain customerID or Churn.
    y is the integer-encoded Churn series.
    """
    from config import TARGET_COLUMN, ID_COLUMN, MIN_ROWS, MIN_FEATURES

    # Encode target
    y = encode_target(df[TARGET_COLUMN])

    # Drop ID and target from features
    drop_cols = [TARGET_COLUMN]
    if ID_COLUMN in df.columns:
        drop_cols.append(ID_COLUMN)
    X = df.drop(columns=drop_cols)

    # Validate minimum dataset requirements
    if len(X) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(X)} rows; at least {MIN_ROWS} required."
        )
    predictor_count = X.shape[1]
    if predictor_count < MIN_FEATURES:
        raise ValueError(
            f"Dataset has only {predictor_count} predictor features; "
            f"at least {MIN_FEATURES} required."
        )

    # Ensure numeric dtypes
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


# ---------------------------------------------------------------------------
# Preprocessing transformer
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles numeric and categorical features.

    Numeric pipeline  : median imputation -> StandardScaler
    Categorical pipeline: most-frequent imputation -> OneHotEncoder
                          (handle_unknown='ignore', sparse_output=False)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # drop any unexpected extra columns silently
    )
    return preprocessor


def build_model_pipeline(classifier) -> Pipeline:
    """
    Wrap a classifier in a full preprocessing + classification Pipeline.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Schema validation helpers (used by app.py)
# ---------------------------------------------------------------------------

def get_required_feature_columns() -> list:
    """Return the list of raw feature columns required for prediction."""
    return list(ALL_FEATURE_COLUMNS)


def validate_uploaded_dataframe(df: pd.DataFrame, required_cols: list) -> list:
    """
    Check that all required feature columns are present in the uploaded
    DataFrame.  Returns a list of missing column names (empty if all present).

    Extra columns (including accidental Unnamed index columns and optional
    customerID / Churn) are tolerated.
    """
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def separate_id_and_target(df: pd.DataFrame):
    """
    Separate optional customerID and Churn columns from the feature set.

    Returns
    -------
    X_raw    : DataFrame with feature columns only (no ID, no target)
    churn_col: Series or None - uploaded Churn labels if present
    id_col   : Series or None - customerID if present
    """
    from config import TARGET_COLUMN, ID_COLUMN

    churn_col = None
    id_col = None

    if TARGET_COLUMN in df.columns:
        raw_churn = df[TARGET_COLUMN].copy()
        # Accept numeric (0/1) or string (Yes/No) labels.
        # Use pd.api.types.is_numeric_dtype to correctly handle
        # ArrowStringArray / StringDtype that pandas uses in newer versions.
        if pd.api.types.is_numeric_dtype(raw_churn):
            int_vals = raw_churn.dropna().astype(int).unique()
            unexpected_int = set(int_vals) - {0, 1}
            if unexpected_int:
                raise ValueError(
                    f"Numeric Churn column contains unexpected values: {unexpected_int}. "
                    "Expected only 0 and 1."
                )
            churn_col = raw_churn.astype(int)
        else:
            churn_col = encode_target(raw_churn)

    if ID_COLUMN in df.columns:
        id_col = df[ID_COLUMN].copy()

    drop_cols = [c for c in [TARGET_COLUMN, ID_COLUMN] if c in df.columns]
    # Also drop accidental unnamed index columns
    unnamed = [c for c in df.columns if c.startswith("Unnamed:")]
    drop_cols.extend(unnamed)

    X_raw = df.drop(columns=drop_cols, errors="ignore")
    return X_raw, churn_col, id_col
