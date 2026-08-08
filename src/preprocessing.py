"""
preprocessing.py
-----------------
Feature engineering and the sklearn ColumnTransformer used by both
training and inference, so the exact same transformation is guaranteed
at prediction time (no train/serve skew).
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
    "satisfaction_score",
    "avg_monthly_usage_gb",
    "charges_per_tenure_month",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "tech_support",
    "online_security",
    "streaming_service",
    "paperless_billing",
    "payment_method",
]

TARGET = "churn"
DROP_COLS = ["customer_id"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Applied identically at train and inference time."""
    df = df.copy()
    # Average revenue per month of tenure — flags customers paying a lot
    # relative to how long they've stuck around (an early-warning signal).
    df["charges_per_tenure_month"] = df["total_charges"] / df["tenure_months"].clip(lower=1)
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_FEATURES),
        ]
    )


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df = engineer_features(df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df[TARGET] == "Yes").astype(int)
    return X, y, df
