"""
test_pipeline.py
-----------------
A small but real test suite: run with `pytest` from the project root.
Covers data integrity, the preprocessing contract, and end-to-end
prediction behavior on the saved model.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features, load_and_prepare
from src.predict import DEMO_CUSTOMER, score

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telecom_churn.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(DATA_PATH)


def test_data_has_no_missing_values(raw_df):
    assert raw_df.isna().sum().sum() == 0


def test_churn_column_is_binary_labels(raw_df):
    assert set(raw_df["churn"].unique()) == {"Yes", "No"}


def test_churn_rate_is_plausible(raw_df):
    rate = (raw_df["churn"] == "Yes").mean()
    # Real-world telecom churn rates typically fall in this range
    assert 0.10 < rate < 0.40


def test_engineered_feature_present():
    df = engineer_features(pd.DataFrame([{
        "total_charges": 100.0, "tenure_months": 5,
    }]))
    assert "charges_per_tenure_month" in df.columns
    assert df["charges_per_tenure_month"].iloc[0] == pytest.approx(20.0)


def test_engineered_feature_handles_zero_tenure():
    df = engineer_features(pd.DataFrame([{"total_charges": 50.0, "tenure_months": 0}]))
    # tenure is clipped to >= 1 to avoid division by zero
    assert df["charges_per_tenure_month"].iloc[0] == pytest.approx(50.0)


def test_load_and_prepare_shapes(raw_df):
    X, y, _ = load_and_prepare(str(DATA_PATH))
    assert len(X) == len(y) == len(raw_df)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - {"charges_per_tenure_month"} <= set(X.columns) | {"charges_per_tenure_month"}
    assert y.isin([0, 1]).all()


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="model not trained yet")
def test_saved_pipeline_predicts_valid_probabilities():
    pipeline = joblib.load(MODELS_DIR / "churn_pipeline.joblib")
    X, y, _ = load_and_prepare(str(DATA_PATH))
    proba = pipeline.predict_proba(X.head(20))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.skipif(not MODELS_DIR.exists(), reason="model not trained yet")
def test_demo_customer_scores_high_risk():
    result = score(pd.DataFrame([DEMO_CUSTOMER]), str(MODELS_DIR))
    # New, month-to-month, low satisfaction, high support-call customer
    # should clearly land in the upper risk range.
    assert result.loc[0, "churn_probability"] > 0.5
    assert result.loc[0, "risk_band"] in {"Medium", "High"}
