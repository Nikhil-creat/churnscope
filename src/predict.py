"""
predict.py
----------
Loads the saved pipeline and scores new customer records.

Run on a CSV of new customers (same columns as the training data, minus churn):
    python -m src.predict --input new_customers.csv --output scored.csv

Or score one customer inline for a quick check:
    python -m src.predict --demo
"""
import argparse

import joblib
import pandas as pd

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, engineer_features

RISK_BANDS = [(0.66, "High"), (0.33, "Medium"), (0.0, "Low")]


def band(prob: float) -> str:
    for threshold, label in RISK_BANDS:
        if prob >= threshold:
            return label
    return "Low"


def score(df_raw: pd.DataFrame, models_dir: str) -> pd.DataFrame:
    pipeline = joblib.load(f"{models_dir}/churn_pipeline.joblib")
    df = engineer_features(df_raw)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = pipeline.predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["churn_probability"] = proba.round(4)
    out["risk_band"] = [band(p) for p in proba]
    return out


DEMO_CUSTOMER = {
    "customer_id": "CUST-DEMO",
    "age": 34,
    "tenure_months": 3,
    "contract_type": "Month-to-month",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "online_security": "No",
    "streaming_service": "Yes",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.50,
    "total_charges": 268.50,
    "num_support_calls": 3,
    "satisfaction_score": 2,
    "avg_monthly_usage_gb": 340.0,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="CSV of customers to score")
    parser.add_argument("--output", type=str, default="scored.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--demo", action="store_true", help="Score one built-in example customer")
    args = parser.parse_args()

    if args.demo:
        result = score(pd.DataFrame([DEMO_CUSTOMER]), args.models_dir)
        prob = result.loc[0, "churn_probability"]
        risk = result.loc[0, "risk_band"]
        print(f"Demo customer churn probability: {prob:.1%}  (risk band: {risk})")
    elif args.input:
        df_raw = pd.read_csv(args.input)
        result = score(df_raw, args.models_dir)
        result.to_csv(args.output, index=False)
        print(f"Scored {len(result)} customers -> {args.output}")
    else:
        parser.error("Provide --input <csv> or --demo")
