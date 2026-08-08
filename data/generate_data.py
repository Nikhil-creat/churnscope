"""
generate_data.py
-----------------
Generates a synthetic but realistic telecom customer-churn dataset.

Why synthetic data: this project is built as a portfolio/reference
implementation, so the data-generating process (DGP) is fully documented
here instead of depending on a third-party CSV of unknown licensing.
The DGP encodes real churn drivers reported in telecom analytics
literature (contract type, tenure, support-call volume, satisfaction,
add-on services) so the resulting dataset is learnable and the model's
feature-importance results are interpretable and realistic.

Run:
    python generate_data.py [--n 6000] [--seed 42] [--out telecom_churn.csv]
"""
import argparse
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:06d}" for i in range(1, n + 1)]

    age = np.clip(rng.normal(45, 15, n), 18, 85).round().astype(int)

    tenure_months = np.clip(rng.exponential(scale=22, size=n), 0, 72).round().astype(int)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.55, 0.25, 0.20]
    )

    # Note: "No internet service" (not "None") — pandas' CSV reader treats the
    # literal string "None" as a missing value, which would silently corrupt
    # this column on reload.
    internet_service = rng.choice(
        ["Fiber optic", "DSL", "No internet service"],
        size=n, p=[0.44, 0.34, 0.22]
    )

    tech_support = rng.choice(["Yes", "No"], size=n, p=[0.38, 0.62])
    online_security = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    streaming = rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58])
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.60, 0.40])

    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n, p=[0.34, 0.19, 0.22, 0.25]
    )

    # Monthly charges built from a base fee + service add-ons + noise
    base = 20.0
    internet_addon = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL"],
        [45.0, 25.0], default=0.0
    )
    addon_fees = (
        (tech_support == "Yes") * 8.0
        + (online_security == "Yes") * 6.0
        + (streaming == "Yes") * 12.0
    )
    monthly_charges = np.round(
        base + internet_addon + addon_fees + rng.normal(0, 5, n), 2
    )
    monthly_charges = np.clip(monthly_charges, 18, None)

    total_charges = np.round(
        monthly_charges * np.maximum(tenure_months, 1) + rng.normal(0, 20, n), 2
    )
    total_charges = np.clip(total_charges, 0, None)

    # Support calls rise when satisfaction is low; generate jointly
    satisfaction_score = np.clip(rng.normal(3.4, 1.0, n), 1, 5).round().astype(int)
    support_call_lambda = np.clip(1.8 - 0.3 * satisfaction_score, 0.1, None)
    num_support_calls = rng.poisson(support_call_lambda, n)

    avg_monthly_usage_gb = np.clip(
        np.where(
            internet_service == "Fiber optic", rng.normal(320, 90, n),
            np.where(internet_service == "DSL", rng.normal(140, 50, n),
                     rng.normal(5, 3, n))
        ), 0, None
    ).round(1)

    # --- Churn generating process (logistic model with realistic drivers) ---
    logit = (
        -1.35
        + 1.30 * (contract_type == "Month-to-month")
        - 0.85 * (contract_type == "Two year")
        - 0.035 * tenure_months
        + 0.42 * num_support_calls
        - 0.38 * satisfaction_score
        + 0.012 * monthly_charges
        + 0.45 * (internet_service == "Fiber optic")
        - 0.40 * (tech_support == "Yes")
        - 0.28 * (online_security == "Yes")
        + 0.30 * (payment_method == "Electronic check")
        + rng.normal(0, 0.55, n)  # unexplained variance
    )
    churn_prob = sigmoid(logit)
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)
    churn_label = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "online_security": online_security,
        "streaming_service": streaming,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls": num_support_calls,
        "satisfaction_score": satisfaction_score,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "churn": churn_label,
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic telecom churn data")
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="telecom_churn.csv")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    out_path = args.out
    df.to_csv(out_path, index=False)
    churn_rate = (df["churn"] == "Yes").mean()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Churn rate: {churn_rate:.2%}")
