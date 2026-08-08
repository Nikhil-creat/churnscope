"""
evaluate.py
-----------
Loads the saved pipeline and the held-out test split, then produces:
  - classification report (precision/recall/F1)
  - confusion matrix plot
  - ROC curve plot
  - precision-recall curve plot
  - feature importance / coefficient plot
  - a business-framed summary (estimated retention value)

Run:
    python -m src.evaluate --models-dir models --figures-dir reports/figures
"""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def plot_confusion_matrix(y_test, y_pred, out_path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y_test, y_proba, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="Churn model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve — Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr(y_test, y_proba, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name="Churn model")
    ax.set_title("Precision-Recall Curve — Test Set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(pipeline, out_path):
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    feature_names = prep.get_feature_names_out()

    if hasattr(clf, "coef_"):
        importance = clf.coef_[0]
        title = "Logistic Regression Coefficients (standardized)"
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
        title = "Feature Importances"
    else:
        return None

    order = np.argsort(np.abs(importance))[-15:]
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#F2495C" if v > 0 else "#3DDC97" for v in importance[order]]
    ax.barh([feature_names[i] for i in order], importance[order], color=colors)
    ax.set_title(title)
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return dict(zip(
        [feature_names[i] for i in order][::-1],
        [float(importance[i]) for i in order][::-1],
    ))


def main(models_dir: str, figures_dir: str):
    pipeline = joblib.load(f"{models_dir}/churn_pipeline.joblib")
    test_df = pd.read_csv(f"{models_dir}/_test_split.csv")

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["churn"]

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"], output_dict=True)

    plot_confusion_matrix(y_test, y_pred, f"{figures_dir}/confusion_matrix.png")
    plot_roc(y_test, y_proba, f"{figures_dir}/roc_curve.png")
    plot_pr(y_test, y_proba, f"{figures_dir}/pr_curve.png")
    top_features = plot_feature_importance(pipeline, f"{figures_dir}/feature_importance.png")

    # Business framing: revenue at risk among customers flagged high-risk (p >= 0.5)
    flagged = y_proba >= 0.5
    avg_monthly_charge = test_df.loc[flagged, "monthly_charges"].mean() if flagged.sum() else 0.0
    annual_revenue_at_risk = flagged.sum() * avg_monthly_charge * 12

    summary = {
        "test_roc_auc": float(auc),
        "classification_report": report,
        "n_test": int(len(y_test)),
        "n_flagged_high_risk": int(flagged.sum()),
        "avg_monthly_charge_flagged": float(avg_monthly_charge),
        "estimated_annual_revenue_at_risk": float(annual_revenue_at_risk),
        "top_features": top_features,
    }

    with open(f"{models_dir}/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Test ROC-AUC: {auc:.4f}")
    print(f"Precision (Churned): {report['Churned']['precision']:.3f}")
    print(f"Recall (Churned):    {report['Churned']['recall']:.3f}")
    print(f"F1 (Churned):        {report['Churned']['f1-score']:.3f}")
    print(f"High-risk customers flagged: {flagged.sum()} / {len(y_test)}")
    print(f"Estimated annual revenue at risk among flagged customers: ${annual_revenue_at_risk:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()
    main(args.models_dir, args.figures_dir)
