"""
train.py
--------
Trains and compares several models for churn prediction, selects the
best by cross-validated ROC-AUC, refits it on the full training split,
and persists the fitted pipeline + metadata for evaluation/inference.

Run:
    python -m src.train --data data/telecom_churn.csv
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, load_and_prepare

CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "param_grid": {"clf__C": [0.05, 0.1, 0.5, 1.0, 3.0]},
    },
    "random_forest": {
        "estimator": RandomForestClassifier(class_weight="balanced", random_state=42),
        "param_grid": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [5, 8, None],
            "clf__min_samples_leaf": [1, 3],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [150, 300],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [2, 3],
        },
    },
}


def main(data_path: str, models_dir: str):
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    X, y, _ = load_and_prepare(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    fitted_pipelines = {}

    for name, cfg in CANDIDATES.items():
        t0 = time.time()
        pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", cfg["estimator"]),
        ])
        search = GridSearchCV(
            pipe, cfg["param_grid"], scoring="roc_auc", cv=cv, n_jobs=-1
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0

        best_pipe = search.best_estimator_
        test_auc = float(
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(
                y_test, best_pipe.predict_proba(X_test)[:, 1]
            )
        )
        results[name] = {
            "cv_best_roc_auc": float(search.best_score_),
            "test_roc_auc": test_auc,
            "best_params": search.best_params_,
            "train_seconds": round(elapsed, 1),
        }
        fitted_pipelines[name] = best_pipe
        print(f"[{name}] CV ROC-AUC={search.best_score_:.4f}  Test ROC-AUC={test_auc:.4f}  ({elapsed:.1f}s)")

    best_name = max(results, key=lambda k: results[k]["test_roc_auc"])
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nSelected model: {best_name}")

    joblib.dump(best_pipeline, f"{models_dir}/churn_pipeline.joblib")
    joblib.dump(X_train.columns.tolist(), f"{models_dir}/feature_columns.joblib")

    with open(f"{models_dir}/train_results.json", "w") as f:
        json.dump({"results": results, "selected_model": best_name}, f, indent=2)

    # Stash test split for evaluate.py so metrics are computed on identical data
    X_test.assign(churn=y_test.values).to_csv(f"{models_dir}/_test_split.csv", index=False)

    print(f"Saved pipeline to {models_dir}/churn_pipeline.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telecom_churn.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()
    main(args.data, args.models_dir)
