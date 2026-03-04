#!/usr/bin/env python3
"""Generate golden test fixtures from scikit-learn for ferrolearn oracle tests.

Run this script to regenerate fixtures after scikit-learn upgrades or when
adding new test cases. Output JSON files are written to the fixtures/ directory
relative to the project root.

Usage:
    python scripts/generate_fixtures.py
"""

import json
import os
import sys

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

SKLEARN_VERSION = sklearn.__version__

# Output directory is fixtures/ relative to this script's parent (project root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "fixtures")

os.makedirs(FIXTURES_DIR, exist_ok=True)


def to_list(arr):
    """Convert a numpy array to a plain Python list (nested for 2-D)."""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def write_fixture(name: str, data: dict) -> None:
    path = os.path.join(FIXTURES_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# 1. Linear Regression
# ---------------------------------------------------------------------------
def gen_linear_regression():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "linear_regression",
        {
            "description": "LinearRegression fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# 2. Ridge Regression
# ---------------------------------------------------------------------------
def gen_ridge():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "ridge",
        {
            "description": "Ridge regression fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 1.0, "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# 3. Lasso
# ---------------------------------------------------------------------------
def gen_lasso():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = Lasso(alpha=0.1, fit_intercept=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "lasso",
        {
            "description": "Lasso regression fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 0.1, "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# 4. Logistic Regression  (binary, iris-like 100×4)
# ---------------------------------------------------------------------------
def gen_logistic_regression():
    rng = np.random.default_rng(42)
    n, p = 100, 4
    # Two separable blobs in 4-D
    X0 = rng.standard_normal((n // 2, p)) + np.array([-1.0, -1.0, -1.0, -1.0])
    X1 = rng.standard_normal((n // 2, p)) + np.array([1.0, 1.0, 1.0, 1.0])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    model.fit(X, y)
    pred_classes = model.predict(X)
    pred_proba = model.predict_proba(X)

    write_fixture(
        "logistic_regression",
        {
            "description": "LogisticRegression binary fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"C": 1.0, "solver": "lbfgs", "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": to_list(model.intercept_),
                "predicted_classes": to_list(pred_classes),
                "predicted_proba": to_list(pred_proba),
            },
        },
    )


# ---------------------------------------------------------------------------
# 5. StandardScaler
# ---------------------------------------------------------------------------
def gen_standard_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3  # non-zero mean/std

    scaler = StandardScaler()
    X_out = scaler.fit_transform(X)

    write_fixture(
        "standard_scaler",
        {
            "description": "StandardScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"with_mean": True, "with_std": True},
            "expected": {
                "mean": to_list(scaler.mean_),
                "std": to_list(scaler.scale_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 6. MinMaxScaler
# ---------------------------------------------------------------------------
def gen_minmax_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_out = scaler.fit_transform(X)

    write_fixture(
        "minmax_scaler",
        {
            "description": "MinMaxScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"feature_range": [0.0, 1.0]},
            "expected": {
                "data_min": to_list(scaler.data_min_),
                "data_max": to_list(scaler.data_max_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 7. RobustScaler
# ---------------------------------------------------------------------------
def gen_robust_scaler():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 4)) * 5 + 3

    scaler = RobustScaler()
    X_out = scaler.fit_transform(X)

    write_fixture(
        "robust_scaler",
        {
            "description": "RobustScaler fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"with_centering": True, "with_scaling": True},
            "expected": {
                "center": to_list(scaler.center_),
                "scale": to_list(scaler.scale_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 8. Classification metrics
# ---------------------------------------------------------------------------
def gen_classification_metrics():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    # Introduce ~10% errors for non-trivial metrics
    flip_mask = rng.random(n) < 0.10
    y_pred = y_true.copy()
    y_pred[flip_mask] = 1 - y_pred[flip_mask]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    write_fixture(
        "classification_metrics",
        {
            "description": "Binary classification metrics fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_pred": to_list(y_pred)},
            "params": {"average": "binary"},
            "expected": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "confusion_matrix": to_list(cm),
            },
        },
    )


# ---------------------------------------------------------------------------
# 9. Regression metrics
# ---------------------------------------------------------------------------
def gen_regression_metrics():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.standard_normal(n) * 10
    noise = rng.standard_normal(n) * 2
    y_pred = y_true + noise

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)

    write_fixture(
        "regression_metrics",
        {
            "description": "Regression metrics fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"y_true": to_list(y_true), "y_pred": to_list(y_pred)},
            "params": {},
            "expected": {
                "mae": float(mae),
                "mse": float(mse),
                "rmse": rmse,
                "r2": float(r2),
            },
        },
    )


# ---------------------------------------------------------------------------
# 10. KFold cross-validation indices
# ---------------------------------------------------------------------------
def gen_kfold():
    n_samples = 100
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=False)

    folds = []
    indices = np.arange(n_samples)
    for train_idx, test_idx in kf.split(indices):
        folds.append(
            {
                "train": to_list(train_idx),
                "test": to_list(test_idx),
            }
        )

    write_fixture(
        "kfold",
        {
            "description": "KFold cross-validation index fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"n_samples": n_samples},
            "params": {"n_splits": n_splits, "shuffle": False},
            "expected": {"folds": folds},
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"scikit-learn version: {SKLEARN_VERSION}")
    print(f"Writing fixtures to: {FIXTURES_DIR}")
    print()

    generators = [
        ("linear_regression", gen_linear_regression),
        ("ridge", gen_ridge),
        ("lasso", gen_lasso),
        ("logistic_regression", gen_logistic_regression),
        ("standard_scaler", gen_standard_scaler),
        ("minmax_scaler", gen_minmax_scaler),
        ("robust_scaler", gen_robust_scaler),
        ("classification_metrics", gen_classification_metrics),
        ("regression_metrics", gen_regression_metrics),
        ("kfold", gen_kfold),
    ]

    errors = []
    for name, fn in generators:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR generating {name}: {exc}", file=sys.stderr)
            errors.append(name)

    print()
    if errors:
        print(f"FAILED: {', '.join(errors)}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All {len(generators)} fixtures generated successfully.")


if __name__ == "__main__":
    main()
