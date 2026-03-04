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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
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
    silhouette_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.datasets import load_iris, load_diabetes

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
# 11. Decision Tree Classifier
# ---------------------------------------------------------------------------
def gen_decision_tree_classifier():
    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    importances = model.feature_importances_

    write_fixture(
        "decision_tree_classifier",
        {
            "description": "DecisionTreeClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"max_depth": 3, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(importances),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 12. Decision Tree Regressor
# ---------------------------------------------------------------------------
def gen_decision_tree_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "decision_tree_regressor",
        {
            "description": "DecisionTreeRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {"max_depth": 4, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(model.feature_importances_),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 13. Random Forest Classifier
# ---------------------------------------------------------------------------
def gen_random_forest_classifier():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X, y)
    preds = model.predict(X)
    importances = model.feature_importances_

    write_fixture(
        "random_forest_classifier",
        {
            "description": "RandomForestClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(importances),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 14. Random Forest Regressor
# ---------------------------------------------------------------------------
def gen_random_forest_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = RandomForestRegressor(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "random_forest_regressor",
        {
            "description": "RandomForestRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(model.feature_importances_),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 15. Gradient Boosting Classifier
# ---------------------------------------------------------------------------
def gen_gradient_boosting_classifier():
    X, y = load_iris(return_X_y=True)
    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "gradient_boosting_classifier",
        {
            "description": "GradientBoostingClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "expected": {
                "predictions": to_list(preds),
                "feature_importances": to_list(model.feature_importances_),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 16. Gradient Boosting Regressor
# ---------------------------------------------------------------------------
def gen_gradient_boosting_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "gradient_boosting_regressor",
        {
            "description": "GradientBoostingRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {
                "n_estimators": 50,
                "max_depth": 2,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "expected": {
                "predictions": to_list(preds),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 17. AdaBoost Classifier
# ---------------------------------------------------------------------------
def gen_adaboost_classifier():
    X, y = load_iris(return_X_y=True)
    model = AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42, algorithm="SAMME"
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "adaboost_classifier",
        {
            "description": "AdaBoostClassifier (SAMME) on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {
                "n_estimators": 50,
                "learning_rate": 1.0,
                "algorithm": "SAMME",
                "random_state": 42,
            },
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 18. KNeighbors Classifier
# ---------------------------------------------------------------------------
def gen_kneighbors_classifier():
    X, y = load_iris(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "kneighbors_classifier",
        {
            "description": "KNeighborsClassifier on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"n_neighbors": 5},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 19. KNeighbors Regressor
# ---------------------------------------------------------------------------
def gen_kneighbors_regressor():
    X, y = load_diabetes(return_X_y=True)
    X_small, y_small = X[:100], y[:100]
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_small, y_small)
    preds = model.predict(X_small)

    write_fixture(
        "kneighbors_regressor",
        {
            "description": "KNeighborsRegressor on diabetes (first 100) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"X": to_list(X_small), "y": to_list(y_small)},
            "params": {"n_neighbors": 5},
            "expected": {
                "predictions": to_list(preds),
                "r2": float(r2_score(y_small, preds)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 20. Gaussian Naive Bayes
# ---------------------------------------------------------------------------
def gen_gaussian_nb():
    X, y = load_iris(return_X_y=True)
    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "gaussian_nb",
        {
            "description": "GaussianNB on iris from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": None,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {},
            "expected": {
                "predictions": to_list(preds),
                "accuracy": float(accuracy_score(y, preds)),
                "class_prior": to_list(model.class_prior_),
                "theta": to_list(model.theta_),
                "var": to_list(model.var_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 21. KMeans
# ---------------------------------------------------------------------------
def gen_kmeans():
    rng = np.random.default_rng(42)
    # 3 well-separated clusters
    c0 = rng.standard_normal((50, 2)) + np.array([-5.0, -5.0])
    c1 = rng.standard_normal((50, 2)) + np.array([5.0, -5.0])
    c2 = rng.standard_normal((50, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c0, c1, c2])

    model = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_
    inertia = model.inertia_

    write_fixture(
        "kmeans",
        {
            "description": "KMeans (k=3) on synthetic blobs from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "random_state": 42, "n_init": 10},
            "expected": {
                "labels": to_list(labels),
                "cluster_centers": to_list(centers),
                "inertia": float(inertia),
                "n_iter": int(model.n_iter_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 22. DBSCAN
# ---------------------------------------------------------------------------
def gen_dbscan():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((30, 2)) + np.array([-5.0, 0.0])
    c1 = rng.standard_normal((30, 2)) + np.array([5.0, 0.0])
    noise = rng.uniform(-10, 10, size=(5, 2))
    X = np.vstack([c0, c1, noise])

    model = DBSCAN(eps=1.5, min_samples=5)
    labels = model.fit_predict(X)
    core_indices = model.core_sample_indices_

    write_fixture(
        "dbscan",
        {
            "description": "DBSCAN on synthetic data from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"eps": 1.5, "min_samples": 5},
            "expected": {
                "labels": to_list(labels),
                "core_sample_indices": to_list(core_indices),
                "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
                "n_noise": int(np.sum(labels == -1)),
            },
        },
    )


# ---------------------------------------------------------------------------
# 23. Agglomerative Clustering
# ---------------------------------------------------------------------------
def gen_agglomerative():
    rng = np.random.default_rng(42)
    c0 = rng.standard_normal((30, 2)) + np.array([-4.0, 0.0])
    c1 = rng.standard_normal((30, 2)) + np.array([4.0, 0.0])
    c2 = rng.standard_normal((30, 2)) + np.array([0.0, 6.0])
    X = np.vstack([c0, c1, c2])

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X)

    write_fixture(
        "agglomerative_clustering",
        {
            "description": "AgglomerativeClustering (ward, k=3) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_clusters": 3, "linkage": "ward"},
            "expected": {
                "labels": to_list(labels),
                "n_clusters": 3,
            },
        },
    )


# ---------------------------------------------------------------------------
# 24. PCA
# ---------------------------------------------------------------------------
def gen_pca():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))

    model = PCA(n_components=3)
    X_out = model.fit_transform(X)

    write_fixture(
        "pca",
        {
            "description": "PCA (3 components) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_components": 3},
            "expected": {
                "components": to_list(model.components_),
                "explained_variance": to_list(model.explained_variance_),
                "explained_variance_ratio": to_list(model.explained_variance_ratio_),
                "mean": to_list(model.mean_),
                "transformed": to_list(X_out),
            },
        },
    )


# ---------------------------------------------------------------------------
# 25. NMF
# ---------------------------------------------------------------------------
def gen_nmf():
    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal((40, 6)))  # NMF requires non-negative

    model = NMF(n_components=3, init="nndsvd", random_state=42, max_iter=500)
    W = model.fit_transform(X)
    H = model.components_

    write_fixture(
        "nmf",
        {
            "description": "NMF (3 components, nndsvd) from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X)},
            "params": {"n_components": 3, "init": "nndsvd", "random_state": 42},
            "expected": {
                "W": to_list(W),
                "H": to_list(H),
                "reconstruction_error": float(model.reconstruction_err_),
            },
        },
    )


# ---------------------------------------------------------------------------
# 26. ElasticNet
# ---------------------------------------------------------------------------
def gen_elastic_net():
    rng = np.random.default_rng(42)
    n, p = 50, 5
    X = rng.standard_normal((n, p))
    true_coef = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.standard_normal(n) * 0.1
    y = X @ true_coef + noise

    model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture(
        "elastic_net",
        {
            "description": "ElasticNet fixture from scikit-learn",
            "sklearn_version": SKLEARN_VERSION,
            "random_state": 42,
            "input": {"X": to_list(X), "y": to_list(y)},
            "params": {"alpha": 0.1, "l1_ratio": 0.5, "fit_intercept": True},
            "expected": {
                "coefficients": to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": to_list(preds),
            },
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
        ("decision_tree_classifier", gen_decision_tree_classifier),
        ("decision_tree_regressor", gen_decision_tree_regressor),
        ("random_forest_classifier", gen_random_forest_classifier),
        ("random_forest_regressor", gen_random_forest_regressor),
        ("gradient_boosting_classifier", gen_gradient_boosting_classifier),
        ("gradient_boosting_regressor", gen_gradient_boosting_regressor),
        ("adaboost_classifier", gen_adaboost_classifier),
        ("kneighbors_classifier", gen_kneighbors_classifier),
        ("kneighbors_regressor", gen_kneighbors_regressor),
        ("gaussian_nb", gen_gaussian_nb),
        ("kmeans", gen_kmeans),
        ("dbscan", gen_dbscan),
        ("agglomerative_clustering", gen_agglomerative),
        ("pca", gen_pca),
        ("nmf", gen_nmf),
        ("elastic_net", gen_elastic_net),
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
