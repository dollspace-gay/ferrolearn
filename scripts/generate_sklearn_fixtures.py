#!/usr/bin/env python3
"""Generate sklearn reference fixtures for statistical equivalence tests.

This script produces JSON fixture files that capture sklearn's exact outputs
for various models and metrics. The Rust equivalence tests load these fixtures
and verify that ferrolearn produces numerically identical (within tolerance)
results.

Usage:
    python scripts/generate_sklearn_fixtures.py
"""

import json
import os
import sys

import numpy as np
import sklearn
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_regression,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
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
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

SKLEARN_VERSION = sklearn.__version__

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures", "sklearn_reference")

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


# ============================================================================
# Dataset generators (shared across models)
# ============================================================================

def make_regression_default():
    """Standard regression dataset."""
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=42
    )
    return X, y


def make_regression_single_feature():
    """Single-feature regression."""
    X, y = make_regression(
        n_samples=50, n_features=1, noise=0.1, random_state=42
    )
    return X, y


def make_classification_default():
    """Standard binary classification dataset."""
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3,
        n_redundant=1, n_clusters_per_class=1, random_state=42
    )
    return X, y


def make_classification_multiclass():
    """Multiclass classification dataset."""
    X, y = make_classification(
        n_samples=60, n_features=5, n_informative=3,
        n_redundant=1, n_classes=3, n_clusters_per_class=1,
        random_state=42
    )
    return X, y


def make_blobs_default():
    """Standard blobs dataset for clustering/transformers."""
    X, y = make_blobs(
        n_samples=50, n_features=5, centers=3, random_state=42
    )
    return X, y


# ============================================================================
# 1. LINEAR REGRESSION
# ============================================================================

def gen_linear_regression():
    X, y = make_regression_default()

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    # Also test on a held-out portion
    X_test = X[:10]
    preds_test = model.predict(X_test)

    write_fixture("linear_regression", {
        "model": "LinearRegression",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"fit_intercept": True},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X_test),
        "coef": to_list(model.coef_),
        "intercept": float(model.intercept_),
        "predictions_train": to_list(preds),
        "predictions_test": to_list(preds_test),
        "r2_score": float(r2_score(y, preds)),
    })


def gen_linear_regression_single_feature():
    X, y = make_regression_single_feature()

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("linear_regression_single_feature", {
        "model": "LinearRegression",
        "dataset": "make_regression_single_feature",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"fit_intercept": True},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "coef": to_list(model.coef_),
        "intercept": float(model.intercept_),
        "predictions_train": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "r2_score": float(r2_score(y, preds)),
    })


# ============================================================================
# 2. RIDGE REGRESSION
# ============================================================================

def gen_ridge():
    X, y = make_regression_default()

    model = Ridge(alpha=1.0, fit_intercept=True)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("ridge", {
        "model": "Ridge",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"alpha": 1.0, "fit_intercept": True},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "coef": to_list(model.coef_),
        "intercept": float(model.intercept_),
        "predictions_train": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "r2_score": float(r2_score(y, preds)),
    })


# ============================================================================
# 3. LASSO
# ============================================================================

def gen_lasso():
    X, y = make_regression_default()

    model = Lasso(alpha=0.1, fit_intercept=True, max_iter=10000, tol=1e-8, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("lasso", {
        "model": "Lasso",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"alpha": 0.1, "fit_intercept": True, "max_iter": 10000, "tol": 1e-8},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "coef": to_list(model.coef_),
        "intercept": float(model.intercept_),
        "predictions_train": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "r2_score": float(r2_score(y, preds)),
    })


# ============================================================================
# 4. ELASTIC NET
# ============================================================================

def gen_elastic_net():
    X, y = make_regression_default()

    model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True,
                       max_iter=10000, tol=1e-8, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("elastic_net", {
        "model": "ElasticNet",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"alpha": 0.1, "l1_ratio": 0.5, "fit_intercept": True,
                   "max_iter": 10000, "tol": 1e-8},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "coef": to_list(model.coef_),
        "intercept": float(model.intercept_),
        "predictions_train": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "r2_score": float(r2_score(y, preds)),
    })


# ============================================================================
# 5. LOGISTIC REGRESSION
# ============================================================================

def gen_logistic_regression():
    X, y = make_classification_default()

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                               fit_intercept=True, random_state=42)
    model.fit(X, y)
    pred_classes = model.predict(X)
    pred_proba = model.predict_proba(X)

    write_fixture("logistic_regression", {
        "model": "LogisticRegression",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"C": 1.0, "solver": "lbfgs", "fit_intercept": True, "max_iter": 1000},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "coef": to_list(model.coef_),
        "intercept": to_list(model.intercept_),
        "predicted_classes": to_list(pred_classes),
        "predicted_proba": to_list(pred_proba),
        "accuracy": float(accuracy_score(y, pred_classes)),
    })


# ============================================================================
# 6. DECISION TREE CLASSIFIER
# ============================================================================

def gen_decision_tree_classifier():
    X, y = make_classification_default()

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("decision_tree_classifier", {
        "model": "DecisionTreeClassifier",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"max_depth": 3, "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "feature_importances": to_list(model.feature_importances_),
        "accuracy": float(accuracy_score(y, preds)),
    })


# ============================================================================
# 7. DECISION TREE REGRESSOR
# ============================================================================

def gen_decision_tree_regressor():
    X, y = make_regression_default()

    model = DecisionTreeRegressor(max_depth=4, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("decision_tree_regressor", {
        "model": "DecisionTreeRegressor",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"max_depth": 4, "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "feature_importances": to_list(model.feature_importances_),
        "r2": float(r2_score(y, preds)),
    })


# ============================================================================
# 8. RANDOM FOREST CLASSIFIER
# ============================================================================

def gen_random_forest_classifier():
    X, y = make_classification_default()

    model = RandomForestClassifier(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("random_forest_classifier", {
        "model": "RandomForestClassifier",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "feature_importances": to_list(model.feature_importances_),
        "accuracy": float(accuracy_score(y, preds)),
    })


# ============================================================================
# 9. RANDOM FOREST REGRESSOR
# ============================================================================

def gen_random_forest_regressor():
    X, y = make_regression_default()

    model = RandomForestRegressor(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=1
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("random_forest_regressor", {
        "model": "RandomForestRegressor",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "feature_importances": to_list(model.feature_importances_),
        "r2": float(r2_score(y, preds)),
    })


# ============================================================================
# 10. GRADIENT BOOSTING CLASSIFIER
# ============================================================================

def gen_gradient_boosting_classifier():
    X, y = make_classification_default()

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("gradient_boosting_classifier", {
        "model": "GradientBoostingClassifier",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1, "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "feature_importances": to_list(model.feature_importances_),
        "accuracy": float(accuracy_score(y, preds)),
    })


# ============================================================================
# 11. GRADIENT BOOSTING REGRESSOR
# ============================================================================

def gen_gradient_boosting_regressor():
    X, y = make_regression_default()

    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("gradient_boosting_regressor", {
        "model": "GradientBoostingRegressor",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1, "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "r2": float(r2_score(y, preds)),
    })


# ============================================================================
# 12. ADABOOST CLASSIFIER
# ============================================================================

def gen_adaboost_classifier():
    X, y = make_classification_default()

    model = AdaBoostClassifier(
        n_estimators=50, learning_rate=1.0, random_state=42, algorithm="SAMME"
    )
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("adaboost_classifier", {
        "model": "AdaBoostClassifier",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_estimators": 50, "learning_rate": 1.0, "algorithm": "SAMME", "random_state": 42},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "accuracy": float(accuracy_score(y, preds)),
    })


# ============================================================================
# 13. K-NEIGHBORS CLASSIFIER
# ============================================================================

def gen_kneighbors_classifier():
    X, y = make_classification_default()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("kneighbors_classifier", {
        "model": "KNeighborsClassifier",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_neighbors": 5},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "accuracy": float(accuracy_score(y, preds)),
    })


# ============================================================================
# 14. K-NEIGHBORS REGRESSOR
# ============================================================================

def gen_kneighbors_regressor():
    X, y = make_regression_default()

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X, y)
    preds = model.predict(X)

    write_fixture("kneighbors_regressor", {
        "model": "KNeighborsRegressor",
        "dataset": "make_regression_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_neighbors": 5},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "r2": float(r2_score(y, preds)),
    })


# ============================================================================
# 15. GAUSSIAN NAIVE BAYES
# ============================================================================

def gen_gaussian_nb():
    X, y = make_classification_default()

    model = GaussianNB()
    model.fit(X, y)
    preds = model.predict(X)
    pred_proba = model.predict_proba(X)

    write_fixture("gaussian_nb", {
        "model": "GaussianNB",
        "dataset": "make_classification_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {},
        "X_train": to_list(X),
        "y_train": to_list(y),
        "X_test": to_list(X[:10]),
        "predictions": to_list(preds),
        "predictions_test": to_list(model.predict(X[:10])),
        "predicted_proba": to_list(pred_proba),
        "accuracy": float(accuracy_score(y, preds)),
        "class_prior": to_list(model.class_prior_),
        "theta": to_list(model.theta_),
        "var": to_list(model.var_),
    })


# ============================================================================
# 16. KMEANS
# ============================================================================

def gen_kmeans():
    X, y = make_blobs_default()

    model = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    model.fit(X)

    write_fixture("kmeans", {
        "model": "KMeans",
        "dataset": "make_blobs_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_clusters": 3, "random_state": 42, "n_init": 10},
        "X_train": to_list(X),
        "labels": to_list(model.labels_),
        "cluster_centers": to_list(model.cluster_centers_),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
    })


# ============================================================================
# 17. DBSCAN
# ============================================================================

def gen_dbscan():
    X, _ = make_blobs_default()

    model = DBSCAN(eps=3.0, min_samples=5)
    labels = model.fit_predict(X)

    write_fixture("dbscan", {
        "model": "DBSCAN",
        "dataset": "make_blobs_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"eps": 3.0, "min_samples": 5},
        "X_train": to_list(X),
        "labels": to_list(labels),
        "core_sample_indices": to_list(model.core_sample_indices_),
        "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
        "n_noise": int(np.sum(labels == -1)),
    })


# ============================================================================
# 18. AGGLOMERATIVE CLUSTERING
# ============================================================================

def gen_agglomerative():
    X, _ = make_blobs_default()

    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X)

    write_fixture("agglomerative_clustering", {
        "model": "AgglomerativeClustering",
        "dataset": "make_blobs_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_clusters": 3, "linkage": "ward"},
        "X_train": to_list(X),
        "labels": to_list(labels),
        "n_clusters": 3,
    })


# ============================================================================
# 19. PCA
# ============================================================================

def gen_pca():
    X, _ = make_blobs_default()

    model = PCA(n_components=3)
    X_transformed = model.fit_transform(X)

    write_fixture("pca", {
        "model": "PCA",
        "dataset": "make_blobs_default",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_components": 3},
        "X_train": to_list(X),
        "components": to_list(model.components_),
        "explained_variance": to_list(model.explained_variance_),
        "explained_variance_ratio": to_list(model.explained_variance_ratio_),
        "mean": to_list(model.mean_),
        "transformed": to_list(X_transformed),
    })


# ============================================================================
# 20. NMF
# ============================================================================

def gen_nmf():
    # NMF requires non-negative data
    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal((50, 5)))

    model = NMF(n_components=3, init="nndsvd", random_state=42, max_iter=500)
    W = model.fit_transform(X)
    H = model.components_

    write_fixture("nmf", {
        "model": "NMF",
        "dataset": "abs_random_50x5",
        "sklearn_version": SKLEARN_VERSION,
        "params": {"n_components": 3, "init": "nndsvd", "random_state": 42, "max_iter": 500},
        "X_train": to_list(X),
        "W": to_list(W),
        "H": to_list(H),
        "reconstruction_error": float(model.reconstruction_err_),
    })


# ============================================================================
# 21. METRICS -- Classification
# ============================================================================

def gen_classification_metrics():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, 2, size=n)
    # Introduce ~15% errors for non-trivial metrics
    flip_mask = rng.random(n) < 0.15
    y_pred = y_true.copy()
    y_pred[flip_mask] = 1 - y_pred[flip_mask]

    # For ROC AUC, we need probability scores
    y_scores = rng.random(n)
    # Make scores correlate with true labels
    y_scores = y_scores * 0.5 + y_true * 0.5

    acc = accuracy_score(y_true, y_pred)
    prec_binary = precision_score(y_true, y_pred, zero_division=0)
    rec_binary = recall_score(y_true, y_pred, zero_division=0)
    f1_binary = f1_score(y_true, y_pred, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Log loss requires probability arrays
    y_proba = np.column_stack([1 - y_scores, y_scores])
    ll = log_loss(y_true, y_proba)

    write_fixture("classification_metrics", {
        "description": "Classification metric reference values",
        "sklearn_version": SKLEARN_VERSION,
        "y_true": to_list(y_true),
        "y_pred": to_list(y_pred),
        "y_scores": to_list(y_scores),
        "y_proba": to_list(y_proba),
        "accuracy": float(acc),
        "precision_binary": float(prec_binary),
        "recall_binary": float(rec_binary),
        "f1_binary": float(f1_binary),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": to_list(cm),
        "roc_auc": float(roc_auc),
        "log_loss": float(ll),
    })


# ============================================================================
# 22. METRICS -- Regression
# ============================================================================

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
    mape = mean_absolute_percentage_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)

    write_fixture("regression_metrics", {
        "description": "Regression metric reference values",
        "sklearn_version": SKLEARN_VERSION,
        "y_true": to_list(y_true),
        "y_pred": to_list(y_pred),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": rmse,
        "r2": float(r2),
        "mape": float(mape),
        "explained_variance": float(evs),
    })


# ============================================================================
# 23. METRICS -- Clustering
# ============================================================================

def gen_clustering_metrics():
    X, y_true = make_blobs_default()

    # Generate a slightly perturbed labeling
    rng = np.random.default_rng(42)
    y_pred = y_true.copy()
    flip_mask = rng.random(len(y_true)) < 0.1
    y_pred[flip_mask] = (y_pred[flip_mask] + 1) % 3

    sil = silhouette_score(X, y_true)

    from sklearn.metrics import adjusted_rand_score as sk_ari
    from sklearn.metrics import adjusted_mutual_info_score as sk_ami
    from sklearn.metrics import davies_bouldin_score as sk_dbi

    ari = sk_ari(y_true, y_pred)
    ami = sk_ami(y_true, y_pred)
    dbi = sk_dbi(X, y_true)

    write_fixture("clustering_metrics", {
        "description": "Clustering metric reference values",
        "sklearn_version": SKLEARN_VERSION,
        "X": to_list(X),
        "labels_true": to_list(y_true),
        "labels_pred": to_list(y_pred),
        "silhouette_score": float(sil),
        "adjusted_rand_score": float(ari),
        "adjusted_mutual_info": float(ami),
        "davies_bouldin_score": float(dbi),
    })


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"scikit-learn version: {SKLEARN_VERSION}")
    print(f"Writing fixtures to: {FIXTURES_DIR}")
    print()

    generators = [
        ("linear_regression", gen_linear_regression),
        ("linear_regression_single_feature", gen_linear_regression_single_feature),
        ("ridge", gen_ridge),
        ("lasso", gen_lasso),
        ("elastic_net", gen_elastic_net),
        ("logistic_regression", gen_logistic_regression),
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
        ("classification_metrics", gen_classification_metrics),
        ("regression_metrics", gen_regression_metrics),
        ("clustering_metrics", gen_clustering_metrics),
    ]

    errors = []
    for name, fn in generators:
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR generating {name}: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            errors.append(name)

    print()
    if errors:
        print(f"FAILED: {', '.join(errors)}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"All {len(generators)} fixtures generated successfully.")


if __name__ == "__main__":
    main()
