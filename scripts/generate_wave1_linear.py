#!/usr/bin/env python3
"""Wave 1 — sklearn reference fixtures for previously-untested linear models.

Adds fixtures for: HuberRegressor, BayesianRidge, ARDRegression,
QuantileRegressor, Lars, LassoLars, OrthogonalMatchingPursuit,
RidgeCV, LassoCV, ElasticNetCV, LinearDiscriminantAnalysis (LDA),
QuadraticDiscriminantAnalysis (QDA), RidgeClassifier, LinearSVC, LinearSVR,
SVC, SVR, NuSVC, NuSVR, OneClassSVM, SGDClassifier, SGDRegressor,
RANSACRegressor, IsotonicRegression, PoissonRegressor, GammaRegressor,
TweedieRegressor, LogisticRegressionCV.

All fixtures pin sklearn version + random_state. Iterative models
(Bayesian*, SVMs, SGD, RANSAC) use seeded RNG.

Usage:
    python3 scripts/generate_wave1_linear.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_regression,
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNetCV,
    GammaRegressor,
    HuberRegressor,
    Lars,
    LassoCV,
    LassoLars,
    LogisticRegressionCV,
    OrthogonalMatchingPursuit,
    PoissonRegressor,
    QuantileRegressor,
    RANSACRegressor,
    RidgeClassifier,
    RidgeCV,
    SGDClassifier,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.svm import (
    NuSVC,
    NuSVR,
    OneClassSVM,
    SVC,
    SVR,
    LinearSVC,
    LinearSVR,
)

SKLEARN_VERSION = sklearn.__version__
SKLEARN_PIN = f"scikit-learn=={SKLEARN_VERSION}"

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
FIXTURES.mkdir(exist_ok=True)


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def write_fixture(name: str, payload: dict) -> None:
    payload.setdefault("sklearn_version", SKLEARN_VERSION)
    payload.setdefault("sklearn_pin", SKLEARN_PIN)
    out = FIXTURES / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")


# Shared datasets — reproducible via random_state.
def reg_data(seed=42, n=50, p=5, noise=0.1):
    return make_regression(n_samples=n, n_features=p, noise=noise, random_state=seed)


def clf_data(seed=42, n=100, p=5, n_classes=2, n_informative=4):
    return make_classification(
        n_samples=n,
        n_features=p,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Robust regressors
# ---------------------------------------------------------------------------

def gen_huber_regressor() -> None:
    X, y = reg_data()
    model = HuberRegressor(max_iter=200, alpha=1e-4, epsilon=1.35)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "huber_regressor",
        {
            "description": "HuberRegressor — robust IRLS",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"max_iter": 200, "alpha": 1e-4, "epsilon": 1.35},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
                "scale": float(model.scale_),
            },
        },
    )


def gen_ransac() -> None:
    X, y = reg_data(n=80)
    # Inject some outliers so RANSAC has something to ignore.
    rng = np.random.RandomState(7)
    n_outliers = 10
    outlier_idx = rng.choice(len(y), size=n_outliers, replace=False)
    y_perturbed = y.copy()
    y_perturbed[outlier_idx] += rng.normal(scale=5 * np.std(y), size=n_outliers)
    model = RANSACRegressor(random_state=42, min_samples=0.5)
    model.fit(X, y_perturbed)
    preds = model.predict(X)
    write_fixture(
        "ransac_regressor",
        {
            "description": "RANSACRegressor — random sample consensus over LinearRegression",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y_perturbed)},
            "params": {"min_samples": 0.5, "random_state": 42},
            "expected": {
                "predictions": _to_list(preds),
                "inlier_mask": _to_list(model.inlier_mask_.astype(int)),
                "n_trials": int(model.n_trials_),
            },
        },
    )


def gen_quantile_regressor() -> None:
    X, y = reg_data(n=60)
    model = QuantileRegressor(quantile=0.5, alpha=0.01, solver="highs")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "quantile_regressor",
        {
            "description": "QuantileRegressor — median (q=0.5) via interior point",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"quantile": 0.5, "alpha": 0.01},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# Bayesian / probabilistic linear regressors
# ---------------------------------------------------------------------------

def gen_bayesian_ridge() -> None:
    X, y = reg_data()
    model = BayesianRidge(max_iter=300, tol=1e-3)
    model.fit(X, y)
    preds, std = model.predict(X, return_std=True)
    write_fixture(
        "bayesian_ridge",
        {
            "description": "BayesianRidge — evidence-maximization with auto hyperparam",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"max_iter": 300, "tol": 1e-3},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
                "predict_std": _to_list(std),
                "alpha": float(model.alpha_),
                "lambda": float(model.lambda_),
            },
        },
    )


def gen_ard_regression() -> None:
    X, y = reg_data()
    model = ARDRegression(max_iter=300, tol=1e-3)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "ard_regression",
        {
            "description": "ARDRegression — automatic relevance determination",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"max_iter": 300, "tol": 1e-3},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
                "alpha": float(model.alpha_),
                "lambdas": _to_list(model.lambda_),
            },
        },
    )


# ---------------------------------------------------------------------------
# Greedy / forward-selection regressors
# ---------------------------------------------------------------------------

def gen_lars() -> None:
    X, y = reg_data(n=60, p=10)
    model = Lars(n_nonzero_coefs=5)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "lars",
        {
            "description": "Lars — least angle regression up to 5 nonzero coefficients",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"n_nonzero_coefs": 5},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_lasso_lars() -> None:
    X, y = reg_data(n=60, p=10)
    model = LassoLars(alpha=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "lasso_lars",
        {
            "description": "LassoLars — LARS-Lasso variant with alpha=0.1",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"alpha": 0.1},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_omp() -> None:
    X, y = reg_data(n=60, p=10)
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=4)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "orthogonal_matching_pursuit",
        {
            "description": "OMP — greedy with 4 nonzero coefficients",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"n_nonzero_coefs": 4},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# Cross-validated linear regressors
# ---------------------------------------------------------------------------

def gen_ridge_cv() -> None:
    X, y = reg_data(n=80)
    alphas = (0.1, 1.0, 10.0)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "ridge_cv",
        {
            "description": "RidgeCV over [0.1, 1.0, 10.0] with 5-fold CV",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"alphas": list(alphas), "cv": 5},
            "expected": {
                "alpha": float(model.alpha_),
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_lasso_cv() -> None:
    X, y = reg_data(n=80)
    model = LassoCV(cv=5, random_state=42, max_iter=2000)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "lasso_cv",
        {
            "description": "LassoCV with 5-fold and seeded RNG",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"cv": 5, "random_state": 42, "max_iter": 2000},
            "expected": {
                "alpha": float(model.alpha_),
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_elastic_net_cv() -> None:
    X, y = reg_data(n=80)
    model = ElasticNetCV(cv=5, random_state=42, max_iter=2000)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "elastic_net_cv",
        {
            "description": "ElasticNetCV with 5-fold and seeded RNG",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"cv": 5, "random_state": 42, "max_iter": 2000},
            "expected": {
                "alpha": float(model.alpha_),
                "l1_ratio": float(model.l1_ratio_),
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_logistic_regression_cv() -> None:
    X, y = clf_data(n=120, n_classes=2)
    model = LogisticRegressionCV(cv=5, random_state=42, max_iter=2000)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    write_fixture(
        "logistic_regression_cv",
        {
            "description": "LogisticRegressionCV — 5-fold over default Cs",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"cv": 5, "random_state": 42, "max_iter": 2000},
            "expected": {
                "C": _to_list(model.C_),
                "coefficients": _to_list(model.coef_),
                "intercept": _to_list(model.intercept_),
                "predicted_classes": _to_list(preds),
                "predicted_proba": _to_list(proba),
            },
        },
    )


# ---------------------------------------------------------------------------
# Discriminant analysis (classifiers in linear crate)
# ---------------------------------------------------------------------------

def gen_lda() -> None:
    X, y = clf_data(n=120, n_classes=3, n_informative=4)
    model = LinearDiscriminantAnalysis()
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    write_fixture(
        "lda",
        {
            "description": "LinearDiscriminantAnalysis — default SVD solver",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {},
            "expected": {
                "classes": _to_list(model.classes_),
                "predicted_classes": _to_list(preds),
                "predicted_proba": _to_list(proba),
                "means": _to_list(model.means_),
                "priors": _to_list(model.priors_),
            },
        },
    )


def gen_qda() -> None:
    X, y = clf_data(n=120, n_classes=3, n_informative=4)
    model = QuadraticDiscriminantAnalysis()
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    write_fixture(
        "qda",
        {
            "description": "QuadraticDiscriminantAnalysis — per-class covariance",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {},
            "expected": {
                "classes": _to_list(model.classes_),
                "predicted_classes": _to_list(preds),
                "predicted_proba": _to_list(proba),
                "means": _to_list(model.means_),
                "priors": _to_list(model.priors_),
            },
        },
    )


def gen_ridge_classifier() -> None:
    X, y = clf_data(n=120, n_classes=2)
    model = RidgeClassifier(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "ridge_classifier",
        {
            "description": "RidgeClassifier with alpha=1.0",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"alpha": 1.0},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": _to_list(model.intercept_),
                "predicted_classes": _to_list(preds),
                "classes": _to_list(model.classes_),
            },
        },
    )


# ---------------------------------------------------------------------------
# SVM family
# ---------------------------------------------------------------------------

def gen_linear_svc() -> None:
    X, y = clf_data(n=120, n_classes=2)
    model = LinearSVC(C=1.0, max_iter=2000, random_state=42, dual="auto")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "linear_svc",
        {
            "description": "LinearSVC with hinge + L2 penalty",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"C": 1.0, "max_iter": 2000, "random_state": 42},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": _to_list(model.intercept_),
                "predicted_classes": _to_list(preds),
            },
        },
    )


def gen_linear_svr() -> None:
    X, y = reg_data(n=80)
    model = LinearSVR(C=1.0, epsilon=0.1, max_iter=2000, random_state=42, dual="auto")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "linear_svr",
        {
            "description": "LinearSVR with epsilon-insensitive loss",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"C": 1.0, "epsilon": 0.1, "max_iter": 2000, "random_state": 42},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": _to_list(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_svc() -> None:
    X, y = clf_data(n=80, n_classes=2)
    model = SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42, probability=True)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    write_fixture(
        "svc",
        {
            "description": "SVC with RBF kernel, probability=True",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "random_state": 42, "probability": True},
            "expected": {
                "n_support": _to_list(model.n_support_),
                "predicted_classes": _to_list(preds),
                "predicted_proba": _to_list(proba),
            },
        },
    )


def gen_svr() -> None:
    X, y = reg_data(n=60)
    model = SVR(C=1.0, kernel="rbf", gamma="scale", epsilon=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "svr",
        {
            "description": "SVR with RBF kernel",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1},
            "expected": {
                "n_support": int(model.support_.shape[0]),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_nu_svc() -> None:
    X, y = clf_data(n=80, n_classes=2)
    model = NuSVC(nu=0.5, kernel="rbf", gamma="scale", random_state=42, probability=True)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    write_fixture(
        "nu_svc",
        {
            "description": "NuSVC with RBF kernel, nu=0.5",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"nu": 0.5, "kernel": "rbf", "gamma": "scale", "random_state": 42, "probability": True},
            "expected": {
                "n_support": _to_list(model.n_support_),
                "predicted_classes": _to_list(preds),
                "predicted_proba": _to_list(proba),
            },
        },
    )


def gen_nu_svr() -> None:
    X, y = reg_data(n=60)
    model = NuSVR(nu=0.5, C=1.0, kernel="rbf", gamma="scale")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "nu_svr",
        {
            "description": "NuSVR with RBF kernel, nu=0.5",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"nu": 0.5, "C": 1.0, "kernel": "rbf", "gamma": "scale"},
            "expected": {
                "n_support": int(model.support_.shape[0]),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_one_class_svm() -> None:
    X, _ = reg_data(n=80)
    # OneClassSVM is unsupervised — anomaly detection
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
    model.fit(X)
    preds = model.predict(X)  # +1 inliers, -1 outliers
    write_fixture(
        "one_class_svm",
        {
            "description": "OneClassSVM with RBF kernel, nu=0.1",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"nu": 0.1, "kernel": "rbf", "gamma": "scale"},
            "expected": {
                "n_support": int(model.support_.shape[0]),
                "predictions": _to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

def gen_sgd_classifier() -> None:
    X, y = clf_data(n=200, n_classes=2)
    model = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "sgd_classifier",
        {
            "description": "SGDClassifier log_loss + L2 penalty",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"loss": "log_loss", "alpha": 1e-4, "max_iter": 1000, "tol": 1e-3, "random_state": 42},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": _to_list(model.intercept_),
                "predicted_classes": _to_list(preds),
            },
        },
    )


def gen_sgd_regressor() -> None:
    X, y = reg_data(n=200)
    model = SGDRegressor(
        loss="squared_error",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "sgd_regressor",
        {
            "description": "SGDRegressor squared_error + L2 penalty",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"loss": "squared_error", "alpha": 1e-4, "max_iter": 1000, "tol": 1e-3, "random_state": 42},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_[0]) if hasattr(model.intercept_, "__len__") else float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# GLM family
# ---------------------------------------------------------------------------

def gen_poisson_regressor() -> None:
    X, y = reg_data(n=60, p=4)
    y = np.exp(y / np.abs(y).max() * 1.5).astype(int)  # positive integer counts
    model = PoissonRegressor(alpha=0.1, max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "poisson_regressor",
        {
            "description": "PoissonRegressor — GLM log-link, alpha=0.1",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"alpha": 0.1, "max_iter": 200},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_gamma_regressor() -> None:
    X, y = reg_data(n=60, p=4)
    y = np.abs(y) + 1.0  # positive
    model = GammaRegressor(alpha=0.1, max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "gamma_regressor",
        {
            "description": "GammaRegressor — GLM log-link, alpha=0.1",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"alpha": 0.1, "max_iter": 200},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


def gen_tweedie_regressor() -> None:
    X, y = reg_data(n=60, p=4)
    y = np.abs(y) + 1.0
    model = TweedieRegressor(power=1.5, alpha=0.1, max_iter=200, link="log")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "tweedie_regressor",
        {
            "description": "TweedieRegressor — power=1.5, log link, alpha=0.1",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"power": 1.5, "alpha": 0.1, "max_iter": 200, "link": "log"},
            "expected": {
                "coefficients": _to_list(model.coef_),
                "intercept": float(model.intercept_),
                "predictions": _to_list(preds),
            },
        },
    )


# ---------------------------------------------------------------------------
# Isotonic
# ---------------------------------------------------------------------------

def gen_isotonic_regression() -> None:
    rng = np.random.RandomState(42)
    x = np.sort(rng.uniform(0, 10, size=50))
    y = np.log1p(x) + rng.normal(scale=0.1, size=50)
    model = IsotonicRegression(increasing=True)
    model.fit(x, y)
    preds = model.predict(x)
    write_fixture(
        "isotonic_regression",
        {
            "description": "IsotonicRegression — monotonically increasing fit on log curve + noise",
            "random_state": 42,
            "input": {"x": _to_list(x), "y": _to_list(y)},
            "params": {"increasing": True},
            "expected": {
                "x_thresholds": _to_list(model.X_thresholds_),
                "y_thresholds": _to_list(model.y_thresholds_),
                "predictions": _to_list(preds),
            },
        },
    )


def main() -> None:
    print(f"Wave 1 — linear gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [
        gen_huber_regressor,
        gen_ransac,
        gen_quantile_regressor,
        gen_bayesian_ridge,
        gen_ard_regression,
        gen_lars,
        gen_lasso_lars,
        gen_omp,
        gen_ridge_cv,
        gen_lasso_cv,
        gen_elastic_net_cv,
        gen_logistic_regression_cv,
        gen_lda,
        gen_qda,
        gen_ridge_classifier,
        gen_linear_svc,
        gen_linear_svr,
        gen_svc,
        gen_svr,
        gen_nu_svc,
        gen_nu_svr,
        gen_one_class_svm,
        gen_sgd_classifier,
        gen_sgd_regressor,
        gen_poisson_regressor,
        gen_gamma_regressor,
        gen_tweedie_regressor,
        gen_isotonic_regression,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
