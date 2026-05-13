#!/usr/bin/env python3
"""Wave 5 — sklearn kernel-method fixtures: GP regressor/classifier, Nystroem, RBFSampler."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_approximation import Nystroem, RBFSampler

SKLEARN_VERSION = sklearn.__version__
SKLEARN_PIN = f"scikit-learn=={SKLEARN_VERSION}"
ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def write_fixture(name, payload):
    payload.setdefault("sklearn_version", SKLEARN_VERSION)
    payload.setdefault("sklearn_pin", SKLEARN_PIN)
    (FIXTURES / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote fixtures/{name}.json")


def gen_gp_regressor():
    X, y = make_regression(n_samples=40, n_features=4, noise=0.1, random_state=42)
    model = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), random_state=42, alpha=1e-6)
    model.fit(X, y)
    preds, std = model.predict(X, return_std=True)
    write_fixture("gaussian_process_regressor", {
        "description": "GaussianProcessRegressor with RBF kernel",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"length_scale": 1.0, "alpha": 1e-6, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "std": _to_list(std)},
    })


def gen_gp_classifier():
    X, y = make_classification(n_samples=40, n_features=4, n_informative=3, n_redundant=0,
                                n_classes=2, random_state=42)
    model = GaussianProcessClassifier(kernel=RBF(length_scale=1.0), random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    write_fixture("gaussian_process_classifier", {
        "description": "GaussianProcessClassifier with RBF kernel (binary)",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"length_scale": 1.0, "random_state": 42},
        "expected": {
            "predictions": _to_list(preds),
            "predicted_proba": _to_list(proba),
            "accuracy": float(np.mean(preds == y)),
        },
    })


def gen_nystroem():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    model = Nystroem(kernel="rbf", gamma=0.1, n_components=10, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture("nystroem", {
        "description": "Nystroem with RBF kernel, n_components=10",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"kernel": "rbf", "gamma": 0.1, "n_components": 10, "random_state": 42},
        "expected": {"transformed": _to_list(Xt)},
    })


def gen_rbf_sampler():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    model = RBFSampler(gamma=0.5, n_components=20, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture("rbf_sampler", {
        "description": "RBFSampler (random Fourier features)",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"gamma": 0.5, "n_components": 20, "random_state": 42},
        "expected": {"transformed": _to_list(Xt)},
    })


def gen_kernel_ridge_rbf():
    X, y = make_regression(n_samples=60, n_features=5, noise=0.1, random_state=42)
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("kernel_ridge_rbf", {
        "description": "KernelRidge with RBF kernel",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"alpha": 1.0, "kernel": "rbf", "gamma": 0.1},
        "expected": {"predictions": _to_list(preds)},
    })


def gen_kernel_ridge_poly():
    X, y = make_regression(n_samples=60, n_features=5, noise=0.1, random_state=42)
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge(alpha=1.0, kernel="poly", degree=2, gamma=0.1, coef0=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("kernel_ridge_poly", {
        "description": "KernelRidge with polynomial kernel deg=2",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"alpha": 1.0, "kernel": "poly", "degree": 2, "gamma": 0.1, "coef0": 1.0},
        "expected": {"predictions": _to_list(preds)},
    })


def main():
    print(f"Wave 5 — kernel gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [gen_gp_regressor, gen_gp_classifier, gen_nystroem, gen_rbf_sampler,
              gen_kernel_ridge_rbf, gen_kernel_ridge_poly]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
