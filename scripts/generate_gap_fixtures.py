#!/usr/bin/env python3
"""Gap-filler fixtures for ferrolearn conformance suite.

Generates fixtures for estimators that previously had no oracle coverage:
covariance (5), kernel-ridge (1), MLP (1). Output schema is fixture v2
(matches `fixtures/README.md`).

Usage:
    python3 scripts/generate_gap_fixtures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import sklearn
from sklearn.covariance import (
    EmpiricalCovariance,
    LedoitWolf,
    MinCovDet,
    OAS,
    ShrunkCovariance,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPClassifier

SKLEARN_VERSION = sklearn.__version__
SKLEARN_PIN = f"scikit-learn=={SKLEARN_VERSION}"

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
FIXTURES.mkdir(exist_ok=True)


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def write_fixture(name: str, payload: dict) -> None:
    """Write a fixture with v2 metadata stamps."""
    payload.setdefault("sklearn_version", SKLEARN_VERSION)
    payload.setdefault("sklearn_pin", SKLEARN_PIN)
    out = FIXTURES / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Covariance fixtures — closed-form math, tolerances tight.
# ---------------------------------------------------------------------------

def make_cov_data(seed: int = 42, n_samples: int = 200, n_features: int = 5):
    """Standard regression-style X used across all covariance fixtures."""
    rng = np.random.RandomState(seed)
    # Slightly correlated features so shrinkage estimators have something to
    # actually shrink towards.
    base = rng.randn(n_samples, n_features)
    mixing = np.eye(n_features) + 0.3 * rng.randn(n_features, n_features)
    return base @ mixing


def gen_empirical_covariance() -> None:
    X = make_cov_data()
    model = EmpiricalCovariance(assume_centered=False)
    model.fit(X)
    write_fixture(
        "empirical_covariance",
        {
            "description": "EmpiricalCovariance — closed-form sample covariance",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"assume_centered": False},
            "expected": {
                "location": _to_list(model.location_),
                "covariance": _to_list(model.covariance_),
                "precision": _to_list(model.precision_),
            },
            "tolerance": {"rel": 1e-10, "abs": 1e-13},
        },
    )


def gen_shrunk_covariance() -> None:
    X = make_cov_data()
    shrinkage = 0.1
    model = ShrunkCovariance(shrinkage=shrinkage, assume_centered=False)
    model.fit(X)
    write_fixture(
        "shrunk_covariance",
        {
            "description": "ShrunkCovariance — sample cov with explicit shrinkage",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"shrinkage": shrinkage, "assume_centered": False},
            "expected": {
                "location": _to_list(model.location_),
                "covariance": _to_list(model.covariance_),
                "precision": _to_list(model.precision_),
            },
            "tolerance": {"rel": 1e-10, "abs": 1e-13},
        },
    )


def gen_ledoit_wolf() -> None:
    X = make_cov_data()
    model = LedoitWolf(assume_centered=False)
    model.fit(X)
    write_fixture(
        "ledoit_wolf",
        {
            "description": "LedoitWolf — automatic shrinkage selection",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"assume_centered": False},
            "expected": {
                "location": _to_list(model.location_),
                "covariance": _to_list(model.covariance_),
                "precision": _to_list(model.precision_),
                "shrinkage": float(model.shrinkage_),
            },
            "tolerance": {"rel": 1e-6, "abs": 1e-9},
        },
    )


def gen_oas() -> None:
    X = make_cov_data()
    model = OAS(assume_centered=False)
    model.fit(X)
    write_fixture(
        "oas",
        {
            "description": "OAS — oracle-approximating shrinkage estimator",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"assume_centered": False},
            "expected": {
                "location": _to_list(model.location_),
                "covariance": _to_list(model.covariance_),
                "precision": _to_list(model.precision_),
                "shrinkage": float(model.shrinkage_),
            },
            # OAS Chen-2010 vs sklearn simplified — see _divergences.toml
            # (oas-chen-2010-vs-sklearn-simplified). Conformance test sets
            # its own widened tolerance.
        },
    )


def gen_min_cov_det() -> None:
    X = make_cov_data(seed=7, n_samples=120, n_features=4)
    model = MinCovDet(random_state=42, support_fraction=0.9)
    model.fit(X)
    write_fixture(
        "min_cov_det",
        {
            "description": "MinCovDet — robust covariance via FastMCD",
            "random_state": 42,
            "input": {"X": _to_list(X)},
            "params": {"support_fraction": 0.9, "random_state": 42},
            "expected": {
                "location": _to_list(model.location_),
                "covariance": _to_list(model.covariance_),
                "precision": _to_list(model.precision_),
                "support": _to_list(model.support_.astype(int)),
            },
            # FastMCD random subset variance — see _divergences.toml
            # (fastmcd-subset-selection-variance). Conformance test sets
            # its own widened tolerance.
        },
    )


# ---------------------------------------------------------------------------
# Kernel-Ridge fixture.
# ---------------------------------------------------------------------------

def gen_kernel_ridge() -> None:
    X, y = make_regression(
        n_samples=80, n_features=5, noise=0.1, random_state=42
    )
    model = KernelRidge(alpha=1.0, kernel="linear")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture(
        "kernel_ridge",
        {
            "description": "KernelRidge with linear kernel (sklearn default after 1.4)",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {"alpha": 1.0, "kernel": "linear"},
            "expected": {
                "dual_coef": _to_list(model.dual_coef_),
                "predictions": _to_list(preds),
            },
            "tolerance": {"rel": 1e-9, "abs": 1e-12},
        },
    )


# ---------------------------------------------------------------------------
# MLPClassifier fixture — seed-sensitive, looser tolerance.
# ---------------------------------------------------------------------------

def gen_mlp_classifier() -> None:
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    model = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="lbfgs",
        max_iter=500,
        random_state=42,
        alpha=1e-4,
    )
    model.fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    acc = float(np.mean(preds == y))
    write_fixture(
        "mlp_classifier",
        {
            "description": "MLPClassifier — single hidden layer, LBFGS, 3-class",
            "random_state": 42,
            "input": {"X": _to_list(X), "y": _to_list(y)},
            "params": {
                "hidden_layer_sizes": [16],
                "activation": "relu",
                "solver": "lbfgs",
                "alpha": 1e-4,
                "max_iter": 500,
                "random_state": 42,
            },
            "expected": {
                "predictions": _to_list(preds),
                "predicted_proba": _to_list(proba),
                "accuracy": acc,
                "n_layers": int(model.n_layers_),
                "n_outputs": int(model.n_outputs_),
            },
            "tolerance": {"rel": 1e-2, "abs": 1e-3},
        },
    )


def main() -> None:
    print(f"Generating gap fixtures with sklearn {SKLEARN_VERSION}")
    print(f"Writing to {FIXTURES}")
    gen_empirical_covariance()
    gen_shrunk_covariance()
    gen_ledoit_wolf()
    gen_oas()
    gen_min_cov_det()
    gen_kernel_ridge()
    gen_mlp_classifier()
    print("Done.")


if __name__ == "__main__":
    main()
