#!/usr/bin/env python3
"""Wave 3 — sklearn reference fixtures for previously-untested tree estimators.

Adds: ExtraTreeClassifier, ExtraTreeRegressor (single trees),
ExtraTreesClassifier, ExtraTreesRegressor (ensembles),
BaggingClassifier/Regressor, AdaBoostRegressor,
HistGradientBoostingClassifier/Regressor, IsolationForest,
RandomTreesEmbedding, VotingClassifier/Regressor.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomTreesEmbedding,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)

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
    out = FIXTURES / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}")


def clf_data(n=120, p=5, n_classes=2):
    return make_classification(
        n_samples=n, n_features=p, n_informative=4, n_redundant=0,
        n_classes=n_classes, random_state=42,
    )


def reg_data(n=80, p=5):
    return make_regression(n_samples=n, n_features=p, noise=0.1, random_state=42)


def gen_extra_tree_classifier():
    X, y = clf_data()
    model = ExtraTreeClassifier(max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("extra_tree_classifier", {
        "description": "ExtraTreeClassifier (single random tree)",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"max_depth": 5, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "accuracy": float(np.mean(preds == y))},
    })


def gen_extra_tree_regressor():
    X, y = reg_data()
    model = ExtraTreeRegressor(max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    ss_res = ((y - preds)**2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    write_fixture("extra_tree_regressor", {
        "description": "ExtraTreeRegressor (single random tree)",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"max_depth": 5, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "r2": r2},
    })


def gen_extra_trees_classifier():
    X, y = clf_data()
    model = ExtraTreesClassifier(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("extra_trees_classifier", {
        "description": "ExtraTreesClassifier ensemble",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"n_estimators": 20, "max_depth": 5, "random_state": 42},
        "expected": {
            "predictions": _to_list(preds),
            "accuracy": float(np.mean(preds == y)),
            "feature_importances": _to_list(model.feature_importances_),
        },
    })


def gen_extra_trees_regressor():
    X, y = reg_data()
    model = ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("extra_trees_regressor", {
        "description": "ExtraTreesRegressor ensemble",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"n_estimators": 20, "max_depth": 5, "random_state": 42},
        "expected": {
            "predictions": _to_list(preds),
            "r2": r2,
            "feature_importances": _to_list(model.feature_importances_),
        },
    })


def gen_bagging_classifier():
    X, y = clf_data()
    base = DecisionTreeClassifier(max_depth=3, random_state=42)
    model = BaggingClassifier(estimator=base, n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("bagging_classifier", {
        "description": "BaggingClassifier over DecisionTreeClassifier",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"n_estimators": 10, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "accuracy": float(np.mean(preds == y))},
    })


def gen_bagging_regressor():
    X, y = reg_data()
    base = DecisionTreeRegressor(max_depth=3, random_state=42)
    model = BaggingRegressor(estimator=base, n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("bagging_regressor", {
        "description": "BaggingRegressor over DecisionTreeRegressor",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"n_estimators": 10, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "r2": r2},
    })


def gen_adaboost_regressor():
    X, y = reg_data()
    model = AdaBoostRegressor(n_estimators=20, learning_rate=1.0, loss="linear", random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("adaboost_regressor", {
        "description": "AdaBoostRegressor over DecisionTreeRegressor",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"n_estimators": 20, "learning_rate": 1.0, "loss": "linear", "random_state": 42},
        "expected": {"predictions": _to_list(preds), "r2": r2},
    })


def gen_hist_gbc():
    X, y = clf_data(n=200, n_classes=2)
    model = HistGradientBoostingClassifier(max_iter=50, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("hist_gradient_boosting_classifier", {
        "description": "HistGradientBoostingClassifier",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"max_iter": 50, "max_depth": 5, "learning_rate": 0.1, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "accuracy": float(np.mean(preds == y))},
    })


def gen_hist_gbr():
    X, y = reg_data(n=200)
    model = HistGradientBoostingRegressor(max_iter=50, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("hist_gradient_boosting_regressor", {
        "description": "HistGradientBoostingRegressor",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"max_iter": 50, "max_depth": 5, "learning_rate": 0.1, "random_state": 42},
        "expected": {"predictions": _to_list(preds), "r2": r2},
    })


def gen_isolation_forest():
    rng = np.random.RandomState(42)
    n_normal = 100
    n_outlier = 10
    X = np.vstack([rng.randn(n_normal, 4), rng.randn(n_outlier, 4) * 5 + 10])
    model = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    model.fit(X)
    preds = model.predict(X)  # +1 normal, -1 outlier
    scores = model.score_samples(X)
    write_fixture("isolation_forest", {
        "description": "IsolationForest with 10% contamination",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_estimators": 50, "contamination": 0.1, "random_state": 42},
        "expected": {
            "predictions": _to_list(preds),
            "score_samples": _to_list(scores),
        },
    })


def gen_random_trees_embedding():
    rng = np.random.RandomState(42)
    X = rng.randn(80, 5)
    model = RandomTreesEmbedding(n_estimators=20, max_depth=4, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture("random_trees_embedding", {
        "description": "RandomTreesEmbedding",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_estimators": 20, "max_depth": 4, "random_state": 42},
        "expected": {
            "transformed_shape": list(Xt.shape if hasattr(Xt, "shape") else Xt.toarray().shape),
        },
    })


def gen_voting_classifier():
    X, y = clf_data(n=100)
    clf1 = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    clf2 = DecisionTreeClassifier(max_depth=5, random_state=42)
    model = VotingClassifier(estimators=[("lr", clf1), ("dt", clf2)], voting="soft")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("voting_classifier", {
        "description": "VotingClassifier (LR + DT soft voting)",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"voting": "soft"},
        "expected": {"predictions": _to_list(preds), "accuracy": float(np.mean(preds == y))},
    })


def gen_voting_regressor():
    X, y = reg_data()
    r1 = LinearRegression()
    r2 = DecisionTreeRegressor(max_depth=5, random_state=42)
    model = VotingRegressor(estimators=[("lr", r1), ("dt", r2)])
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2_score = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("voting_regressor", {
        "description": "VotingRegressor (LR + DT)",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {},
        "expected": {"predictions": _to_list(preds), "r2": r2_score},
    })


def main():
    print(f"Wave 3 — tree gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [
        gen_extra_tree_classifier,
        gen_extra_tree_regressor,
        gen_extra_trees_classifier,
        gen_extra_trees_regressor,
        gen_bagging_classifier,
        gen_bagging_regressor,
        gen_adaboost_regressor,
        gen_hist_gbc,
        gen_hist_gbr,
        gen_isolation_forest,
        gen_random_trees_embedding,
        gen_voting_classifier,
        gen_voting_regressor,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
