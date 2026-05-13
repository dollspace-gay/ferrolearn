#!/usr/bin/env python3
"""Wave 7 — sklearn fixtures for model-selection splitters and dummies."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
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
    (FIXTURES / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote fixtures/{name}.json")


def _splits_to_folds(splitter, X, y=None, groups=None):
    folds = []
    for train, test in splitter.split(X, y, groups):
        folds.append({"train": train.tolist(), "test": test.tolist()})
    return folds


def gen_leave_one_out():
    n = 20
    X = np.arange(n).reshape(-1, 1)
    folds = _splits_to_folds(LeaveOneOut(), X)
    write_fixture("leave_one_out", {
        "description": "LeaveOneOut on n=20 samples",
        "random_state": 42, "input": {"n_samples": n}, "params": {},
        "expected": {"folds": folds, "n_folds": len(folds)},
    })


def gen_leave_p_out():
    n = 12
    X = np.arange(n).reshape(-1, 1)
    folds = _splits_to_folds(LeavePOut(p=2), X)
    write_fixture("leave_p_out", {
        "description": "LeavePOut(p=2) on n=12 samples",
        "random_state": 42, "input": {"n_samples": n}, "params": {"p": 2},
        "expected": {"folds": folds, "n_folds": len(folds)},
    })


def gen_shuffle_split():
    n = 50
    X = np.arange(n).reshape(-1, 1)
    folds = _splits_to_folds(ShuffleSplit(n_splits=5, test_size=0.2, random_state=42), X)
    write_fixture("shuffle_split", {
        "description": "ShuffleSplit with 5 splits, test_size=0.2, seed=42",
        "random_state": 42, "input": {"n_samples": n},
        "params": {"n_splits": 5, "test_size": 0.2, "random_state": 42},
        "expected": {"folds": folds, "n_folds": 5},
    })


def gen_group_kfold():
    n = 60
    rng = np.random.RandomState(42)
    groups = rng.randint(0, 5, size=n)  # 5 groups
    X = np.arange(n).reshape(-1, 1)
    folds = _splits_to_folds(GroupKFold(n_splits=5), X, groups=groups)
    write_fixture("group_kfold", {
        "description": "GroupKFold n_splits=5",
        "random_state": 42,
        "input": {"n_samples": n, "groups": groups.tolist()},
        "params": {"n_splits": 5},
        "expected": {"folds": folds},
    })


def gen_group_shuffle_split():
    n = 60
    rng = np.random.RandomState(42)
    groups = rng.randint(0, 5, size=n)
    X = np.arange(n).reshape(-1, 1)
    folds = _splits_to_folds(GroupShuffleSplit(n_splits=3, test_size=0.3, random_state=42), X, groups=groups)
    write_fixture("group_shuffle_split", {
        "description": "GroupShuffleSplit with 3 splits, test_size=0.3, seed=42",
        "random_state": 42,
        "input": {"n_samples": n, "groups": groups.tolist()},
        "params": {"n_splits": 3, "test_size": 0.3, "random_state": 42},
        "expected": {"folds": folds},
    })


def gen_leave_one_group_out():
    n = 40
    rng = np.random.RandomState(42)
    groups = rng.randint(0, 4, size=n)  # 4 groups
    X = np.arange(n).reshape(-1, 1)
    folds = _splits_to_folds(LeaveOneGroupOut(), X, groups=groups)
    write_fixture("leave_one_group_out", {
        "description": "LeaveOneGroupOut with 4 groups",
        "random_state": 42,
        "input": {"n_samples": n, "groups": groups.tolist()},
        "params": {},
        "expected": {"folds": folds, "n_folds": 4},
    })


def gen_dummy_classifier():
    X, y = make_classification(n_samples=60, n_features=4, n_classes=3, n_informative=3,
                                n_redundant=0, random_state=42)
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("dummy_classifier", {
        "description": "DummyClassifier most_frequent on 3-class",
        "random_state": 42,
        "input": {"X": X.tolist(), "y": y.tolist()},
        "params": {"strategy": "most_frequent"},
        "expected": {
            "predictions": preds.tolist(),
            "classes": model.classes_.tolist(),
            "class_prior": model.class_prior_.tolist(),
        },
    })


def gen_dummy_regressor():
    X, y = make_regression(n_samples=60, n_features=4, noise=0.1, random_state=42)
    model = DummyRegressor(strategy="mean")
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("dummy_regressor", {
        "description": "DummyRegressor mean strategy",
        "random_state": 42,
        "input": {"X": X.tolist(), "y": y.tolist()},
        "params": {"strategy": "mean"},
        "expected": {
            "constant": float(model.constant_[0]),
            "predictions": preds.tolist(),
        },
    })


def main():
    print(f"Wave 7 — model-sel gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [
        gen_leave_one_out, gen_leave_p_out, gen_shuffle_split,
        gen_group_kfold, gen_group_shuffle_split, gen_leave_one_group_out,
        gen_dummy_classifier, gen_dummy_regressor,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
