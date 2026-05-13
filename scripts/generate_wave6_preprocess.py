#!/usr/bin/env python3
"""Wave 6 — preprocess gap fixtures vs sklearn."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    MultiLabelBinarizer,
    OrdinalEncoder,
    SplineTransformer,
)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

SKLEARN_VERSION = sklearn.__version__
SKLEARN_PIN = f"scikit-learn=={SKLEARN_VERSION}"
ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, "toarray"):
        return x.toarray().tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def write_fixture(name, payload):
    payload.setdefault("sklearn_version", SKLEARN_VERSION)
    payload.setdefault("sklearn_pin", SKLEARN_PIN)
    (FIXTURES / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote fixtures/{name}.json")


def gen_ordinal_encoder():
    rng = np.random.RandomState(42)
    X = rng.choice(["a", "b", "c", "d"], size=(40, 3))
    model = OrdinalEncoder()
    Xt = model.fit_transform(X)
    write_fixture("ordinal_encoder", {
        "description": "OrdinalEncoder on categorical strings",
        "random_state": 42,
        "input": {"X": X.tolist()},
        "params": {},
        "expected": {"transformed": _to_list(Xt), "categories": [c.tolist() for c in model.categories_]},
    })


def gen_label_binarizer():
    y = np.array([0, 1, 2, 0, 1, 2, 0])
    model = LabelBinarizer()
    Yt = model.fit_transform(y)
    write_fixture("label_binarizer", {
        "description": "LabelBinarizer on 3-class labels",
        "random_state": 42, "input": {"y": y.tolist()},
        "params": {},
        "expected": {"transformed": _to_list(Yt), "classes": _to_list(model.classes_)},
    })


def gen_multilabel_binarizer():
    yy = [[0, 1], [1, 2], [0], [2], [0, 1, 2]]
    model = MultiLabelBinarizer()
    Yt = model.fit_transform(yy)
    write_fixture("multilabel_binarizer", {
        "description": "MultiLabelBinarizer on label-set lists",
        "random_state": 42, "input": {"y": yy},
        "params": {},
        "expected": {"transformed": _to_list(Yt), "classes": _to_list(model.classes_)},
    })


def gen_variance_threshold():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 6)
    X[:, 1] = X[0, 1]  # constant column → variance 0
    model = VarianceThreshold(threshold=0.0)
    Xt = model.fit_transform(X)
    write_fixture("variance_threshold", {
        "description": "VarianceThreshold removes constant column",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"threshold": 0.0},
        "expected": {
            "transformed": _to_list(Xt),
            "support": _to_list(model.get_support().astype(int)),
            "variances": _to_list(model.variances_),
        },
    })


def gen_select_k_best():
    X, y = make_classification(n_samples=80, n_features=10, n_informative=4, n_redundant=0,
                                random_state=42)
    model = SelectKBest(score_func=f_classif, k=4)
    Xt = model.fit_transform(X, y)
    write_fixture("select_k_best", {
        "description": "SelectKBest with f_classif, k=4",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"score_func": "f_classif", "k": 4},
        "expected": {
            "transformed": _to_list(Xt),
            "support": _to_list(model.get_support().astype(int)),
            "scores": _to_list(model.scores_),
        },
    })


def gen_select_percentile():
    X, y = make_regression(n_samples=80, n_features=10, noise=0.1, random_state=42)
    model = SelectPercentile(score_func=f_regression, percentile=40)
    Xt = model.fit_transform(X, y)
    write_fixture("select_percentile", {
        "description": "SelectPercentile with f_regression, percentile=40",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"score_func": "f_regression", "percentile": 40},
        "expected": {
            "transformed": _to_list(Xt),
            "support": _to_list(model.get_support().astype(int)),
            "scores": _to_list(model.scores_),
        },
    })


def gen_select_from_model():
    X, y = make_classification(n_samples=80, n_features=10, n_informative=4, n_redundant=0,
                                random_state=42)
    base = LogisticRegression(C=1.0, max_iter=500, random_state=42, penalty="l1", solver="liblinear")
    model = SelectFromModel(estimator=base)
    Xt = model.fit_transform(X, y)
    write_fixture("select_from_model", {
        "description": "SelectFromModel with L1-regularized LogisticRegression",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"C": 1.0, "penalty": "l1", "max_iter": 500, "random_state": 42},
        "expected": {
            "support": _to_list(model.get_support().astype(int)),
            "n_selected": int(model.get_support().sum()),
        },
    })


def gen_rfe():
    X, y = make_classification(n_samples=80, n_features=10, n_informative=4, n_redundant=0,
                                random_state=42)
    base = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model = RFE(estimator=base, n_features_to_select=4)
    model.fit(X, y)
    write_fixture("rfe", {
        "description": "RFE with LogisticRegression, n_features_to_select=4",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"n_features_to_select": 4},
        "expected": {
            "support": _to_list(model.support_.astype(int)),
            "ranking": _to_list(model.ranking_),
        },
    })


def gen_knn_imputer():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 5)
    # Inject some NaNs.
    X[5, 0] = np.nan
    X[12, 2] = np.nan
    X[20, 4] = np.nan
    model = KNNImputer(n_neighbors=5)
    Xt = model.fit_transform(X)
    write_fixture("knn_imputer", {
        "description": "KNNImputer with k=5",
        "random_state": 42,
        "input": {"X": [[v if not np.isnan(v) else "NaN" for v in row] for row in X.tolist()]},
        "params": {"n_neighbors": 5},
        "expected": {"transformed": _to_list(Xt)},
    })


def gen_spline_transformer():
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 10, size=(40, 2))
    model = SplineTransformer(n_knots=5, degree=3, include_bias=False)
    Xt = model.fit_transform(X)
    write_fixture("spline_transformer", {
        "description": "SplineTransformer cubic with 5 knots",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_knots": 5, "degree": 3, "include_bias": False},
        "expected": {"transformed": _to_list(Xt)},
    })


def gen_gaussian_random_projection():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 20)
    model = GaussianRandomProjection(n_components=8, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture("gaussian_random_projection", {
        "description": "GaussianRandomProjection to 8 components",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_components": 8, "random_state": 42},
        "expected": {"transformed_shape": list(Xt.shape)},
    })


def gen_sparse_random_projection():
    rng = np.random.RandomState(42)
    X = rng.randn(40, 30)
    model = SparseRandomProjection(n_components=8, random_state=42)
    Xt = model.fit_transform(X)
    write_fixture("sparse_random_projection", {
        "description": "SparseRandomProjection to 8 components",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_components": 8, "random_state": 42},
        "expected": {"transformed_shape": list(Xt.shape)},
    })


def gen_function_transformer():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # log1p deterministic transform
    model = FunctionTransformer(func=np.log1p)
    Xt = model.fit_transform(X)
    write_fixture("function_transformer", {
        "description": "FunctionTransformer with log1p",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"func": "log1p"},
        "expected": {"transformed": _to_list(Xt)},
    })


def main():
    print(f"Wave 6 — preprocess gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [
        gen_ordinal_encoder,
        gen_label_binarizer,
        gen_multilabel_binarizer,
        gen_variance_threshold,
        gen_select_k_best,
        gen_select_percentile,
        gen_select_from_model,
        gen_rfe,
        gen_knn_imputer,
        gen_spline_transformer,
        gen_gaussian_random_projection,
        gen_sparse_random_projection,
        gen_function_transformer,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
