#!/usr/bin/env python3
"""Wave 4 — sklearn fixtures for previously-untested estimators across
cluster, neighbors, bayes, neural, and covariance crates.

Cluster: AffinityPropagation, BayesianGaussianMixture, BisectingKMeans,
         FeatureAgglomeration, HDBSCAN, LabelPropagation, LabelSpreading
Neighbors: LocalOutlierFactor, NearestCentroid, NearestNeighbors,
           RadiusNeighborsClassifier, RadiusNeighborsRegressor
Bayes: CategoricalNB
Neural: MLPRegressor (with adam to match what ferrolearn supports), BernoulliRBM
Covariance: GraphicalLasso, EllipticEnvelope
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.cluster import (
    AffinityPropagation,
    BisectingKMeans,
    FeatureAgglomeration,
    HDBSCAN,
)
from sklearn.covariance import EllipticEnvelope, GraphicalLasso
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import (
    LocalOutlierFactor,
    NearestCentroid,
    NearestNeighbors,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.neural_network import BernoulliRBM, MLPRegressor
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

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


# ---------------------------- cluster ----------------------------

def gen_affinity_propagation():
    X, _ = make_blobs(n_samples=80, centers=3, random_state=42, n_features=4, cluster_std=1.5)
    model = AffinityPropagation(random_state=42, max_iter=200, damping=0.7, preference=-50.0)
    model.fit(X)
    write_fixture("affinity_propagation", {
        "description": "AffinityPropagation on 3-blob dataset",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"random_state": 42, "max_iter": 200, "damping": 0.7, "preference": -50.0},
        "expected": {
            "labels": _to_list(model.labels_),
            "n_clusters": int(len(model.cluster_centers_indices_)),
            "cluster_centers": _to_list(model.cluster_centers_),
        },
    })


def gen_bayesian_gaussian_mixture():
    X, _ = make_blobs(n_samples=80, centers=3, random_state=42, n_features=4)
    model = BayesianGaussianMixture(n_components=3, random_state=42, max_iter=200, n_init=3, init_params="kmeans")
    model.fit(X)
    write_fixture("bayesian_gaussian_mixture", {
        "description": "BayesianGaussianMixture (variational EM)",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_components": 3, "random_state": 42, "max_iter": 200, "n_init": 3},
        "expected": {
            "labels": _to_list(model.predict(X)),
            "means": _to_list(model.means_),
            "weights": _to_list(model.weights_),
        },
    })


def gen_bisecting_kmeans():
    X, _ = make_blobs(n_samples=80, centers=3, random_state=42, n_features=4)
    model = BisectingKMeans(n_clusters=3, random_state=42, n_init=3)
    model.fit(X)
    write_fixture("bisecting_kmeans", {
        "description": "BisectingKMeans with k=3",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_clusters": 3, "random_state": 42, "n_init": 3},
        "expected": {
            "labels": _to_list(model.labels_),
            "cluster_centers": _to_list(model.cluster_centers_),
            "inertia": float(model.inertia_),
        },
    })


def gen_feature_agglomeration():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 10)
    model = FeatureAgglomeration(n_clusters=3)
    Xt = model.fit_transform(X)
    write_fixture("feature_agglomeration", {
        "description": "FeatureAgglomeration with 3 feature clusters",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_clusters": 3},
        "expected": {
            "labels": _to_list(model.labels_),
            "transformed": _to_list(Xt),
        },
    })


def gen_hdbscan():
    X, _ = make_blobs(n_samples=120, centers=3, random_state=42, n_features=4, cluster_std=1.2)
    model = HDBSCAN(min_cluster_size=10)
    model.fit(X)
    write_fixture("hdbscan", {
        "description": "HDBSCAN with min_cluster_size=10",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"min_cluster_size": 10},
        "expected": {
            "labels": _to_list(model.labels_),
            "n_clusters": int(len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)),
        },
    })


def gen_label_propagation():
    rng = np.random.RandomState(42)
    X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0,
                                n_classes=3, random_state=42)
    # Mark 60% of labels as unlabeled (-1).
    mask = rng.rand(len(y)) < 0.6
    y_partial = y.copy()
    y_partial[mask] = -1
    model = LabelPropagation(kernel="rbf", gamma=1.0)
    model.fit(X, y_partial)
    preds = model.predict(X)
    write_fixture("label_propagation", {
        "description": "LabelPropagation with 60% unlabeled samples",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y_partial.astype(int))},
        "params": {"kernel": "rbf", "gamma": 1.0},
        "expected": {
            "transduction": _to_list(model.transduction_),
            "predictions": _to_list(preds),
        },
    })


def gen_label_spreading():
    rng = np.random.RandomState(42)
    X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0,
                                n_classes=3, random_state=42)
    mask = rng.rand(len(y)) < 0.6
    y_partial = y.copy()
    y_partial[mask] = -1
    model = LabelSpreading(kernel="rbf", gamma=1.0, alpha=0.2)
    model.fit(X, y_partial)
    preds = model.predict(X)
    write_fixture("label_spreading", {
        "description": "LabelSpreading with alpha=0.2",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y_partial.astype(int))},
        "params": {"kernel": "rbf", "gamma": 1.0, "alpha": 0.2},
        "expected": {
            "transduction": _to_list(model.transduction_),
            "predictions": _to_list(preds),
        },
    })


# ---------------------------- neighbors ----------------------------

def gen_local_outlier_factor():
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(80, 4), rng.randn(10, 4) * 5 + 8])
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    preds = model.fit_predict(X)
    scores = -model.negative_outlier_factor_
    write_fixture("local_outlier_factor", {
        "description": "LOF with n_neighbors=20, contamination=0.1",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_neighbors": 20, "contamination": 0.1},
        "expected": {"predictions": _to_list(preds), "scores": _to_list(scores)},
    })


def gen_nearest_centroid():
    X, y = make_classification(n_samples=100, n_features=4, n_informative=3, n_redundant=0,
                                n_classes=3, random_state=42)
    model = NearestCentroid()
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("nearest_centroid", {
        "description": "NearestCentroid",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {},
        "expected": {
            "centroids": _to_list(model.centroids_),
            "classes": _to_list(model.classes_),
            "predictions": _to_list(preds),
            "accuracy": float(np.mean(preds == y)),
        },
    })


def gen_nearest_neighbors():
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    model = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean")
    model.fit(X)
    dists, idxs = model.kneighbors(X[:10])
    write_fixture("nearest_neighbors", {
        "description": "NearestNeighbors kneighbors query for first 10 rows",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_neighbors": 5, "algorithm": "auto"},
        "expected": {
            "distances": _to_list(dists),
            "indices": _to_list(idxs),
        },
    })


def gen_radius_neighbors_classifier():
    X, y = make_classification(n_samples=60, n_features=4, n_informative=3, n_redundant=0,
                                n_classes=3, random_state=42)
    # Pick a radius large enough that each query has at least 1 neighbor.
    model = RadiusNeighborsClassifier(radius=4.0)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("radius_neighbors_classifier", {
        "description": "RadiusNeighborsClassifier with radius=4.0",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"radius": 4.0},
        "expected": {"predictions": _to_list(preds), "accuracy": float(np.mean(preds == y))},
    })


def gen_radius_neighbors_regressor():
    X, y = make_regression(n_samples=60, n_features=4, noise=0.1, random_state=42)
    model = RadiusNeighborsRegressor(radius=4.0)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("radius_neighbors_regressor", {
        "description": "RadiusNeighborsRegressor with radius=4.0",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"radius": 4.0},
        "expected": {"predictions": _to_list(preds), "r2": r2},
    })


# ---------------------------- bayes ----------------------------

def gen_categorical_nb():
    rng = np.random.RandomState(42)
    X = (rng.rand(80, 5) * 4).astype(int)  # categorical features 0..3
    y = rng.randint(0, 3, size=80)
    model = CategoricalNB(alpha=1.0)
    model.fit(X, y)
    preds = model.predict(X)
    write_fixture("categorical_nb", {
        "description": "CategoricalNB on 4-level discrete features",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"alpha": 1.0},
        "expected": {"predictions": _to_list(preds), "accuracy": float(np.mean(preds == y))},
    })


# ---------------------------- neural ----------------------------

def gen_mlp_regressor():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    # Use adam — ferrolearn supports this; lbfgs is not implemented.
    model = MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                         max_iter=500, random_state=42, alpha=1e-4)
    model.fit(X, y)
    preds = model.predict(X)
    ss_tot = ((y - y.mean())**2).sum()
    r2 = 1.0 - ((y - preds)**2).sum() / ss_tot if ss_tot > 0 else 0.0
    write_fixture("mlp_regressor", {
        "description": "MLPRegressor — single hidden layer, Adam",
        "random_state": 42, "input": {"X": _to_list(X), "y": _to_list(y)},
        "params": {"hidden_layer_sizes": [16], "activation": "relu", "solver": "adam",
                    "max_iter": 500, "random_state": 42, "alpha": 1e-4},
        "expected": {"predictions": _to_list(preds), "r2": r2, "n_layers": int(model.n_layers_)},
    })


def gen_bernoulli_rbm():
    rng = np.random.RandomState(42)
    X = (rng.rand(80, 6) > 0.5).astype(float)
    model = BernoulliRBM(n_components=4, learning_rate=0.1, n_iter=10, random_state=42, batch_size=10)
    Xt = model.fit_transform(X)
    write_fixture("bernoulli_rbm", {
        "description": "BernoulliRBM with 4 hidden units",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"n_components": 4, "learning_rate": 0.1, "n_iter": 10, "random_state": 42, "batch_size": 10},
        "expected": {"transformed": _to_list(Xt), "components_shape": list(model.components_.shape)},
    })


# ---------------------------- covariance ----------------------------

def gen_graphical_lasso():
    rng = np.random.RandomState(42)
    X = rng.randn(120, 5)
    model = GraphicalLasso(alpha=0.1, max_iter=200)
    model.fit(X)
    write_fixture("graphical_lasso", {
        "description": "GraphicalLasso with alpha=0.1",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"alpha": 0.1, "max_iter": 200},
        "expected": {
            "location": _to_list(model.location_),
            "covariance": _to_list(model.covariance_),
            "precision": _to_list(model.precision_),
        },
    })


def gen_elliptic_envelope():
    rng = np.random.RandomState(42)
    X = np.vstack([rng.randn(100, 4), rng.randn(10, 4) * 4 + 5])
    model = EllipticEnvelope(contamination=0.1, random_state=42, support_fraction=0.95)
    model.fit(X)
    preds = model.predict(X)
    write_fixture("elliptic_envelope", {
        "description": "EllipticEnvelope with 10% contamination",
        "random_state": 42, "input": {"X": _to_list(X)},
        "params": {"contamination": 0.1, "random_state": 42, "support_fraction": 0.95},
        "expected": {"predictions": _to_list(preds)},
    })


def main():
    print(f"Wave 4 — gap fixtures with sklearn {SKLEARN_VERSION}")
    for fn in [
        gen_affinity_propagation,
        gen_bayesian_gaussian_mixture,
        gen_bisecting_kmeans,
        gen_feature_agglomeration,
        gen_hdbscan,
        gen_label_propagation,
        gen_label_spreading,
        gen_local_outlier_factor,
        gen_nearest_centroid,
        gen_nearest_neighbors,
        gen_radius_neighbors_classifier,
        gen_radius_neighbors_regressor,
        gen_categorical_nb,
        gen_mlp_regressor,
        gen_bernoulli_rbm,
        gen_graphical_lasso,
        gen_elliptic_envelope,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ERROR in {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
