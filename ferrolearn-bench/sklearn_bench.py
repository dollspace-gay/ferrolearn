#!/usr/bin/env python3
"""Benchmark scikit-learn algorithms for comparison with ferrolearn.

Prints results as a table. Each benchmark runs 20 iterations and reports
the median wall-clock time.
"""

import time
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIZES = [
    ("tiny_50x5", 50, 5),
    ("small_1Kx10", 1_000, 10),
    ("medium_10Kx100", 10_000, 100),
]

METRIC_SIZES = [
    ("1K", 1_000),
    ("10K", 10_000),
    ("100K", 100_000),
]

N_ITER = 20


def median_time(fn, n=N_ITER):
    """Run fn() n times, return median elapsed seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sorted(times)[len(times) // 2]


def fmt(seconds):
    """Format seconds into a human-readable string."""
    us = seconds * 1e6
    if us < 1000:
        return f"{us:.1f} us"
    ms = us / 1000
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{seconds:.2f} s"


def make_cls_data(n, p):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=max(2, p // 2),
        n_classes=2, random_state=42,
    )
    return X, y


def make_reg_data(n, p):
    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=p, noise=0.1, random_state=42,
    )
    return X, y


def make_cluster_data(n, p):
    X, y = make_blobs(n_samples=n, n_features=p, centers=8, random_state=42)
    return X, y


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_regressors():
    print("\n### Regressors (fit)\n")
    print(f"{'Algorithm':<25} {'Dataset':<18} {'Time':>12}")
    print("-" * 58)

    models = [
        ("LinearRegression", lambda: LinearRegression()),
        ("Ridge", lambda: Ridge()),
        ("Lasso", lambda: Lasso()),
        ("ElasticNet", lambda: ElasticNet()),
    ]

    for name, make_model in models:
        for label, n, p in SIZES:
            X, y = make_reg_data(n, p)
            model = make_model()
            t = median_time(lambda: model.fit(X, y))
            print(f"{name:<25} {label:<18} {fmt(t):>12}")


def bench_classifiers():
    print("\n### Classifiers (fit)\n")
    print(f"{'Algorithm':<25} {'Dataset':<18} {'Time':>12}")
    print("-" * 58)

    models = [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=200)),
        ("DecisionTree", lambda: DecisionTreeClassifier(random_state=42)),
        ("RandomForest", lambda: RandomForestClassifier(random_state=42, n_jobs=-1)),
        ("GaussianNB", lambda: GaussianNB()),
        ("KNeighborsClassifier", lambda: KNeighborsClassifier()),
        ("SVC(linear)", lambda: SVC(kernel="linear")),
        ("SVC(rbf)", lambda: SVC(kernel="rbf")),
        ("GradientBoosting", lambda: GradientBoostingClassifier(random_state=42)),
        ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(random_state=42)),
    ]

    for name, make_model in models:
        for label, n, p in SIZES:
            X, y = make_cls_data(n, p)
            model = make_model()
            # Skip slow combos
            if name.startswith("SVC") and n > 1000:
                n_iter = 5
            else:
                n_iter = N_ITER
            t = median_time(lambda: model.fit(X, y), n=n_iter)
            print(f"{name:<25} {label:<18} {fmt(t):>12}")


def bench_transformers():
    print("\n### Transformers\n")
    print(f"{'Algorithm':<25} {'Op':<12} {'Dataset':<18} {'Time':>12}")
    print("-" * 70)

    for label, n, p in SIZES:
        X, _ = make_reg_data(n, p)
        n_comp = min(p, 10)

        # StandardScaler
        scaler = StandardScaler()
        t = median_time(lambda: scaler.fit(X))
        print(f"{'StandardScaler':<25} {'fit':<12} {label:<18} {fmt(t):>12}")
        scaler.fit(X)
        t = median_time(lambda: scaler.transform(X))
        print(f"{'StandardScaler':<25} {'transform':<12} {label:<18} {fmt(t):>12}")

        # PCA
        pca = PCA(n_components=n_comp)
        t = median_time(lambda: pca.fit(X))
        print(f"{'PCA':<25} {'fit':<12} {label:<18} {fmt(t):>12}")
        pca.fit(X)
        t = median_time(lambda: pca.transform(X))
        print(f"{'PCA':<25} {'transform':<12} {label:<18} {fmt(t):>12}")


def bench_clustering():
    print("\n### Clustering (fit)\n")
    print(f"{'Algorithm':<25} {'Dataset':<18} {'Time':>12}")
    print("-" * 58)

    for label, n, p in SIZES:
        X, _ = make_cluster_data(n, p)

        km = KMeans(n_clusters=8, n_init=3, random_state=42)
        t = median_time(lambda: km.fit(X), n=min(N_ITER, 10))
        print(f"{'KMeans':<25} {label:<18} {fmt(t):>12}")

        if n <= 10_000:
            db = DBSCAN(eps=3.0, min_samples=5)
            t = median_time(lambda: db.fit(X), n=min(N_ITER, 10))
            print(f"{'DBSCAN':<25} {label:<18} {fmt(t):>12}")


def bench_metrics():
    print("\n### Metrics\n")
    print(f"{'Metric':<25} {'Size':<12} {'Time':>12}")
    print("-" * 52)

    for label, n in METRIC_SIZES:
        y_true_cls = np.array([i % 3 for i in range(n)])
        y_pred_cls = np.array([(i + 1) % 3 for i in range(n)])
        y_true_reg = np.arange(n, dtype=np.float64) * 0.1
        y_pred_reg = y_true_reg + 0.01

        t = median_time(lambda: accuracy_score(y_true_cls, y_pred_cls))
        print(f"{'accuracy_score':<25} {label:<12} {fmt(t):>12}")

        t = median_time(lambda: f1_score(y_true_cls, y_pred_cls, average="macro"))
        print(f"{'f1_score':<25} {label:<12} {fmt(t):>12}")

        t = median_time(lambda: mean_squared_error(y_true_reg, y_pred_reg))
        print(f"{'mean_squared_error':<25} {label:<12} {fmt(t):>12}")

        t = median_time(lambda: r2_score(y_true_reg, y_pred_reg))
        print(f"{'r2_score':<25} {label:<12} {fmt(t):>12}")


def bench_predict():
    """Benchmark predict times for classifiers and regressors."""
    print("\n### Classifier predict\n")
    print(f"{'Algorithm':<25} {'Dataset':<18} {'Time':>12}")
    print("-" * 58)

    clf_models = [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=200)),
        ("DecisionTree", lambda: DecisionTreeClassifier(random_state=42)),
        ("RandomForest", lambda: RandomForestClassifier(random_state=42, n_jobs=-1)),
        ("GaussianNB", lambda: GaussianNB()),
        ("KNeighborsClassifier", lambda: KNeighborsClassifier(n_jobs=-1)),
    ]

    for name, make_model in clf_models:
        for label, n, p in SIZES:
            X, y = make_cls_data(n, p)
            model = make_model()
            model.fit(X, y)
            t = median_time(lambda: model.predict(X))
            print(f"{name:<25} {label:<18} {fmt(t):>12}")

    print("\n### Regressor predict\n")
    print(f"{'Algorithm':<25} {'Dataset':<18} {'Time':>12}")
    print("-" * 58)

    reg_models = [
        ("LinearRegression", lambda: LinearRegression()),
        ("Ridge", lambda: Ridge()),
        ("Lasso", lambda: Lasso()),
        ("ElasticNet", lambda: ElasticNet()),
    ]

    for name, make_model in reg_models:
        for label, n, p in SIZES:
            X, y = make_reg_data(n, p)
            model = make_model()
            model.fit(X, y)
            t = median_time(lambda: model.predict(X))
            print(f"{name:<25} {label:<18} {fmt(t):>12}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("sklearn benchmark suite")
    print(f"sklearn version: {__import__('sklearn').__version__}")
    print(f"numpy version:   {np.__version__}")
    print(f"iterations:      {N_ITER} (median)")
    print("=" * 70)

    bench_regressors()
    bench_classifiers()
    bench_predict()
    bench_transformers()
    bench_clustering()
    bench_metrics()

    print("\n" + "=" * 70)
    print("Done.")
