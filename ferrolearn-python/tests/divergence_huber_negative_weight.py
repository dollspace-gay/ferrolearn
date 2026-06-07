"""Divergence pin (#2159): `ferrolearn.HuberRegressor.fit(X, y, sample_weight=w)`
rejects a negative weight that `sklearn.linear_model.HuberRegressor` accepts.

sklearn's `HuberRegressor.fit` (`sklearn/linear_model/_huber.py:306`) runs
`sample_weight` through `_check_sample_weight`, which does NOT forbid negative
weights — the fit converges and returns `coef_`/`intercept_`/`scale_`. ferrolearn
(the Rust core `huber_regressor.rs:682-687`, surfaced through the binding) raises
`ValueError("weights must be non-negative")`.

Expected values are computed live from sklearn 1.5.2 in the SAME test (R-CHAR-3),
never copied from ferrolearn.

Run:
    cd ferrolearn-python && PYTHONPATH=python \
        python3 -m pytest tests/divergence_huber_negative_weight.py -q
"""

import numpy as np
import pytest
from sklearn.linear_model import HuberRegressor as SkHuberRegressor

import ferrolearn as fl


def _data_with_negative_weight():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 2)
    y = X @ np.array([2.0, -1.0]) + 1.0 + 0.3 * rng.randn(12)
    y[3] += 8.0
    w = np.ones(12)
    w[0] = -1.0
    return X, y, w


def test_huber_negative_sample_weight_matches_sklearn():
    """sklearn accepts a negative `sample_weight` and returns a fitted model;
    ferrolearn must reproduce its `coef_`/`intercept_`/`scale_`. Currently
    ferrolearn raises ValueError, so this FAILS (tracking #2159)."""
    X, y, w = _data_with_negative_weight()

    sk = SkHuberRegressor(
        alpha=1e-4, epsilon=1.35, max_iter=200, tol=1e-5
    ).fit(X, y, sample_weight=w)

    fr = fl.HuberRegressor(
        alpha=1e-4, epsilon=1.35, max_iter=200, tol=1e-5
    ).fit(X, y, sample_weight=w)

    np.testing.assert_allclose(np.asarray(fr.coef_), sk.coef_, atol=1e-3)
    assert abs(fr.intercept_ - sk.intercept_) < 1e-3
    assert abs(fr.scale_ - sk.scale_) < 1e-3
