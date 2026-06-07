"""Critic divergence pins: ferrolearn.IsolationForest parameter-validation
divergences vs scikit-learn 1.5.2 (`sklearn.ensemble.IsolationForest`), found
auditing the #2180 binding against the live oracle.

Tracking: #2181 (blocker). These tests are RELEASE-BLOCKER pins and are left
un-ignored (RED) so CI surfaces them until the generator fixes the validation.

The #2180 wrapper's docstring + design-doc entry claim its `_validate_static`
/ `_resolve_max_samples` "mirror sklearn's `_parameter_constraints`
(`_iforest.py:199-219`)". These tests show three places where the wrapper does
NOT mirror sklearn's observable accept/reject behavior. Every expected value is
computed by the LIVE sklearn 1.5.2 oracle in the same test (R-CHAR-3); none is
copied from the ferrolearn side.

These are REAL divergences, distinct from the honest RNG-substrate gap (exact
scores) and the honest NotImplementedError gaps (max_features / bootstrap /
warm_start). The generator must fix the validation to mirror sklearn's
`Interval(Integral, ...)` (which accepts bool) and sklearn's non-guarded
`int(max_samples * n_samples)` float resolution.
"""

import warnings

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.ensemble import IsolationForest as SkIsolationForest


def _iforest_data(seed=0):
    """40 tight inliers + 5 far outliers (n=45), same shape the #2180 suite uses."""
    rng = np.random.RandomState(seed)
    inliers = rng.randn(40, 2) * 0.3
    outliers = rng.uniform(low=-6.0, high=6.0, size=(5, 2))
    return np.r_[inliers, outliers]


def test_iforest_n_estimators_true_accepted_like_sklearn():
    """Divergence (tracking #2181): `_extras.py::IsolationForest._validate_static`
    (`ferrolearn-python/python/ferrolearn/_extras.py:1598-1604`) rejects a bool
    `n_estimators` with a `ValueError`, but sklearn's constraint
    `Interval(Integral, 1, None, closed='left')`
    (`sklearn/ensemble/_iforest.py:200`) ACCEPTS `True` (a `numbers.Integral`,
    == 1) and FITS one estimator.

    Live oracle: `SkIsolationForest(n_estimators=True).fit(X)` succeeds with
    `len(estimators_) == 1`. ferrolearn raises `ValueError`.
    """
    X = _iforest_data()

    # Oracle: sklearn ACCEPTS n_estimators=True (Integral) and builds 1 tree.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = SkIsolationForest(n_estimators=True, random_state=0).fit(X)
    assert len(sk.estimators_) == 1  # sklearn observable: True == 1 estimator

    # ferrolearn must mirror: accept True (== 1) and fit, NOT raise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fl.IsolationForest(n_estimators=True, random_state=0).fit(X)
    assert f.predict(X).shape == (X.shape[0],)


def test_iforest_max_samples_true_accepted_like_sklearn():
    """Divergence (tracking #2181): `_extras.py::IsolationForest._resolve_max_samples`
    (`_extras.py:1660-1664`) rejects a bool `max_samples` with a `ValueError`,
    but sklearn treats `max_samples` under `Interval(Integral, 1, None)`
    (`_iforest.py:203`/`:306`) so `True` (Integral, == 1) resolves to
    `max_samples_ == 1` and FITS.

    Live oracle: `SkIsolationForest(max_samples=True).fit(X)` succeeds with
    `max_samples_ == 1`. ferrolearn raises `ValueError`.
    """
    X = _iforest_data()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = SkIsolationForest(max_samples=True, random_state=0).fit(X)
    assert sk.max_samples_ == 1  # sklearn observable: True resolves to 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fl.IsolationForest(max_samples=True, random_state=0).fit(X)
    assert f.predict(X).shape == (X.shape[0],)


def test_iforest_max_samples_float_truncating_to_zero_matches_sklearn():
    """Divergence (tracking #2181): `_extras.py::IsolationForest._resolve_max_samples`
    (`_extras.py:1683`) guards the float path with `max(1, int(ms * n))`, so a
    tiny float (e.g. 0.01 on n=45 -> int(0.45) == 0) is silently clamped UP to
    1 and the fit SUCCEEDS. sklearn does NOT guard: `_iforest.py:318`
    `max_samples = int(self.max_samples * X.shape[0])` yields 0, and the fit
    then RAISES `ValueError` ("zero-size array to reduction operation maximum
    which has no identity").

    The two libraries therefore disagree on observable behavior for the SAME
    input: sklearn raises, ferrolearn succeeds. Live oracle pins sklearn's raise.
    """
    X = _iforest_data()  # n_samples = 45; int(0.01 * 45) == int(0.45) == 0

    # Oracle: sklearn RAISES (ValueError) for this float -> max_samples_ == 0.
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SkIsolationForest(max_samples=0.01, random_state=0).fit(X)

    # ferrolearn must mirror sklearn's observable behavior: also raise.
    with pytest.raises(ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fl.IsolationForest(max_samples=0.01, random_state=0).fit(X)
