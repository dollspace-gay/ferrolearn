"""Divergence guards for the ferrolearn.Lars / ferrolearn.LassoLars bindings
(ferrolearn issue #2174), pinned against the LIVE scikit-learn 1.5.2 oracle.

Verification model B (goal.md): every expected value is computed by the live
sklearn 1.5.2 oracle in the SAME test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Targets:
  - `ferrolearn-python/src/extras.rs::RsLars` (lines 927-1004) over
    `ferrolearn_linear::Lars` — the LARS-direct path.
  - `ferrolearn-python/python/ferrolearn/_extras.py::Lars` / `LassoLars`
    (lines 339-609) — the sklearn-API wrappers.

Each test pins a specific divergence from
`sklearn.linear_model.Lars` / `LassoLars` and currently FAILS against the
working-tree build. The sklearn site mirrored is cited per test.
"""

import warnings

import numpy as np
import pytest
import sklearn.linear_model as sk

import ferrolearn as fl


# ---------------------------------------------------------------------------
# Divergence 1: Lars coef_/predict diverge from sklearn for deep active sets.
#
# `Lars().fit(X, y)` with n_features ~= n_samples runs the LARS path deep into
# the active set. sklearn's `lars_path` (sklearn/linear_model/_least_angle.py,
# `coef_` set at `:1125` `self.coef_[k] = coef_path[:, -1]`) yields a
# well-defined coefficient vector; ferrolearn's `RsLars` core
# (`extras.rs:955` `ferrolearn_linear::Lars`) produces a COMPLETELY different
# coef_ (max abs differs by many orders of magnitude) once the path runs past
# ~30-40 active features. sklearn emits NO warning for this data.
# Tracking: #2175
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="divergence #2175: Lars LARS-direct path diverges "
                          "for deep active sets (k>=~40)", strict=True)
def test_lars_deep_path_coef_matches_sklearn():
    rng = np.random.RandomState(0)
    X = rng.randn(45, 40)
    y = rng.randn(45)
    sk_model = sk.Lars().fit(X, y)  # default n_nonzero_coefs=500 -> full path
    fl_model = fl.Lars().fit(X, y)
    # Live oracle is sk_model.coef_; assert ferrolearn matches it.
    assert np.allclose(fl_model.coef_, sk_model.coef_, atol=1e-6, rtol=1e-6), (
        f"max |coef diff| = {np.max(np.abs(fl_model.coef_ - sk_model.coef_)):.3e}"
    )


@pytest.mark.xfail(reason="divergence #2175: Lars predict diverges for deep "
                          "active sets (k>=~40)", strict=True)
def test_lars_deep_path_predict_matches_sklearn():
    rng = np.random.RandomState(0)
    X = rng.randn(45, 40)
    y = rng.randn(45)
    Xt = rng.randn(8, 40)
    sk_model = sk.Lars().fit(X, y)
    fl_model = fl.Lars().fit(X, y)
    assert np.allclose(fl_model.predict(Xt), sk_model.predict(Xt),
                       atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Divergence 2: Lars/LassoLars raise on exactly-collinear (duplicate-column) X.
#
# sklearn's LARS handles an exactly-collinear feature (a duplicate column) by
# leaving the tied feature out of the active set; `Lars`/`LassoLars` fit and
# return a finite coef_ (sklearn/linear_model/_least_angle.py lars_path drops
# the redundant atom). ferrolearn's `RsLars`/`RsLassoLars` core raises
# `ValueError: Numerical instability: LARS Gram matrix is singular`.
# Tracking: #2176
# ---------------------------------------------------------------------------

def _dup_col_data():
    rng = np.random.RandomState(3)
    base = rng.randn(30, 4)
    X = np.hstack([base, base[:, :1]])  # last column == first column
    y = rng.randn(30)
    return X, y


@pytest.mark.xfail(reason="divergence #2176: Lars raises on duplicate-column X "
                          "where sklearn fits", strict=True)
def test_lars_duplicate_column_matches_sklearn():
    X, y = _dup_col_data()
    sk_model = sk.Lars(n_nonzero_coefs=3).fit(X, y)  # oracle
    fl_model = fl.Lars(n_nonzero_coefs=3).fit(X, y)  # currently raises ValueError
    assert np.allclose(fl_model.coef_, sk_model.coef_, atol=1e-6, rtol=1e-6)


@pytest.mark.xfail(reason="divergence #2176: LassoLars raises on duplicate-"
                          "column X where sklearn fits", strict=True)
def test_lassolars_duplicate_column_matches_sklearn():
    X, y = _dup_col_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk_model = sk.LassoLars(alpha=0.1).fit(X, y)  # oracle
    fl_model = fl.LassoLars(alpha=0.1).fit(X, y)  # currently raises ValueError
    assert np.allclose(fl_model.coef_, sk_model.coef_, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Divergence 3: copy_X validation gap.
#
# sklearn's `_parameter_constraints["copy_X"] == ['boolean']`
# (sklearn/linear_model/_least_angle.py:1035-ish); a non-bool copy_X is an
# `InvalidParameterError` (a `ValueError` subclass) at fit. ferrolearn's
# `_extras.py::Lars._validate` (lines 400-446) validates n_nonzero_coefs / eps /
# precompute / jitter but NOT copy_X, so a bogus copy_X is silently accepted.
# Tracking: #2177
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="divergence #2177: Lars accepts non-bool copy_X "
                          "where sklearn raises", strict=True)
def test_lars_copy_x_invalid_rejected_like_sklearn():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 5)
    y = rng.randn(20)
    # Oracle: sklearn rejects a non-bool copy_X with a ValueError-catchable error.
    with pytest.raises(ValueError):
        sk.Lars(copy_X="not-a-bool").fit(X, y)
    # ferrolearn must mirror that rejection.
    with pytest.raises(ValueError):
        fl.Lars(copy_X="not-a-bool").fit(X, y)


@pytest.mark.xfail(reason="divergence #2177: LassoLars accepts non-bool copy_X "
                          "where sklearn raises", strict=True)
def test_lassolars_copy_x_invalid_rejected_like_sklearn():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 5)
    y = rng.randn(20)
    with pytest.raises(ValueError):
        sk.LassoLars(copy_X="not-a-bool").fit(X, y)
    with pytest.raises(ValueError):
        fl.LassoLars(copy_X="not-a-bool").fit(X, y)
