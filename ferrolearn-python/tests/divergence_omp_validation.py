"""Divergence pins: ferrolearn.OrthogonalMatchingPursuit (#2172) vs the live
sklearn 1.5.2 oracle `sklearn.linear_model.OrthogonalMatchingPursuit`.

The shipped binding (extras.rs `RsOrthogonalMatchingPursuit`, _extras.py
`OrthogonalMatchingPursuit`) reproduces sklearn's coef_/intercept_/predict on the
HAPPY path (covered by divergence_extras.py::test_omp_*). These tests pin the
VALIDATION / parameter-semantics gaps the happy-path tests skip — every one is a
case where sklearn REJECTS (or REINTERPRETS) an input that ferrolearn silently
accepts and mis-fits.

Verification model B (goal.md): every expected behavior is produced by the LIVE
sklearn oracle in the SAME test (R-CHAR-3) — never copied from ferrolearn.

`sklearn.linear_model._omp.py:735-740` `_parameter_constraints`:
    "n_nonzero_coefs": [Interval(Integral, 1, None, closed="left"), None]
    "tol":             [Interval(Real, 0, None, closed="left"), None]
    "precompute":      [StrOptions({"auto"}), "boolean"]
`_omp.py:782-788`: when `tol is not None`, `n_nonzero_coefs_` is forced to None
(tol OVERRIDES n_nonzero_coefs).
`InvalidParameterError` is a subclass of `ValueError`.
"""

import numpy as np
import pytest

import ferrolearn as fl

from sklearn.datasets import make_regression
from sklearn.linear_model import (
    OrthogonalMatchingPursuit as SkOrthogonalMatchingPursuit,
)


def _data(seed=0, n=40, p=6, informative=3):
    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=informative, random_state=seed,
    )
    return X, y


def _sk_rejects(**ctor):
    """Live oracle: assert sklearn REJECTS this ctor at fit time, and return the
    raised exception TYPE (a ValueError subclass). R-CHAR-3: the expected
    'rejects' fact is observed from sklearn here, not assumed."""
    X, y = _data()
    with pytest.raises(ValueError) as exc:
        SkOrthogonalMatchingPursuit(**ctor).fit(X, y)
    return type(exc.value)


# ---------------------------------------------------------------------------
# Divergence 1: precompute outside {'auto', True, False}.
# sklearn `_omp.py:739` constraint `[StrOptions({"auto"}), "boolean"]` -> any
# other value raises InvalidParameterError (ValueError) at fit. ferrolearn's
# binding (extras.rs:817-819) takes `precompute: Option<PyAny>` and IGNORES it,
# so it accepts an invalid precompute that sklearn rejects.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_precompute", ["bogus", 5])
def test_omp_invalid_precompute_rejected_like_sklearn(bad_precompute):
    """sklearn rejects `precompute` not in {'auto', True, False}; ferrolearn must
    too. Oracle: sklearn raises ValueError live (R-CHAR-3)."""
    sk_exc = _sk_rejects(precompute=bad_precompute)
    assert issubclass(sk_exc, ValueError)  # observed: sklearn rejects.

    X, y = _data()
    # ferrolearn must raise the SAME class of error (ValueError) on this input.
    with pytest.raises(ValueError):
        fl.OrthogonalMatchingPursuit(precompute=bad_precompute).fit(X, y)


# ---------------------------------------------------------------------------
# Divergence 2: n_nonzero_coefs=0.
# sklearn `_omp.py:736` constraint `Interval(Integral, 1, None, closed="left")`
# -> 0 raises InvalidParameterError (ValueError). ferrolearn's binding takes
# `Option<usize>` (extras.rs:814); 0 is a valid usize, threaded to the core,
# producing a degenerate all-zero fit instead of rejecting.
# ---------------------------------------------------------------------------

def test_omp_n_nonzero_coefs_zero_rejected_like_sklearn():
    """sklearn requires n_nonzero_coefs >= 1; ferrolearn must reject 0, not fit a
    degenerate all-zero coef_. Oracle: sklearn raises ValueError live."""
    sk_exc = _sk_rejects(n_nonzero_coefs=0)
    assert issubclass(sk_exc, ValueError)  # observed: sklearn rejects.

    X, y = _data()
    with pytest.raises(ValueError):
        fl.OrthogonalMatchingPursuit(n_nonzero_coefs=0).fit(X, y)


# ---------------------------------------------------------------------------
# Divergence 3: tol < 0.
# sklearn `_omp.py:737` constraint `Interval(Real, 0, None, closed="left")` -> a
# negative tol raises InvalidParameterError (ValueError). ferrolearn threads any
# f64 tol to the core (extras.rs:840) and accepts it.
# ---------------------------------------------------------------------------

def test_omp_negative_tol_rejected_like_sklearn():
    """sklearn requires tol >= 0; ferrolearn must reject a negative tol. Oracle:
    sklearn raises ValueError live."""
    sk_exc = _sk_rejects(tol=-1.0)
    assert issubclass(sk_exc, ValueError)  # observed: sklearn rejects.

    X, y = _data()
    with pytest.raises(ValueError):
        fl.OrthogonalMatchingPursuit(tol=-1.0).fit(X, y)


# ---------------------------------------------------------------------------
# Divergence 4: BOTH n_nonzero_coefs AND tol set -> tol OVERRIDES.
# sklearn `_omp.py:786-787`: `elif self.tol is not None: self.n_nonzero_coefs_ =
# None` — when tol is set, n_nonzero_coefs is IGNORED and the fit equals the
# tol-only fit. ferrolearn's binding passes BOTH to the core (extras.rs:837-841),
# so n_nonzero_coefs still caps the support -> a different (smaller-support) fit.
# ---------------------------------------------------------------------------

def test_omp_tol_overrides_n_nonzero_coefs_like_sklearn():
    """With both set, sklearn ignores n_nonzero_coefs (tol wins); ferrolearn must
    produce the tol-only fit, not a 2-atom fit. Oracle: the expected coef_ is the
    live sklearn tol-only fit (R-CHAR-3)."""
    X, y = _data(n=30, p=5, seed=2)
    tol = 5.0

    # Oracle: sklearn with both set == sklearn with tol only (tol overrides).
    sk_both = SkOrthogonalMatchingPursuit(n_nonzero_coefs=2, tol=tol).fit(X, y)
    sk_tol_only = SkOrthogonalMatchingPursuit(tol=tol).fit(X, y)
    np.testing.assert_allclose(sk_both.coef_, sk_tol_only.coef_, rtol=0, atol=1e-12)
    expected_nnz = int(np.count_nonzero(sk_both.coef_))  # observed: 5 (tol path).

    fr_both = fl.OrthogonalMatchingPursuit(n_nonzero_coefs=2, tol=tol).fit(X, y)
    # ferrolearn must match sklearn's tol-overrides fit.
    np.testing.assert_allclose(
        np.asarray(fr_both.coef_), sk_both.coef_, rtol=0, atol=1e-9,
        err_msg="tol must override n_nonzero_coefs (sklearn _omp.py:786-787)",
    )
    assert int(np.count_nonzero(np.asarray(fr_both.coef_))) == expected_nnz
