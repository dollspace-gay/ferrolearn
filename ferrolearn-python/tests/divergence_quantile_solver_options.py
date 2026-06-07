"""Divergence pin: ferrolearn QuantileRegressor rejects non-empty solver_options
where sklearn 1.5.2 accepts it and returns a normal fit.

Verification model B (goal.md): the expected value is computed by the LIVE
sklearn 1.5.2 oracle in this same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Divergence
----------
sklearn `QuantileRegressor.fit` forwards `solver_options` straight to
`scipy.optimize.linprog(..., options=solver_options)`
(`sklearn/linear_model/_quantile.py:271-277`):

    result = linprog(
        c=c, A_eq=A_eq, b_eq=b_eq, method=solver, options=solver_options,
    )

The `solver_options` parameter is constrained only to `[dict, None]`
(`_quantile.py:125`) — ANY dict is a valid construction. A non-empty dict such
as `{"presolve": False}` / `{"disp": True}` / `{"presolve": True}` is a HiGHS
TUNING knob that does NOT change the LP optimum: sklearn returns a normal fit
with `coef_`/`intercept_`/`predict` identical to the `solver_options=None` fit
(verified live below).

ferrolearn (`ferrolearn-linear/src/quantile_regressor.rs:619-629`) instead
raises on ANY non-empty `solver_options`, marshalled by
`ferrolearn-python/src/extras.rs:764-772` (`quantile_fit_err`) to
`NotImplementedError`. So for the SAME constructor call, sklearn SUCCEEDS and
returns predictions while ferrolearn ERRORS — an observable behavior
divergence. sklearn's RESULT (the fit / its predictions) is exactly what
ferrolearn produces with `solver_options=None`, i.e. ferrolearn CAN produce
sklearn's observable success but refuses to.

Tracking: #2168
"""

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.linear_model import QuantileRegressor as SkQuantileRegressor


def _dataset():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([2.0, 4.5, 5.5, 8.5, 9.5, 12.5])
    return X, y


@pytest.mark.parametrize(
    "opts",
    [
        {"presolve": False},
        {"presolve": True},
        {"disp": True},
    ],
)
def test_quantile_nonempty_solver_options_matches_sklearn(opts):
    """sklearn ACCEPTS a non-empty `solver_options` (it is a HiGHS tuning knob
    forwarded to `scipy.optimize.linprog`, `_quantile.py:271-277`) and returns a
    normal fit whose predictions equal the `solver_options=None` fit. ferrolearn
    must produce the SAME observable success (same predictions), but it instead
    raises NotImplementedError.

    Expected value comes from the LIVE sklearn oracle in this test (R-CHAR-3).
    """
    X, y = _dataset()

    # LIVE oracle: sklearn succeeds and the optimum is unchanged vs None.
    sk = SkQuantileRegressor(
        quantile=0.5, alpha=0.0, solver="highs", solver_options=opts
    ).fit(X, y)
    sk_none = SkQuantileRegressor(
        quantile=0.5, alpha=0.0, solver="highs", solver_options=None
    ).fit(X, y)
    np.testing.assert_allclose(sk.predict(X), sk_none.predict(X), rtol=0, atol=1e-9)

    # ferrolearn must mirror sklearn's observable SUCCESS: a normal fit whose
    # predictions match the sklearn oracle. Currently ferrolearn raises
    # NotImplementedError here, so this fit() call fails -> test FAILS (#2168).
    fr = fl.QuantileRegressor(
        quantile=0.5, alpha=0.0, solver="highs", solver_options=opts
    ).fit(X, y)
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), rtol=0, atol=1e-6)
