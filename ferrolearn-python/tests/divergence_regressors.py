"""Divergence guards for ferrolearn-python regressor bindings (unit #2039).

Targets: `ferrolearn-python/src/regressors.rs`
(RsLinearRegression / RsRidge / RsLasso / RsElasticNet) + the Python wrappers
`ferrolearn-python/python/ferrolearn/_regressors.py`
(class LinearRegression / Ridge / Lasso / ElasticNet), mirroring
`sklearn.linear_model.LinearRegression` (`sklearn/linear_model/_base.py:465`),
`sklearn.linear_model.Ridge` (`sklearn/linear_model/_ridge.py:1016`),
`sklearn.linear_model.Lasso` (`sklearn/linear_model/_coordinate_descent.py:1154`),
and `sklearn.linear_model.ElasticNet`
(`sklearn/linear_model/_coordinate_descent.py:729`).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Contents:
  - THREE FAILING constructor-ABI pins (single-wrapper-fixable, R-DEV-2):
      * Ridge(0.5).alpha       -> sklearn 0.5; ferrolearn raises TypeError
      * Lasso(0.1).alpha       -> sklearn 0.1; ferrolearn raises TypeError
      * ElasticNet(0.1).alpha  -> sklearn 0.1; ferrolearn raises TypeError
        (the EN pin ALSO guards that l1_ratio STAYS keyword-only: only `alpha`
        moves before the `*`. sklearn `ElasticNet(0.1, 0.7)` raises TypeError;
        ferrolearn must too.)
  - PASSING API-conformance green guards for ALL FOUR estimators (method/
    attribute surface + value parity against the live oracle). LinearRegression
    and Ridge are deterministic closed-form -> element-wise to 1e-8.
    Lasso/ElasticNet are coordinate-descent -> assert the exact zero/non-zero
    SUPPORT SET matches sklearn and a LOOSE value tolerance (atol=1e-2), because
    ferrolearn's CD stops at a slightly different point than sklearn's dual-gap
    criterion (downstream gap #412 / #411 / #417); NOT bit-exact.

The structural NOT-STARTED gaps (the 3 alpha-positional ABI fixes; the faked
Lasso/ElasticNet n_iter_; missing ctor params) are filed as -l blocker crosslink
issues. Only the 3 single-wrapper-fixable alpha pins are RED tests here, per
R-DEFER-3 (pin a FAILING test only for a divergence a fixer can close THIS
iteration in a single file).
"""

import inspect

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge as SkRidge


# Deterministic shared fixture. Full-rank, well-conditioned so the closed-form
# estimators are exact and the CD estimators converge to a stable support set.
_X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 4.0], [4.0, 5.0], [5.0, 7.0]])
_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# ---------------------------------------------------------------------------
# RED pins: alpha-positional constructor-ABI divergences
# (single-wrapper-fixable, R-DEV-2). Expected values come from the live sklearn
# oracle, never from ferrolearn.
# ---------------------------------------------------------------------------


def test_red_ridge_alpha_positional():
    """Divergence: ferrolearn.Ridge makes `alpha` keyword-only.

    sklearn `Ridge.__init__(self, alpha=1.0, *, ...)`
    (`sklearn/linear_model/_ridge.py:893-895`) makes `alpha`
    positional-or-keyword; `Ridge(0.5).alpha == 0.5`.
    ferrolearn `_regressors.py::Ridge.__init__(self, *, alpha=1.0, ...)`
    puts `alpha` after the `*`, so `ferrolearn.Ridge(0.5)` raises
    TypeError. Tracking: blocker filed for this unit.
    """
    # Oracle: sklearn accepts alpha positionally.
    assert SkRidge(0.5).alpha == 0.5
    # ferrolearn must mirror this. Currently raises TypeError on construction.
    assert fl.Ridge(0.5).alpha == 0.5


def test_red_lasso_alpha_positional():
    """Divergence: ferrolearn.Lasso makes `alpha` keyword-only.

    sklearn `Lasso.__init__(self, alpha=1.0, *, ...)`
    (`sklearn/linear_model/_coordinate_descent.py:1310-1312`) makes `alpha`
    positional-or-keyword; `Lasso(0.1).alpha == 0.1`.
    ferrolearn `_regressors.py::Lasso.__init__(self, *, alpha=1.0, ...)` puts
    `alpha` after the `*`, so `ferrolearn.Lasso(0.1)` raises TypeError.
    Tracking: blocker filed for this unit.
    """
    assert SkLasso(0.1).alpha == 0.1
    assert fl.Lasso(0.1).alpha == 0.1


def test_red_elasticnet_alpha_positional():
    """Divergence: ferrolearn.ElasticNet makes `alpha` keyword-only.

    sklearn `ElasticNet.__init__(self, alpha=1.0, *, l1_ratio=0.5, ...)`
    (`sklearn/linear_model/_coordinate_descent.py:898-902`) makes ONLY `alpha`
    positional-or-keyword; `l1_ratio` and the rest stay keyword-only.
    `ElasticNet(0.1).alpha == 0.1`, but `ElasticNet(0.1, 0.7)` raises TypeError
    (l1_ratio cannot be positional).
    ferrolearn `_regressors.py::ElasticNet.__init__(self, *, alpha=1.0,
    l1_ratio=0.5, ...)` puts `alpha` after the `*`, so `ferrolearn.ElasticNet(0.1)`
    raises TypeError. The fix moves ONLY `alpha` before the `*`.
    Tracking: blocker filed for this unit.
    """
    # Oracle: sklearn accepts alpha positionally...
    assert SkElasticNet(0.1).alpha == 0.1
    # ...but l1_ratio stays keyword-only (positional l1_ratio is a TypeError).
    with pytest.raises(TypeError):
        SkElasticNet(0.1, 0.7)

    # ferrolearn must mirror BOTH facts: alpha positional works...
    assert fl.ElasticNet(0.1).alpha == 0.1
    # ...and l1_ratio stays keyword-only.
    with pytest.raises(TypeError):
        fl.ElasticNet(0.1, 0.7)


# ---------------------------------------------------------------------------
# GREEN guards: API conformance for all four estimators (must PASS).
# Deterministic closed-form estimators: element-wise to the oracle (1e-8).
# Coordinate-descent estimators: exact support set + loose value tolerance.
# ---------------------------------------------------------------------------


def test_green_linear_regression_api_conform():
    """LinearRegression (deterministic OLS): coef_/intercept_/predict/score
    match the live sklearn oracle element-wise (<=1e-8), both fit_intercept
    paths. Mirrors `sklearn/linear_model/_base.py:582` (fit).
    """
    sk = SkLinearRegression().fit(_X, _y)
    fr = fl.LinearRegression().fit(_X, _y)

    # Method/attribute surface.
    assert hasattr(fr, "coef_") and hasattr(fr, "intercept_")
    assert hasattr(fr, "predict") and hasattr(fr, "score")
    assert np.asarray(fr.coef_).shape == (_X.shape[1],)
    assert np.isscalar(fr.intercept_) or np.asarray(fr.intercept_).ndim == 0

    # Value parity vs the oracle.
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-8
    np.testing.assert_allclose(fr.predict(_X), sk.predict(_X), atol=1e-8)
    assert len(fr.predict(_X)) == _X.shape[0]
    assert abs(fr.score(_X, _y) - sk.score(_X, _y)) <= 1e-8

    # fit_intercept=False path also matches.
    sk0 = SkLinearRegression(fit_intercept=False).fit(_X, _y)
    fr0 = fl.LinearRegression(fit_intercept=False).fit(_X, _y)
    np.testing.assert_allclose(fr0.coef_, sk0.coef_, atol=1e-8)
    assert abs(float(fr0.intercept_) - float(sk0.intercept_)) <= 1e-8


def test_green_ridge_api_conform():
    """Ridge (deterministic cholesky path): coef_/intercept_/predict match the
    live sklearn oracle element-wise (<=1e-8). Mirrors
    `sklearn/linear_model/_ridge.py:914` (fit, default solver='auto'->cholesky).
    `alpha` passed by keyword (positional is the RED pin above).
    """
    sk = SkRidge(alpha=1.0).fit(_X, _y)
    fr = fl.Ridge(alpha=1.0).fit(_X, _y)

    assert hasattr(fr, "coef_") and hasattr(fr, "intercept_")
    assert hasattr(fr, "predict") and hasattr(fr, "score")
    assert np.asarray(fr.coef_).shape == (_X.shape[1],)

    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-8
    np.testing.assert_allclose(fr.predict(_X), sk.predict(_X), atol=1e-8)
    assert len(fr.predict(_X)) == _X.shape[0]
    assert abs(fr.score(_X, _y) - sk.score(_X, _y)) <= 1e-8


def test_green_lasso_api_conform():
    """Lasso (coordinate descent): coef_ matches the live sklearn oracle on the
    exact zero/non-zero SUPPORT SET, plus a LOOSE value tolerance (atol=1e-2).
    Mirrors `sklearn/linear_model/_coordinate_descent.py:932` (CD fit).

    NOT bit-exact: ferrolearn's CD stops at a slightly different point than
    sklearn's dual-gap stopping criterion (downstream gap #412); per the design
    doc the CONVERGED support set matches, so support + loose value is the
    contract here, not ULP parity.
    """
    sk = SkLasso(alpha=0.1).fit(_X, _y)
    fr = fl.Lasso(alpha=0.1).fit(_X, _y)

    assert hasattr(fr, "coef_") and hasattr(fr, "intercept_")
    assert hasattr(fr, "predict") and hasattr(fr, "score")
    assert np.asarray(fr.coef_).shape == (_X.shape[1],)

    # Exact zero/non-zero support set matches the oracle.
    np.testing.assert_array_equal(
        np.asarray(fr.coef_) != 0.0, np.asarray(sk.coef_) != 0.0
    )
    # Loose value parity (CD stopping gap #412 -> not bit-exact).
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-2)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-2
    assert len(fr.predict(_X)) == _X.shape[0]


def test_green_elasticnet_api_conform():
    """ElasticNet (coordinate descent): coef_ matches the live sklearn oracle on
    the exact support set plus a LOOSE value tolerance (atol=1e-2). Mirrors
    `sklearn/linear_model/_coordinate_descent.py:932` (CD fit). Same CD-stopping
    caveat as Lasso (#412); NOT bit-exact.
    """
    sk = SkElasticNet(alpha=0.1, l1_ratio=0.5).fit(_X, _y)
    fr = fl.ElasticNet(alpha=0.1, l1_ratio=0.5).fit(_X, _y)

    assert hasattr(fr, "coef_") and hasattr(fr, "intercept_")
    assert hasattr(fr, "predict") and hasattr(fr, "score")
    assert np.asarray(fr.coef_).shape == (_X.shape[1],)

    np.testing.assert_array_equal(
        np.asarray(fr.coef_) != 0.0, np.asarray(sk.coef_) != 0.0
    )
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-2)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-2
    assert len(fr.predict(_X)) == _X.shape[0]


def test_green_all_four_score_present():
    """All four estimators expose RegressorMixin `score` (R²) and a `predict`
    that returns n_samples values. fit_predict is not applicable to regressors.
    """
    for cls in (fl.LinearRegression, fl.Ridge, fl.Lasso, fl.ElasticNet):
        est = cls()
        assert hasattr(est, "score")
        est.fit(_X, _y)
        assert hasattr(est, "coef_") and hasattr(est, "intercept_")
        assert len(est.predict(_X)) == _X.shape[0]
