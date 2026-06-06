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
import warnings

import numpy as np
import pytest

import ferrolearn as fl
from sklearn.exceptions import ConvergenceWarning
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


# Distinct fixture: a 6-sample problem where the CD solver converges in a small,
# fixed number of sweeps (sklearn n_iter_ == 2 here), so the parity is exact and
# unambiguously below max_iter (1000) — proving n_iter_ is no longer the FAKE.
_X_NITER = np.array(
    [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0], [2.0, 1.0], [0.0, 3.0], [6.0, 5.0]]
)
_y_NITER = np.array([1.0, 2.0, 3.0, 1.5, 0.8, 4.2])


def test_lasso_elasticnet_n_iter_matches_sklearn():
    """`ferrolearn.Lasso`/`ElasticNet` expose the REAL coordinate-descent
    iteration count, matching sklearn EXACTLY — no longer the FAKE `max_iter`.

    Was: `_regressors.py::{Lasso,ElasticNet}.fit` set `self.n_iter_ =
    self.max_iter` (hardcoded 1000). Now they read the real count from the Rust
    `_RsLasso`/`_RsElasticNet` `n_iter_` getter over `FittedLasso`/
    `FittedElasticNet::n_iter()`, which is bit-faithful to sklearn's dual-gap CD
    stopping (`ferrolearn-linear` REQ-11/12). Mirrors sklearn's actual count
    `self.n_iter_.append(this_iter[0])` (`_coordinate_descent.py:1103`,
    single-target collapse `:1106`).

    Oracle (R-CHAR-3): expected value comes from the LIVE sklearn 1.5.2 call in
    this test, never copied from ferrolearn. (On this fixture sklearn yields
    n_iter_ == 2 for all three configs.)
    """
    # Lasso(alpha=0.5): exact parity with the live sklearn n_iter_.
    sk_lasso = SkLasso(alpha=0.5).fit(_X_NITER, _y_NITER)
    fr_lasso = fl.Lasso(alpha=0.5).fit(_X_NITER, _y_NITER)
    assert fr_lasso.n_iter_ == sk_lasso.n_iter_
    # No longer faked at max_iter (1000): the real count is far below it.
    assert fr_lasso.n_iter_ < fr_lasso.max_iter

    # ElasticNet(alpha=0.5): same exact parity.
    sk_enet = SkElasticNet(alpha=0.5).fit(_X_NITER, _y_NITER)
    fr_enet = fl.ElasticNet(alpha=0.5).fit(_X_NITER, _y_NITER)
    assert fr_enet.n_iter_ == sk_enet.n_iter_
    assert fr_enet.n_iter_ < fr_enet.max_iter

    # Lasso(alpha=0.1): a second alpha to guard against a coincidental match.
    sk_lasso2 = SkLasso(alpha=0.1).fit(_X_NITER, _y_NITER)
    fr_lasso2 = fl.Lasso(alpha=0.1).fit(_X_NITER, _y_NITER)
    assert fr_lasso2.n_iter_ == sk_lasso2.n_iter_
    assert fr_lasso2.n_iter_ < fr_lasso2.max_iter


def test_lasso_elasticnet_dual_gap_matches_sklearn():
    """`ferrolearn.Lasso`/`ElasticNet` expose the `dual_gap_` fitted attribute,
    matching sklearn's value at the returned solution.

    sklearn sets `self.dual_gap_ = dual_gaps_[0]`
    (`sklearn/linear_model/_coordinate_descent.py:1108`, single-target collapse;
    `:1111` multi-target) — the duality gap of the coordinate-descent objective
    at convergence. The Rust `FittedLasso::dual_gap()` / `FittedElasticNet::dual_gap()`
    (`ferrolearn-linear` REQ-11/12) are bit-faithful to sklearn's gap; the binding
    surfaces them via the `_RsLasso`/`_RsElasticNet` `dual_gap_` getter, set in
    `_regressors.py::{Lasso,ElasticNet}.fit` as `self.dual_gap_ =
    float(self._rs.dual_gap_)`.

    Oracle (R-CHAR-3): the expected value comes from the LIVE sklearn 1.5.2 call
    in this test, never copied from the ferrolearn side. Fixture matches the
    director's verified probe (Lasso dual_gap_ ~1.17e-4, ElasticNet ~1.06e-4).
    """
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([3.0, 2.5, 7.1, 6.0, 11.2])

    # Lasso(alpha=0.3): ferrolearn's dual_gap_ matches the live sklearn oracle.
    sk_lasso = SkLasso(alpha=0.3).fit(X, y)
    fr_lasso = fl.Lasso(alpha=0.3).fit(X, y)
    assert hasattr(fr_lasso, "dual_gap_")
    assert abs(float(fr_lasso.dual_gap_) - float(sk_lasso.dual_gap_)) < 1e-9

    # ElasticNet(alpha=0.3) (default l1_ratio=0.5, matching sklearn): same parity.
    sk_enet = SkElasticNet(alpha=0.3).fit(X, y)
    fr_enet = fl.ElasticNet(alpha=0.3).fit(X, y)
    assert hasattr(fr_enet, "dual_gap_")
    assert abs(float(fr_enet.dual_gap_) - float(sk_enet.dual_gap_)) < 1e-9


def test_linearregression_rank_singular_match_sklearn():
    """`ferrolearn.LinearRegression` exposes the `rank_`/`singular_` fitted
    attributes, matching sklearn EXACTLY.

    sklearn sets `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`
    on the (centered-when-`fit_intercept`) design matrix
    (`sklearn/linear_model/_base.py:687`; attr docstrings `rank_` `_base.py:505`
    `int`, `singular_` `_base.py:508` 1-D array). The Rust
    `FittedLinearRegression::rank()` / `singular_values()`
    (`ferrolearn-linear/src/linear_regression.rs:471`/`:478`, REQ-9 #374) capture
    both from the single-SVD `solve_lstsq` on the same operands; the binding
    surfaces them via the `_RsLinearRegression` `rank_`/`singular_` getter, set in
    `_regressors.py::LinearRegression.fit` as `self.rank_ = int(self._rs.rank_)` /
    `self.singular_ = np.array(self._rs.singular_)`.

    Oracle (R-CHAR-3): the expected values come from the LIVE sklearn 1.5.2 call
    in this test, never copied from the ferrolearn side. (On this 5x2 fixture
    sklearn yields `rank_ == 2`, `singular_ == [4.24264069, 1.41421356]`.)
    """
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([3.0, 2.5, 7.1, 6.0, 11.2])

    sk = SkLinearRegression().fit(X, y)
    fr = fl.LinearRegression().fit(X, y)

    assert hasattr(fr, "rank_")
    assert hasattr(fr, "singular_")
    assert fr.rank_ == sk.rank_
    np.testing.assert_allclose(fr.singular_, sk.singular_, atol=1e-8)


def test_ridge_multioutput_matches_sklearn():
    """`ferrolearn.Ridge` fits 2-D `Y` (multi-output), matching sklearn EXACTLY.

    sklearn `Ridge` accepts `y` of shape `(n_samples, n_targets)` and sets
    `coef_` of shape `(n_targets, n_features)` and `intercept_` of shape
    `(n_targets,)` (`sklearn/linear_model/_ridge.py:543`/`:550`; the per-target
    closed-form solve at `:207`/`:218`). The Rust `Fit<Array2, Array2> for Ridge`
    (`ferrolearn-linear/src/ridge.rs:725`) produces `FittedRidgeMulti` with
    `coefficients()` `(n_features, n_targets)` (`:713`) and `intercepts()`
    `(n_targets,)` (`:719`), shipped in #29; the binding
    `_RsRidgeMultiOutput.coef_` getter TRANSPOSES to `(n_targets, n_features)` to
    match sklearn's output contract, and `_regressors.py::Ridge.fit` routes the
    2-D-`y` path to it.

    Oracle (R-CHAR-3): every expected value comes from the LIVE sklearn 1.5.2
    call in this test, never copied from the ferrolearn side.
    """
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    Y = np.array([[3.0, 1.0], [2.5, 2.0], [7.1, 0.0], [6.0, 3.0], [11.2, 1.0]])

    sk = SkRidge(alpha=1.0).fit(X, Y)
    fr = fl.Ridge(alpha=1.0).fit(X, Y)

    # coef_ is (n_targets, n_features) == (2, 2), matching sklearn.
    assert fr.coef_.shape == (2, 2)
    assert fr.coef_.shape == sk.coef_.shape
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)

    # intercept_ is (n_targets,) == (2,), matching sklearn.
    assert fr.intercept_.shape == (2,)
    assert fr.intercept_.shape == sk.intercept_.shape
    np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=1e-8)

    # predict returns (n_samples, n_targets), matching sklearn element-wise.
    fr_pred = fr.predict(X[:2])
    sk_pred = sk.predict(X[:2])
    assert fr_pred.shape == sk_pred.shape == (2, 2)
    np.testing.assert_allclose(fr_pred, sk_pred, atol=1e-8)


def test_ridge_singleoutput_unchanged_by_multioutput_path():
    """Single-output `ferrolearn.Ridge` (1-D `y`) is UNCHANGED: `coef_` is 1-D and
    `intercept_` is a scalar float, matching sklearn (`_ridge.py:670-672` ravel).

    Guards that the multi-output routing in `Ridge.fit` did not regress the 1-D
    path. Oracle is the live sklearn 1.5.2 call (R-CHAR-3).
    """
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([3.0, 2.5, 7.1, 6.0, 11.2])

    sk = SkRidge(alpha=1.0).fit(X, y)
    fr = fl.Ridge(alpha=1.0).fit(X, y)

    assert fr.coef_.ndim == 1
    assert fr.coef_.shape == sk.coef_.shape == (2,)
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)

    # intercept_ is a scalar float (not an array), matching the 1-D contract.
    assert np.isscalar(fr.intercept_) or np.ndim(fr.intercept_) == 0
    assert abs(float(fr.intercept_) - float(sk.intercept_)) < 1e-8

    fr_pred = fr.predict(X[:2])
    assert fr_pred.ndim == 1
    np.testing.assert_allclose(fr_pred, sk.predict(X[:2]), atol=1e-8)


# ---------------------------------------------------------------------------
# Multi-output Ridge divergences (unit #2120 adversarial re-audit)
# ---------------------------------------------------------------------------


def test_red_ridge_multioutput_column_vector_y_shape_preserved():
    """Divergence: `ferrolearn.Ridge.fit` gates multi-output on `y.shape[1] > 1`
    (`_regressors.py:111`), so a column-vector `y` of shape `(n_samples, 1)`
    falls to the 1-D path and yields `coef_` 1-D / scalar `intercept_` / 1-D
    `predict`.

    sklearn `Ridge().fit(X, Y)` with `Y` shape `(n, 1)` PRESERVES the 2-D target
    structure: `coef_` is `(1, n_features)`, `intercept_` is `(1,)`, and
    `predict` returns `(n, 1)` (`sklearn/linear_model/_ridge.py:543` coef shape /
    `_base.py:319-324` ndim-2 intercept branch; multi_output shape preservation).

    On the (6,3)/(6,1) fixture the live oracle yields coef_ (1,3),
    intercept_ (1,), predict (6,1); ferrolearn yields (3,), scalar, (6,).
    Tracking: #2121
    """
    rng = np.random.RandomState(0)
    X = rng.rand(6, 3)
    Y = rng.rand(6, 1)

    sk = SkRidge(alpha=1.0).fit(X, Y)
    fr = fl.Ridge(alpha=1.0).fit(X, Y)

    # Live-oracle shapes (R-CHAR-3): never literal-copied from ferrolearn.
    assert sk.coef_.shape == (1, 3)
    assert np.shape(sk.intercept_) == (1,)
    assert sk.predict(X).shape == (6, 1)

    # ferrolearn must mirror the sklearn shapes for the (n,1) target.
    assert fr.coef_.shape == sk.coef_.shape
    assert np.shape(fr.intercept_) == np.shape(sk.intercept_)
    assert fr.predict(X).shape == sk.predict(X).shape


def test_red_ridge_multioutput_no_intercept_scalar_zero():
    """Divergence: `ferrolearn.Ridge(fit_intercept=False)` on 2-D `y` exposes
    `intercept_` as a zero ARRAY of shape `(n_targets,)` (the Rust
    `FittedRidgeMulti::intercepts()` marshalled by `_RsRidgeMultiOutput.intercept_`,
    `regressors.rs:287-294`).

    sklearn sets `intercept_` to the scalar `0.0` (shape `()`) whenever
    `fit_intercept=False`, REGARDLESS of target dimensionality
    (`sklearn/linear_model/_base.py:327`: `self.intercept_ = 0.0`).

    On the (10,3)/(10,2) fixture the live oracle yields scalar 0.0 (ndim 0);
    ferrolearn yields array([0., 0.]) (ndim 1, shape (2,)).
    Tracking: #2122
    """
    rng = np.random.RandomState(2)
    X = rng.rand(10, 3)
    Y = rng.rand(10, 2)
    Y[:, 1] *= 100.0

    sk = SkRidge(alpha=2.0, fit_intercept=False).fit(X, Y)
    fr = fl.Ridge(alpha=2.0, fit_intercept=False).fit(X, Y)

    # Live-oracle ground truth (R-CHAR-3): sklearn intercept_ is scalar 0.0.
    assert np.ndim(sk.intercept_) == 0
    assert float(sk.intercept_) == 0.0

    # ferrolearn must mirror sklearn's scalar-0.0 contract, not a zero vector.
    assert np.ndim(fr.intercept_) == np.ndim(sk.intercept_)


# ---------------------------------------------------------------------------
# Multi-output LinearRegression (unit #2123) — mirrors the multi-output Ridge
# fix. The Rust `Fit<Array2, Array2> for LinearRegression`
# (`ferrolearn-linear/src/linear_regression.rs:491`, REQ-7/#372 SHIPPED) is
# surfaced via `_RsLinearRegressionMultiOutput` and routed by
# `_regressors.py::LinearRegression.fit` on `y.ndim == 2`. KEY DIFFERENCE FROM
# RIDGE: `FittedMultiOutputLinearRegression::coefficients()` (:458) is ALREADY
# `(n_targets, n_features)`, so the `coef_` getter does NOT transpose.
# ---------------------------------------------------------------------------


def test_linearregression_multioutput_matches_sklearn():
    """`ferrolearn.LinearRegression` fits 2-D `Y` (multi-output), matching
    sklearn EXACTLY.

    sklearn `LinearRegression` accepts `y` of shape `(n_samples, n_targets)` and
    sets `coef_` `(n_targets, n_features)` and `intercept_` `(n_targets,)`
    (`sklearn/linear_model/_base.py:687` `coef_.T`; `_set_intercept`
    `:319-324`). The Rust `Fit<Array2, Array2> for LinearRegression`
    (`ferrolearn-linear/src/linear_regression.rs:491`) produces
    `FittedMultiOutputLinearRegression` with `coefficients()`
    `(n_targets, n_features)` (`:458`) and `intercepts()` `(n_targets,)`
    (`:465`), shipped #372; the binding `_RsLinearRegressionMultiOutput.coef_`
    marshals it straight through (NO transpose, unlike Ridge), and
    `_regressors.py::LinearRegression.fit` routes the 2-D-`y` path to it.

    Oracle (R-CHAR-3): every expected value comes from the LIVE sklearn 1.5.2
    call in this test, never copied from the ferrolearn side.
    """
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    Y = np.array([[3.0, 1.0], [2.5, 2.0], [7.1, 0.0], [6.0, 3.0], [11.2, 1.0]])

    sk = SkLinearRegression().fit(X, Y)
    fr = fl.LinearRegression().fit(X, Y)

    # coef_ is (n_targets, n_features) == (2, 2), matching sklearn.
    assert fr.coef_.shape == (2, 2)
    assert fr.coef_.shape == sk.coef_.shape
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)

    # intercept_ is (n_targets,) == (2,), matching sklearn.
    assert fr.intercept_.shape == (2,)
    assert fr.intercept_.shape == sk.intercept_.shape
    np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=1e-8)

    # predict returns (n_samples, n_targets), matching sklearn element-wise.
    fr_pred = fr.predict(X[:2])
    sk_pred = sk.predict(X[:2])
    assert fr_pred.shape == sk_pred.shape == (2, 2)
    np.testing.assert_allclose(fr_pred, sk_pred, atol=1e-8)


def test_linearregression_multioutput_column_vector_y():
    """A column-vector `y` of shape `(n_samples, 1)` PRESERVES the 2-D target
    structure, matching sklearn EXACTLY.

    sklearn `LinearRegression().fit(X, Y)` with `Y` shape `(n, 1)` yields
    `coef_` `(1, n_features)`, `intercept_` `(1,)`, and `predict` `(n, 1)`
    (`_base.py:687` `coef_.T`; multi_output shape preservation; the
    `MultiOutputMixin` tag means NO DataConversionWarning). ferrolearn gates the
    multi-output path on `y.ndim == 2` (NOT `y.shape[1] > 1`), so the `(n, 1)`
    target routes to `_RsLinearRegressionMultiOutput` too.

    Oracle (R-CHAR-3): all expected shapes/values from the live sklearn call.
    """
    rng = np.random.RandomState(0)
    X = rng.rand(6, 3)
    Y = rng.rand(6, 1)

    sk = SkLinearRegression().fit(X, Y)
    fr = fl.LinearRegression().fit(X, Y)

    # Live-oracle shapes (R-CHAR-3): never literal-copied from ferrolearn.
    assert sk.coef_.shape == (1, 3)
    assert np.shape(sk.intercept_) == (1,)
    assert sk.predict(X).shape == (6, 1)

    # ferrolearn must mirror the sklearn shapes for the (n, 1) target.
    assert fr.coef_.shape == sk.coef_.shape
    assert np.shape(fr.intercept_) == np.shape(sk.intercept_)
    assert fr.predict(X).shape == sk.predict(X).shape

    # ...and the VALUES too (deterministic OLS, element-wise to 1e-8).
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)
    np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=1e-8)
    np.testing.assert_allclose(fr.predict(X), sk.predict(X), atol=1e-8)


def test_linearregression_multioutput_no_intercept_scalar():
    """`ferrolearn.LinearRegression(fit_intercept=False)` on 2-D `y` exposes
    `intercept_` as the scalar `0.0` (ndim 0), matching sklearn EXACTLY.

    sklearn sets `intercept_` to the scalar `0.0` (shape `()`) whenever
    `fit_intercept=False`, REGARDLESS of target dimensionality
    (`sklearn/linear_model/_base.py:327`: `self.intercept_ = 0.0`). The wrapper
    sets `self.intercept_ = 0.0` on the multi-output `fit_intercept=False` path.

    Oracle (R-CHAR-3): the live sklearn call is the ground truth (scalar 0.0).
    """
    rng = np.random.RandomState(2)
    X = rng.rand(10, 3)
    Y = rng.rand(10, 2)
    Y[:, 1] *= 100.0

    sk = SkLinearRegression(fit_intercept=False).fit(X, Y)
    fr = fl.LinearRegression(fit_intercept=False).fit(X, Y)

    # Live-oracle ground truth (R-CHAR-3): sklearn intercept_ is scalar 0.0.
    assert np.ndim(sk.intercept_) == 0
    assert float(sk.intercept_) == 0.0

    # ferrolearn must mirror sklearn's scalar-0.0 contract, not a zero vector.
    assert np.ndim(fr.intercept_) == np.ndim(sk.intercept_)
    assert float(fr.intercept_) == 0.0

    # coef_ still (n_targets, n_features) and matches the oracle element-wise.
    assert fr.coef_.shape == sk.coef_.shape == (2, 3)
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)


def test_linearregression_singleoutput_unchanged_by_multioutput_path():
    """Single-output `ferrolearn.LinearRegression` (1-D `y`) is UNCHANGED: 1-D
    `coef_`, scalar `intercept_`, 1-D `predict`, AND the `rank_`/`singular_`
    lstsq diagnostics (#2101) are still set — matching sklearn (R-CHAR-3).

    Guards that the multi-output routing in `LinearRegression.fit` did not
    regress the 1-D path. Oracle is the live sklearn 1.5.2 call.
    """
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
    y = np.array([3.0, 2.5, 7.1, 6.0, 11.2])

    sk = SkLinearRegression().fit(X, y)
    fr = fl.LinearRegression().fit(X, y)

    assert fr.coef_.ndim == 1
    assert fr.coef_.shape == sk.coef_.shape == (2,)
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-8)

    # intercept_ is a scalar float (not an array), matching the 1-D contract.
    assert np.isscalar(fr.intercept_) or np.ndim(fr.intercept_) == 0
    assert abs(float(fr.intercept_) - float(sk.intercept_)) < 1e-8

    fr_pred = fr.predict(X[:2])
    assert fr_pred.ndim == 1
    np.testing.assert_allclose(fr_pred, sk.predict(X[:2]), atol=1e-8)

    # rank_/singular_ (1-D lstsq diagnostics, #2101) are STILL present + match.
    assert hasattr(fr, "rank_") and hasattr(fr, "singular_")
    assert fr.rank_ == sk.rank_
    np.testing.assert_allclose(fr.singular_, sk.singular_, atol=1e-8)


# ---------------------------------------------------------------------------
# RED pins: multi-output LinearRegression divergences (unit #2123 re-audit).
# The builder shipped the 2-D-`y` path but left two observable gaps vs sklearn
# 1.5.2. Every expected value below is computed by the LIVE sklearn oracle in
# the same test (R-CHAR-3); none is literal-copied from the ferrolearn side.
# ---------------------------------------------------------------------------


def test_red_linearregression_multioutput_rank_singular_missing():
    """Multi-output `ferrolearn.LinearRegression` is MISSING `rank_`/`singular_`
    where sklearn exposes them.

    sklearn sets `self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)`
    for the dense path REGARDLESS of single vs multi-output
    (`sklearn/linear_model/_base.py:687`); `rank_`/`singular_` are properties of
    `X` and are documented unconditional fitted attributes
    (`_base.py:505` `rank_`, `_base.py:508` `singular_`). So a 2-D-`y` fit has
    BOTH attributes. ferrolearn's wrapper sets them only on the 1-D path
    (`_regressors.py:96-97`) and skips them on the multi-output branch, so
    `hasattr(fr, "rank_")` is False where sklearn's is True.

    Oracle (R-CHAR-3): sklearn's live `rank_`/`singular_` are the ground truth.

    Tracking: #2124 (release-blocker, left un-ignored per goal.md R-DEFER-3).
    """
    rng = np.random.RandomState(0)
    X = rng.rand(8, 3)
    Y = rng.rand(8, 2)

    sk = SkLinearRegression().fit(X, Y)
    fr = fl.LinearRegression().fit(X, Y)

    # Live-oracle ground truth (R-CHAR-3): sklearn multi-output HAS both attrs.
    assert hasattr(sk, "rank_")
    assert hasattr(sk, "singular_")

    # ferrolearn must mirror the sklearn attribute surface for multi-output.
    assert hasattr(fr, "rank_"), "multi-output LinearRegression missing rank_"
    assert hasattr(fr, "singular_"), (
        "multi-output LinearRegression missing singular_"
    )
    assert fr.rank_ == sk.rank_
    np.testing.assert_allclose(fr.singular_, sk.singular_, atol=1e-8)


def test_red_linearregression_multioutput_pickle_predict_shape():
    """A pickled-then-unpickled multi-output `ferrolearn.LinearRegression`
    PRODUCES THE WRONG `predict`: the no-`_rs` fallback uses `X @ coef_` with
    `coef_` shape `(n_targets, n_features)`, which is a matmul shape error
    (should be `X @ coef_.T`).

    sklearn's `__getstate__`/`__setstate__` round-trips a fitted estimator and
    `predict` is unchanged (`coef_` is `(n_targets, n_features)`, the decision
    function uses `X @ coef_.T`, `_base.py:364`). ferrolearn's `__getstate__`
    pops `_rs` (`_regressors.py:110`), so the unpickled estimator's `predict`
    takes the `_predict_linear(X, coef_, intercept_)` fallback
    (`_regressors.py:106`) which computes `X @ coef_` (`_regressors.py:26`) —
    a `(n,3) @ (2,3)` matmul that RAISES for multi-output (or, when
    n_features == n_targets, silently returns the WRONG answer).

    Oracle (R-CHAR-3): sklearn's live unpickled `predict` is the ground truth.

    Tracking: #2125 (release-blocker, left un-ignored per goal.md R-DEFER-3).
    """
    import pickle

    rng = np.random.RandomState(0)
    X = rng.rand(8, 3)
    Y = rng.rand(8, 2)

    sk = SkLinearRegression().fit(X, Y)
    fr = fl.LinearRegression().fit(X, Y)

    # Live-oracle ground truth (R-CHAR-3): sklearn unpickled predict == fitted.
    sk_un = pickle.loads(pickle.dumps(sk))
    np.testing.assert_allclose(sk_un.predict(X), sk.predict(X), atol=1e-8)

    # ferrolearn must round-trip predict identically (no shape error, right
    # values). The fallback `X @ coef_` is wrong for `(n_targets, n_features)`.
    fr_un = pickle.loads(pickle.dumps(fr))
    fr_pred = fr_un.predict(X)
    assert fr_pred.shape == sk.predict(X).shape
    np.testing.assert_allclose(fr_pred, sk.predict(X), atol=1e-8)


# ---------------------------------------------------------------------------
# Multi-output Lasso / ElasticNet (unit #2126) — sklearn fits each target
# INDEPENDENTLY (multi `coef_` == stack of per-target single-output fits, NOT
# the coupled MultiTask variant). The wrapper loops the existing single-output
# `_RsLasso`/`_RsElasticNet` per target column on the `y.ndim == 2` gate,
# mirroring the just-committed multi-output Ridge/LinearRegression shape
# contract. `n_iter_` is the per-target Python list; `dual_gap_` is `(n_targets,)`
# (`sklearn/linear_model/_coordinate_descent.py:1071-1111`). Every expected value
# is computed by the LIVE sklearn 1.5.2 oracle in the same test (R-CHAR-3).
# ---------------------------------------------------------------------------

# Director's verified oracle fixture (sklearn 1.5.2), alpha=0.5.
_X_MO = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
_Y_MO = np.array(
    [[3.0, 1.0], [2.5, 2.0], [7.1, 0.0], [6.0, 3.0], [11.2, 1.0]]
)


def test_lasso_multioutput_matches_sklearn():
    """`ferrolearn.Lasso` fits 2-D `Y` (multi-output), matching sklearn EXACTLY.

    sklearn `Lasso` (subclass of `ElasticNet(MultiOutputMixin, ...)`) fits each
    target column independently in a per-target coordinate-descent loop
    (`sklearn/linear_model/_coordinate_descent.py:1071-1101`), so `coef_` is
    `(n_targets, n_features)`, `intercept_` is `(n_targets,)`, `n_iter_` is the
    per-target Python list (`:1101`, no single-target collapse at `:1106`), and
    `dual_gap_` is `(n_targets,)` (`:1111`). The wrapper loops the single-output
    `_RsLasso` per target column on the `y.ndim == 2` gate.

    Oracle (R-CHAR-3): every expected value comes from the LIVE sklearn 1.5.2
    call here. (Verified oracle: coef_ (2,2), intercept_=[-0.10680734, 1.55],
    n_iter_=[20, 2].)
    """
    sk = SkLasso(alpha=0.5).fit(_X_MO, _Y_MO)
    fr = fl.Lasso(alpha=0.5).fit(_X_MO, _Y_MO)

    # coef_ is (n_targets, n_features) == (2, 2), matching sklearn.
    assert fr.coef_.shape == (2, 2)
    assert fr.coef_.shape == sk.coef_.shape
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-7)

    # intercept_ is (n_targets,) == (2,), matching sklearn.
    assert fr.intercept_.shape == (2,)
    assert fr.intercept_.shape == sk.intercept_.shape
    np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=1e-7)

    # n_iter_ is the per-target LIST, matching sklearn exactly.
    assert isinstance(fr.n_iter_, list)
    assert fr.n_iter_ == list(sk.n_iter_)

    # dual_gap_ is (n_targets,), matching sklearn.
    assert np.shape(fr.dual_gap_) == np.shape(sk.dual_gap_) == (2,)
    np.testing.assert_allclose(fr.dual_gap_, sk.dual_gap_, atol=1e-9)

    # predict returns (n_samples, n_targets), matching sklearn element-wise.
    fr_pred = fr.predict(_X_MO[:2])
    sk_pred = sk.predict(_X_MO[:2])
    assert fr_pred.shape == sk_pred.shape == (2, 2)
    np.testing.assert_allclose(fr_pred, sk_pred, atol=1e-7)

    # Independence check: multi coef_ == stack of per-target single-output fits.
    stacked = np.stack(
        [fl.Lasso(alpha=0.5).fit(_X_MO, _Y_MO[:, j]).coef_ for j in range(2)]
    )
    np.testing.assert_allclose(fr.coef_, stacked, atol=1e-12)


def test_elasticnet_multioutput_matches_sklearn():
    """`ferrolearn.ElasticNet` fits 2-D `Y` (multi-output), matching sklearn.

    sklearn `ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel)` fits each
    target column independently (per-target CD loop
    `sklearn/linear_model/_coordinate_descent.py:1071-1101`): `coef_`
    `(n_targets, n_features)`, `intercept_` `(n_targets,)`, `n_iter_` the
    per-target list (`:1101`), `dual_gap_` `(n_targets,)` (`:1111`). The wrapper
    loops the single-output `_RsElasticNet` per target column.

    Oracle (R-CHAR-3): every expected value comes from the LIVE sklearn 1.5.2
    call here. (Verified oracle, l1_ratio=0.5:
    intercept_=[-0.10244714, 1.71171022], n_iter_=[14, 14].)
    """
    sk = SkElasticNet(alpha=0.5, l1_ratio=0.5).fit(_X_MO, _Y_MO)
    fr = fl.ElasticNet(alpha=0.5, l1_ratio=0.5).fit(_X_MO, _Y_MO)

    assert fr.coef_.shape == (2, 2)
    assert fr.coef_.shape == sk.coef_.shape
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-7)

    assert fr.intercept_.shape == (2,)
    assert fr.intercept_.shape == sk.intercept_.shape
    np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=1e-7)

    assert isinstance(fr.n_iter_, list)
    assert fr.n_iter_ == list(sk.n_iter_)

    assert np.shape(fr.dual_gap_) == np.shape(sk.dual_gap_) == (2,)
    np.testing.assert_allclose(fr.dual_gap_, sk.dual_gap_, atol=1e-9)

    fr_pred = fr.predict(_X_MO[:2])
    sk_pred = sk.predict(_X_MO[:2])
    assert fr_pred.shape == sk_pred.shape == (2, 2)
    np.testing.assert_allclose(fr_pred, sk_pred, atol=1e-7)

    stacked = np.stack(
        [
            fl.ElasticNet(alpha=0.5, l1_ratio=0.5).fit(_X_MO, _Y_MO[:, j]).coef_
            for j in range(2)
        ]
    )
    np.testing.assert_allclose(fr.coef_, stacked, atol=1e-12)


def test_lasso_elasticnet_multioutput_column_vector_y():
    """A column-vector `Y` of shape `(n, 1)` COLLAPSES the single target for both
    Lasso and ElasticNet, matching sklearn EXACTLY: `coef_` 1-D `(n_features,)`,
    `intercept_` `(1,)`, scalar `n_iter_`/`dual_gap_`, `predict` `(n,)`.

    This DIFFERS from multi-output Ridge/LinearRegression, which preserve the
    2-D `(1, n_features)` coef_ for a `(n, 1)` target. sklearn's coordinate-descent
    path instead collapses `n_targets == 1`
    (`_coordinate_descent.py:1106-1108` `if n_targets == 1: self.coef_ =
    coef_[0]`), regardless of whether `y` arrived 1-D or `(n, 1)`. Only
    `intercept_` stays length-1 (`_set_intercept` `y_offset` is `(1,)`,
    `_base.py:319`). Oracle (R-CHAR-3): every expected value is the live sklearn
    1.5.2 call, verified to give 1-D coef_ / `(1,)` intercept_ / `(n,)` predict.
    """
    Y1 = _Y_MO[:, [0]]  # shape (5, 1)

    for fr_cls, sk_cls, kw in (
        (fl.Lasso, SkLasso, {"alpha": 0.5}),
        (fl.ElasticNet, SkElasticNet, {"alpha": 0.5, "l1_ratio": 0.5}),
    ):
        sk = sk_cls(**kw).fit(_X_MO, Y1)
        fr = fr_cls(**kw).fit(_X_MO, Y1)

        # Live-oracle shapes (R-CHAR-3): never literal-copied from ferrolearn.
        # sklearn COLLAPSES the single target: coef_ is 1-D, intercept_ is (1,),
        # n_iter_/dual_gap_ are scalars, predict is (n,).
        assert sk.coef_.shape == (2,)
        assert np.shape(sk.intercept_) == (1,)
        assert np.isscalar(sk.n_iter_) and np.ndim(sk.dual_gap_) == 0
        assert sk.predict(_X_MO).shape == (5,)

        assert fr.coef_.shape == sk.coef_.shape
        assert np.shape(fr.intercept_) == np.shape(sk.intercept_)
        assert np.isscalar(fr.n_iter_) and np.ndim(fr.dual_gap_) == 0
        assert fr.predict(_X_MO).shape == sk.predict(_X_MO).shape
        np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-7)
        np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=1e-7)
        np.testing.assert_allclose(fr.predict(_X_MO), sk.predict(_X_MO), atol=1e-7)
        assert fr.n_iter_ == sk.n_iter_
        assert abs(float(fr.dual_gap_) - float(sk.dual_gap_)) < 1e-9


def test_lasso_elasticnet_multioutput_no_intercept_scalar_zero():
    """`ferrolearn.Lasso`/`ElasticNet(fit_intercept=False)` on 2-D `y` exposes
    `intercept_` as the scalar `0.0` (ndim 0), matching sklearn EXACTLY.

    sklearn sets `intercept_` to the scalar `0.0` whenever `fit_intercept=False`,
    regardless of target dimensionality (`sklearn/linear_model/_base.py:327`).
    Oracle (R-CHAR-3): the live sklearn call is ground truth (scalar 0.0).
    """
    for fr_cls, sk_cls, kw in (
        (fl.Lasso, SkLasso, {"alpha": 0.5}),
        (fl.ElasticNet, SkElasticNet, {"alpha": 0.5, "l1_ratio": 0.5}),
    ):
        sk = sk_cls(fit_intercept=False, **kw).fit(_X_MO, _Y_MO)
        fr = fr_cls(fit_intercept=False, **kw).fit(_X_MO, _Y_MO)

        # Live-oracle ground truth (R-CHAR-3): sklearn intercept_ is scalar 0.0.
        assert np.ndim(sk.intercept_) == 0
        assert float(sk.intercept_) == 0.0

        assert np.ndim(fr.intercept_) == np.ndim(sk.intercept_)
        assert float(fr.intercept_) == 0.0

        # coef_ still (n_targets, n_features) and matches the oracle.
        assert fr.coef_.shape == sk.coef_.shape == (2, 2)
        np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-7)


def test_lasso_elasticnet_singleoutput_unchanged_by_multioutput_path():
    """Single-output `ferrolearn.Lasso`/`ElasticNet` (1-D `y`) is UNCHANGED: 1-D
    `coef_`, scalar `intercept_`, SCALAR `n_iter_` (int, #2043) and SCALAR
    `dual_gap_` (float, #2096), 1-D `predict` — matching sklearn (R-CHAR-3).

    Guards that the multi-output routing did not regress the 1-D path.
    """
    y = _Y_MO[:, 0]  # shape (5,)

    for fr_cls, sk_cls, kw in (
        (fl.Lasso, SkLasso, {"alpha": 0.5}),
        (fl.ElasticNet, SkElasticNet, {"alpha": 0.5, "l1_ratio": 0.5}),
    ):
        sk = sk_cls(**kw).fit(_X_MO, y)
        fr = fr_cls(**kw).fit(_X_MO, y)

        assert fr.coef_.ndim == 1
        assert fr.coef_.shape == sk.coef_.shape == (2,)
        np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-7)

        # intercept_ scalar float (not array), matching the 1-D contract.
        assert np.isscalar(fr.intercept_) or np.ndim(fr.intercept_) == 0
        assert abs(float(fr.intercept_) - float(sk.intercept_)) < 1e-7

        # n_iter_ is a scalar int (#2043), dual_gap_ a scalar float (#2096) —
        # NOT a list/array as on the multi-output path.
        assert np.isscalar(fr.n_iter_) and not isinstance(fr.n_iter_, list)
        assert fr.n_iter_ == sk.n_iter_
        assert np.isscalar(fr.dual_gap_) and np.ndim(fr.dual_gap_) == 0
        assert abs(float(fr.dual_gap_) - float(sk.dual_gap_)) < 1e-9

        fr_pred = fr.predict(_X_MO[:2])
        assert fr_pred.ndim == 1
        np.testing.assert_allclose(fr_pred, sk.predict(_X_MO[:2]), atol=1e-7)


def test_lasso_elasticnet_multioutput_pickle_predict_shape():
    """A pickled-then-unpickled multi-output `ferrolearn.Lasso`/`ElasticNet`
    round-trips `predict` identically (no shape error), since multi-output
    `predict` uses `coef_`/`intercept_` (not the popped `_rs_list`).

    sklearn's pickle round-trip leaves `predict` unchanged. Oracle (R-CHAR-3):
    sklearn's live unpickled predict is the ground truth.
    """
    import pickle

    for fr_cls, kw in (
        (fl.Lasso, {"alpha": 0.5}),
        (fl.ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5}),
    ):
        fr = fr_cls(**kw).fit(_X_MO, _Y_MO)
        fr_un = pickle.loads(pickle.dumps(fr))
        np.testing.assert_allclose(
            fr_un.predict(_X_MO), fr.predict(_X_MO), atol=1e-12
        )
        assert fr_un.predict(_X_MO).shape == (5, 2)


# Fixtures for the ConvergenceWarning divergence (#2127). `_X_CONV`/`_y_CONV`
# do NOT converge in a single CD sweep (sklearn n_iter_ == 1 at max_iter=1), so
# both ferrolearn and sklearn raise exactly one ConvergenceWarning; at the
# default max_iter=1000 the CD converges (n_iter_ << 1000) and NEITHER warns.
_X_CONV = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 5.0]])
_y_CONV = np.array([3.0, 2.5, 7.1, 6.0, 11.2])
# Two targets on DIFFERENT scales so the per-target CD warning messages are
# byte-distinct on BOTH sides — sklearn emits genuinely 2 warnings (no
# message-identical dedup), pinning ferrolearn's per-target count to 2.
_Y_CONV_MO = np.array(
    [[3.0, 100.0], [2.5, 200.0], [7.1, 300.0], [6.0, 150.0], [11.2, 80.0]]
)


def _capture_convergence_warnings(fit_callable):
    """Run `fit_callable()` and return the list of ConvergenceWarning messages.

    Uses `simplefilter('always')` so the count reflects every warning the
    estimator actually emitted (subject only to Python's byte-identical-message
    collapse, which sklearn's own per-target loop is equally subject to).
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_callable()
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]


def test_lasso_elasticnet_convergence_warning_at_max_iter():
    """`ferrolearn.Lasso`/`ElasticNet` emit a `ConvergenceWarning` when the
    coordinate-descent loop hits `max_iter` without converging — matching
    sklearn 1.5.2 EXACTLY (#2127).

    sklearn's CD warns once per non-converged fit from `_cd_fast.pyx:256-269`
    (the `for/else` that runs only when the loop never `break`s on the duality
    gap), iff `n_iter_ >= max_iter`. ferrolearn surfaces the real `n_iter_`
    (#2043/#2096), so the wrapper warns on the same condition with sklearn's
    exact wording (British "regularisation"). Was: ferrolearn emitted NO warning.

    Oracle (R-CHAR-3): the SAME fit on the LIVE sklearn estimator is asserted to
    warn (max_iter=1) / not warn (max_iter=1000) in this test — the behavior is
    pinned to sklearn, never copied from ferrolearn.
    """
    for fr_cls, sk_cls, kw in (
        (fl.Lasso, SkLasso, {"alpha": 0.5}),
        (fl.ElasticNet, SkElasticNet, {"alpha": 0.5}),
    ):
        # max_iter=1 does NOT converge: exactly one ConvergenceWarning, and the
        # live sklearn estimator warns on the identical fit (R-CHAR-3).
        fr_msgs = _capture_convergence_warnings(
            lambda c=fr_cls, k=kw: c(max_iter=1, **k).fit(_X_CONV, _y_CONV)
        )
        sk_msgs = _capture_convergence_warnings(
            lambda c=sk_cls, k=kw: c(max_iter=1, **k).fit(_X_CONV, _y_CONV)
        )
        assert len(sk_msgs) == 1, f"{sk_cls.__name__} sklearn baseline"
        assert len(fr_msgs) == 1, f"{fr_cls.__name__} should warn once"
        assert fr_msgs[0].startswith("Objective did not converge")
        assert sk_msgs[0].startswith("Objective did not converge")

        # max_iter=1000 converges: NEITHER ferrolearn NOR sklearn warns.
        fr_conv = _capture_convergence_warnings(
            lambda c=fr_cls, k=kw: c(max_iter=1000, **k).fit(_X_CONV, _y_CONV)
        )
        sk_conv = _capture_convergence_warnings(
            lambda c=sk_cls, k=kw: c(max_iter=1000, **k).fit(_X_CONV, _y_CONV)
        )
        assert len(sk_conv) == 0, f"{sk_cls.__name__} converges, no warning"
        assert len(fr_conv) == 0, f"{fr_cls.__name__} converges, no warning"


def test_lasso_convergence_warning_multioutput_per_target():
    """Multi-output `ferrolearn.Lasso` emits one `ConvergenceWarning` per
    non-converged target column, matching sklearn's per-target CD loop count
    (#2127).

    sklearn fits each target independently (`_coordinate_descent.py:1072-1103`),
    each `self.path` call warning from `_cd_fast.pyx:269` when that target hits
    `max_iter`. The wrapper's per-target loop warns on each target's real
    `n_iter_j >= max_iter`. On `_Y_CONV_MO` the two targets are on different
    scales so both warning messages are byte-distinct — sklearn genuinely emits
    2, pinning ferrolearn to 2 (R-CHAR-3).
    """
    sk_msgs = _capture_convergence_warnings(
        lambda: SkLasso(alpha=0.5, max_iter=1).fit(_X_CONV, _Y_CONV_MO)
    )
    fr_msgs = _capture_convergence_warnings(
        lambda: fl.Lasso(alpha=0.5, max_iter=1).fit(_X_CONV, _Y_CONV_MO)
    )
    assert len(sk_msgs) == 2, "sklearn baseline: one warning per target"
    assert len(fr_msgs) == 2, "ferrolearn: one warning per non-converged target"
    assert all(m.startswith("Objective did not converge") for m in fr_msgs)


def _parse_gap(msg):
    """Parse the `Duality gap: <g>` float out of a ConvergenceWarning message."""
    import re

    m = re.search(r"Duality gap:\s*([0-9.eE+-]+)", msg)
    return float(m.group(1)) if m else None


def test_red_lasso_elasticnet_convergence_warning_gap_value_matches_sklearn():
    """Divergence: `ferrolearn.Lasso`/`ElasticNet`'s ConvergenceWarning message
    prints the WRONG duality-gap number.

    sklearn's coordinate-descent warning is emitted from INSIDE `cd_fast`
    (`sklearn/linear_model/_cd_fast.pyx:256-260`):
        ``f"... Duality gap: {gap:.3e}, tolerance: {tol:.3e}"``
    where `gap` is the RAW cd_fast objective gap (the objective in cd_fast is
    `n_samples` times the user-facing objective, per the comment at
    `sklearn/linear_model/_coordinate_descent.py:707-709`
    ``dual_gaps[i] = dual_gap_ / n_samples``). The fitted attribute
    `self.dual_gap_` is therefore the cd_fast gap DIVIDED by `n_samples`, while
    the WARNING MESSAGE shows the UNDIVIDED gap (`n_samples` times larger).

    ferrolearn's `_regressors.py::_warn_convergence(self.dual_gap_, tol_scaled)`
    (`ferrolearn-python/python/ferrolearn/_regressors.py:330`) passes the
    rescaled fitted attribute `self.dual_gap_` (== cd_fast gap / n_samples) into
    the message, so its printed gap is `n_samples`x too small relative to
    sklearn's. On the 5-sample `_X_CONV`/`_y_CONV` fixture at `max_iter=1`
    sklearn prints `7.071e+00` (Lasso) while ferrolearn prints `1.414e+00`.

    The STORED `dual_gap_` attribute matches sklearn exactly (that is NOT the
    divergence); only the human-readable warning text diverges. Note the root:
    the binding's `_warn_convergence` call site, NOT the Rust crate.

    Oracle (R-CHAR-3): the expected gap is parsed from the LIVE sklearn 1.5.2
    warning message in this same test, never copied from the ferrolearn side.

    Tracking: #2128 (release-blocker, left un-ignored per goal.md R-DEFER-3).
    """
    for fr_cls, sk_cls, kw in (
        (fl.Lasso, SkLasso, {"alpha": 0.5}),
        (fl.ElasticNet, SkElasticNet, {"alpha": 0.5}),
    ):
        fr_msgs = _capture_convergence_warnings(
            lambda c=fr_cls, k=kw: c(max_iter=1, **k).fit(_X_CONV, _y_CONV)
        )
        sk_msgs = _capture_convergence_warnings(
            lambda c=sk_cls, k=kw: c(max_iter=1, **k).fit(_X_CONV, _y_CONV)
        )
        assert len(sk_msgs) == 1 and len(fr_msgs) == 1

        sk_gap = _parse_gap(sk_msgs[0])
        fr_gap = _parse_gap(fr_msgs[0])
        assert sk_gap is not None and fr_gap is not None

        # ferrolearn's printed gap must match sklearn's printed gap (the RAW
        # cd_fast gap), to 0.1% relative. Currently ferrolearn prints
        # dual_gap_ == sk_gap / n_samples, so this FAILS by a factor of
        # n_samples (== 5 on this fixture).
        assert abs(fr_gap - sk_gap) <= 1e-3 * abs(sk_gap), (
            f"{fr_cls.__name__}: ferrolearn warning gap {fr_gap:.3e} != "
            f"sklearn warning gap {sk_gap:.3e} (factor "
            f"{sk_gap / fr_gap:.2f}, == n_samples)"
        )


# ---------------------------------------------------------------------------
# positive= constructor parameter (#2129). sklearn's `positive=True` constrains
# `coef_ >= 0` for all four estimators (LinearRegression NNLS `_base.py:645-653`;
# Ridge projected solve `_ridge.py:923-928`; Lasso/ElasticNet non-negative
# soft-threshold `_cd_fast.pyx:191-195`). `positive` is keyword-only and LAST in
# every sklearn `__init__` (`_base.py:574`, `_ridge.py:902`,
# `_coordinate_descent.py:909`). The Rust crates already support + oracle-verify
# it; this surfaces it through the binding ctors + the Python wrappers.
#
# Fixture: an unconstrained fit of this (X, y) has a NEGATIVE 2nd coefficient for
# LinearRegression/Ridge/ElasticNet, so `positive=True` clamps it to 0.0 and the
# whole solution differs — the constraint is genuinely ACTIVE (not a no-op).
# Every expected value is the LIVE sklearn 1.5.2 oracle computed in the same test
# (R-CHAR-3); none is literal-copied from ferrolearn.
# ---------------------------------------------------------------------------

_X_POS = np.array(
    [[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0], [6.0, 2.0]]
)
_y_POS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.5])


def test_positive_param_keyword_only_last():
    """`ferrolearn.{LinearRegression,Ridge,Lasso,ElasticNet}` expose `positive`
    as a KEYWORD-ONLY constructor param, matching sklearn's signature placement
    (`_base.py:574`, `_ridge.py:902`, `_coordinate_descent.py:909` — `positive`
    after the `*`). Oracle (R-CHAR-3): sklearn's live signature kind.
    """
    pairs = (
        (fl.LinearRegression, SkLinearRegression),
        (fl.Ridge, SkRidge),
        (fl.Lasso, SkLasso),
        (fl.ElasticNet, SkElasticNet),
    )
    for fr_cls, sk_cls in pairs:
        sk_kind = inspect.signature(sk_cls.__init__).parameters["positive"].kind
        assert sk_kind == inspect.Parameter.KEYWORD_ONLY, sk_cls.__name__
        fr_params = inspect.signature(fr_cls.__init__).parameters
        assert "positive" in fr_params, f"{fr_cls.__name__} missing positive"
        assert fr_params["positive"].kind == inspect.Parameter.KEYWORD_ONLY
        # Default is False on both sides.
        assert sk_cls().positive is False
        assert fr_cls().positive is False


def test_linearregression_positive_matches_sklearn():
    """`ferrolearn.LinearRegression(positive=True)` constrains `coef_ >= 0` and
    matches the live sklearn NNLS oracle element-wise (`_base.py:645-647`).
    Unconstrained 2nd coef is negative here, so the constraint is active.
    """
    sk = SkLinearRegression(positive=True).fit(_X_POS, _y_POS)
    fr = fl.LinearRegression(positive=True).fit(_X_POS, _y_POS)

    assert np.all(np.asarray(fr.coef_) >= 0.0)
    assert np.any(np.asarray(SkLinearRegression().fit(_X_POS, _y_POS).coef_) < 0.0)
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-6)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-6
    np.testing.assert_allclose(fr.predict(_X_POS), sk.predict(_X_POS), atol=1e-6)


def test_ridge_positive_matches_sklearn():
    """`ferrolearn.Ridge(alpha=1.0, positive=True)` constrains `coef_ >= 0` and
    matches the live sklearn projected oracle (`_ridge.py:923-928`).
    """
    sk = SkRidge(alpha=1.0, positive=True).fit(_X_POS, _y_POS)
    fr = fl.Ridge(alpha=1.0, positive=True).fit(_X_POS, _y_POS)

    assert np.all(np.asarray(fr.coef_) >= 0.0)
    assert np.any(np.asarray(SkRidge(alpha=1.0).fit(_X_POS, _y_POS).coef_) < 0.0)
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-6)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-6
    np.testing.assert_allclose(fr.predict(_X_POS), sk.predict(_X_POS), atol=1e-6)


def test_lasso_positive_matches_sklearn():
    """`ferrolearn.Lasso(alpha=0.3, positive=True)` constrains `coef_ >= 0` and
    matches the live sklearn non-negative-soft-threshold oracle
    (`_cd_fast.pyx:191-195`).
    """
    sk = SkLasso(alpha=0.3, positive=True).fit(_X_POS, _y_POS)
    fr = fl.Lasso(alpha=0.3, positive=True).fit(_X_POS, _y_POS)

    assert np.all(np.asarray(fr.coef_) >= 0.0)
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-6)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-6
    np.testing.assert_allclose(fr.predict(_X_POS), sk.predict(_X_POS), atol=1e-6)


def test_elasticnet_positive_matches_sklearn():
    """`ferrolearn.ElasticNet(alpha=0.3, positive=True)` constrains `coef_ >= 0`
    and matches the live sklearn oracle (`_cd_fast.pyx:191-195`). Unconstrained
    2nd coef is negative here, so the constraint is active.
    """
    sk = SkElasticNet(alpha=0.3, positive=True).fit(_X_POS, _y_POS)
    fr = fl.ElasticNet(alpha=0.3, positive=True).fit(_X_POS, _y_POS)

    assert np.all(np.asarray(fr.coef_) >= 0.0)
    assert np.any(
        np.asarray(SkElasticNet(alpha=0.3).fit(_X_POS, _y_POS).coef_) < 0.0
    )
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=1e-6)
    assert abs(float(fr.intercept_) - float(sk.intercept_)) <= 1e-6
    np.testing.assert_allclose(fr.predict(_X_POS), sk.predict(_X_POS), atol=1e-6)


def test_all_four_positive_false_unchanged():
    """`positive=False` (the DEFAULT) is byte-identical to the unconstrained fit
    for all four estimators — the new param does not perturb the default path.
    Oracle (R-CHAR-3): the live sklearn default fit is the ground truth.
    """
    configs = (
        (fl.LinearRegression, SkLinearRegression, {}),
        (fl.Ridge, SkRidge, {"alpha": 1.0}),
        (fl.Lasso, SkLasso, {"alpha": 0.3}),
        (fl.ElasticNet, SkElasticNet, {"alpha": 0.3}),
    )
    for fr_cls, sk_cls, kw in configs:
        sk = sk_cls(**kw).fit(_X_POS, _y_POS)
        fr_default = fr_cls(**kw).fit(_X_POS, _y_POS)
        fr_explicit = fr_cls(positive=False, **kw).fit(_X_POS, _y_POS)

        # explicit positive=False == default == sklearn default (element-wise).
        np.testing.assert_allclose(
            fr_explicit.coef_, fr_default.coef_, atol=0.0
        )
        np.testing.assert_allclose(fr_default.coef_, sk.coef_, atol=1e-6)
        assert abs(float(fr_default.intercept_) - float(sk.intercept_)) <= 1e-6


def test_ridge_positive_multioutput_matches_sklearn():
    """Multi-output `ferrolearn.Ridge(positive=True)` on a 2-D `Y` constrains each
    target's `coef_ >= 0`, matching sklearn EXACTLY.

    sklearn's positive Ridge solve is per-target-independent (`_ridge.py:923-928`
    dispatch; per-target solve `:207`), so multi `coef_` `(n_targets, n_features)`
    equals the stack of per-target single-output positive fits. The wrapper routes
    the `positive=True` + 2-D `y` path through a per-target loop over the
    single-output positive `_RsRidge` (the `_RsRidgeMultiOutput` binding has no
    positivity support). Oracle (R-CHAR-3): the live sklearn 1.5.2 call.

    Value tolerance: where a target's positivity constraint binds at a vertex
    (a coef clamped to 0) the agreement is bit-tight; where the optimum is
    interior, ferrolearn's downstream projected coordinate descent
    (`ferrolearn-linear` Ridge REQ-9 #387) and sklearn's L-BFGS-B
    (`_ridge.py:300`) stop at slightly different iterates for a fixed
    `(max_iter, tol)` — a downstream stopping-criterion gap, NOT a binding bug —
    so the constrained values agree to ~2e-3, not ULP (cf. the Lasso/ElasticNet
    CD-stopping caveat). The STRUCTURE (coef_ >= 0, shape, per-target-stack
    independence) is exact; the binding is a faithful per-target router.
    """
    Y = np.array(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 2.0], [5.5, 6.0]]
    )
    sk = SkRidge(alpha=1.0, positive=True).fit(_X_POS, Y)
    fr = fl.Ridge(alpha=1.0, positive=True).fit(_X_POS, Y)

    assert fr.coef_.shape == sk.coef_.shape == (2, 2)
    assert np.all(fr.coef_ >= 0.0)
    # Target 0 (constraint binds at a vertex) matches bit-tight; target 1
    # (interior optimum) agrees to the downstream solver-stopping tolerance.
    np.testing.assert_allclose(fr.coef_, sk.coef_, atol=2e-3)

    assert fr.intercept_.shape == sk.intercept_.shape == (2,)
    np.testing.assert_allclose(fr.intercept_, sk.intercept_, atol=2e-3)

    np.testing.assert_allclose(fr.predict(_X_POS), sk.predict(_X_POS), atol=2e-3)

    # Independence (EXACT): multi coef_ == stack of per-target single-output
    # positive fits — proves the wrapper routes per-target, not coupled.
    stacked = np.vstack(
        [fl.Ridge(alpha=1.0, positive=True).fit(_X_POS, Y[:, j]).coef_ for j in range(2)]
    )
    np.testing.assert_allclose(fr.coef_, stacked, atol=1e-12)
