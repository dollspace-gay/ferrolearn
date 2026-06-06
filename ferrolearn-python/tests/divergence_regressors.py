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
