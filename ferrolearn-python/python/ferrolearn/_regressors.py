"""sklearn-compatible wrappers for ferrolearn regression models."""

import re
import warnings

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import (
    _RsElasticNet,
    _RsLasso,
    _RsLinearRegression,
    _RsLinearRegressionMultiOutput,
    _RsRidge,
    _RsRidgeMultiOutput,
)


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _predict_linear(X, coef, intercept):
    """Fallback linear prediction using stored coefficients.

    Mirrors sklearn's `_decision_function` `X @ coef_.T + intercept_`
    (`sklearn/linear_model/_base.py:290`/`:364`). `coef.T` is a no-op for a 1-D
    `coef` (single-output, shape `(n_features,)`), so `X @ coef.T == X @ coef`
    stays `(n,)`; for a 2-D multi-output `coef` `(n_targets, n_features)` it
    yields the correct `(n, n_targets)` orientation (#2125).
    """
    coef = np.asarray(coef)
    return X @ coef.T + intercept


def _warn_convergence(gap, tol_scaled):
    """Emit sklearn's coordinate-descent non-convergence warning.

    Matches sklearn 1.5.2's exact wording from
    `sklearn/linear_model/_cd_fast.pyx:256-260` (note the British spelling
    "regularisation"), raised once per non-converged fit when the CD loop hits
    `max_iter` without the duality gap dropping below the scaled tolerance.
    """
    warnings.warn(
        f"Objective did not converge. You might want to increase the number "
        f"of iterations, check the scale of the features or consider "
        f"increasing regularisation. Duality gap: {gap:.3e}, tolerance: "
        f"{tol_scaled:.3e}",
        ConvergenceWarning,
    )


def _fit_rust(rs, X, y=None):
    """Call rs.fit() and translate Rust errors to sklearn-conforming messages."""
    try:
        if y is not None:
            rs.fit(X, y)
        else:
            rs.fit(X)
    except ValueError as e:
        msg = str(e)
        # Translate "Insufficient samples: need at least N, got M" to sklearn format
        m = re.search(r"got (\d+)", msg)
        if m and "Insufficient" in msg:
            n = m.group(1)
            raise ValueError(
                f"n_samples={n} is not enough; this estimator needs at least "
                f"as many samples as features. {msg}"
            ) from e
        raise


class LinearRegression(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Ordinary Least Squares regression backed by Rust.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Detect multi-output BEFORE validation: ANY 2-D `y` routes to the Rust
        # `Fit<Array2, Array2>` path producing `coef_` (n_targets, n_features) /
        # `intercept_` (n_targets,), mirroring sklearn `LinearRegression`
        # (`sklearn/linear_model/_base.py:687` `coef_.T`). sklearn PRESERVES a
        # 2-D target shape (incl. a single-column `(n, 1)` `y` -> `coef_` `(1,
        # n_features)`, `intercept_` `(1,)`), so the gate is on `y.ndim == 2`
        # (NOT `y.shape[1] > 1`) and there is no DataConversionWarning
        # (the `MultiOutputMixin` tag tells `check_supervised_y_2d` so).
        y_arr = np.asarray(y)
        multioutput = y_arr.ndim == 2
        if multioutput:
            X, y = self._validate_data(
                X, y, dtype="float64", multi_output=True, y_numeric=True
            )
            X, y = _ensure_f64(X), _ensure_f64(y)
            self._rs = _RsLinearRegressionMultiOutput(fit_intercept=self.fit_intercept)
            _fit_rust(self._rs, X, y)
            self.coef_ = np.array(self._rs.coef_)
            # sklearn `_set_intercept` sets a scalar `0.0` when fit_intercept is
            # False, regardless of target count (`_base.py:327`).
            self.intercept_ = (
                np.array(self._rs.intercept_) if self.fit_intercept else 0.0
            )
            # sklearn sets `rank_`/`singular_` from `linalg.lstsq(X, y)`
            # REGARDLESS of single vs multi-output (`_base.py:687`), so the
            # multi-output path exposes them too (#2124).
            self.rank_ = int(self._rs.rank_)
            self.singular_ = np.array(self._rs.singular_)
        else:
            X, y = self._validate_data(X, y, dtype="float64", y_numeric=True)
            X, y = _ensure_f64(X), _ensure_f64(y)
            self._rs = _RsLinearRegression(fit_intercept=self.fit_intercept)
            _fit_rust(self._rs, X, y)
            self.coef_ = np.array(self._rs.coef_)
            self.intercept_ = float(self._rs.intercept_)
            # rank_/singular_ are the single-output (1-D) lstsq diagnostics
            # (#2101). The multi-output branch sets the same attrs from its own
            # lstsq solve (#2124), so BOTH paths expose them like sklearn
            # (`_base.py:687`).
            self.rank_ = int(self._rs.rank_)
            self.singular_ = np.array(self._rs.singular_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Ridge(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Ridge regression backed by Rust.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, alpha=1.0, *, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Detect multi-output BEFORE validation: a 2-D `y` with >1 column routes
        # to the Rust `Fit<Array2, Array2>` path producing `coef_`
        # (n_targets, n_features) / `intercept_` (n_targets,), mirroring sklearn
        # `Ridge` (`sklearn/linear_model/_ridge.py:543`/`:550`). sklearn PRESERVES
        # a 2-D target shape (incl. a single-column `(n, 1)` `y` -> `coef_` `(1,
        # n_features)`, `intercept_` `(1,)`), so ANY 2-D `y` routes to the
        # multi-output path (`_base.py:319-324`).
        y_arr = np.asarray(y)
        multioutput = y_arr.ndim == 2
        if multioutput:
            X, y = self._validate_data(
                X, y, dtype="float64", multi_output=True, y_numeric=True
            )
            X, y = _ensure_f64(X), _ensure_f64(y)
            self._rs = _RsRidgeMultiOutput(
                alpha=self.alpha, fit_intercept=self.fit_intercept
            )
            _fit_rust(self._rs, X, y)
            self.coef_ = np.array(self._rs.coef_)
            # sklearn `_set_intercept` sets a scalar `0.0` when fit_intercept is
            # False, regardless of target count (`_base.py:327`).
            self.intercept_ = (
                np.array(self._rs.intercept_) if self.fit_intercept else 0.0
            )
        else:
            X, y = self._validate_data(X, y, dtype="float64", y_numeric=True)
            X, y = _ensure_f64(X), _ensure_f64(y)
            self._rs = _RsRidge(alpha=self.alpha, fit_intercept=self.fit_intercept)
            _fit_rust(self._rs, X, y)
            self.coef_ = np.array(self._rs.coef_)
            self.intercept_ = float(self._rs.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class Lasso(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Lasso regression backed by Rust.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, alpha=1.0, *, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Detect multi-output BEFORE validation. sklearn `Lasso` (subclass of
        # `ElasticNet(MultiOutputMixin, ...)`) fits each target column
        # INDEPENDENTLY via a per-target coordinate-descent loop
        # (`sklearn/linear_model/_coordinate_descent.py:1071-1101`), producing
        # `coef_` `(n_targets, n_features)`, `intercept_` `(n_targets,)`,
        # `n_iter_` a PER-TARGET LIST, and `dual_gap_` `(n_targets,)`
        # (`:1108-1111`). A 2-D `y` — including a single-column `(n, 1)` — routes
        # to this path (`:1054` keeps `y` 2-D); sklearn verified: multi `coef_`
        # equals the stack of per-target single-output fits (NOT MultiTaskLasso).
        # The `MultiOutputMixin` tag tells `check_supervised_y_2d` so there is no
        # DataConversionWarning (#2126).
        #
        # n_targets == 1 COLLAPSE (differs from Ridge/LinearRegression): the CD
        # path collapses a single-target fit — INCLUDING a `(n, 1)` `y` — back to
        # 1-D `coef_`, scalar `n_iter_`, scalar `dual_gap_`, and `(n,)` `predict`
        # (`_coordinate_descent.py:1106-1108` `if n_targets == 1`). Only
        # `intercept_` stays a length-1 array `(1,)` from `_set_intercept`
        # (`_base.py:319` `y_offset` is `(1,)`). Verified live on the sklearn
        # 1.5.2 oracle (#2126).
        y_arr = np.asarray(y)
        if y_arr.ndim == 2:
            X, y = self._validate_data(
                X, y, dtype="float64", multi_output=True, y_numeric=True
            )
            X, y = _ensure_f64(X), _ensure_f64(y)
            coef_list = []
            intercept_list = []
            n_iter_list = []
            dual_gap_list = []
            self._rs_list = []
            for j in range(y.shape[1]):
                rs_j = _RsLasso(
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                )
                _fit_rust(rs_j, X, _ensure_f64(y[:, j]))
                coef_list.append(np.array(rs_j.coef_))
                intercept_list.append(float(rs_j.intercept_))
                n_iter_j = int(rs_j.n_iter_)
                dual_gap_j = float(rs_j.dual_gap_)
                n_iter_list.append(n_iter_j)
                dual_gap_list.append(dual_gap_j)
                self._rs_list.append(rs_j)
                # Per-target ConvergenceWarning: sklearn's CD loop warns once per
                # non-converged target (`_cd_fast.pyx:256-269`), driven here by the
                # real per-target `n_iter_` hitting `max_iter` (#2127).
                if n_iter_j >= self.max_iter:
                    yc_j = y[:, j] - y[:, j].mean() if self.fit_intercept else y[:, j]
                    tol_scaled_j = self.tol * float(np.dot(yc_j, yc_j))
                    # sklearn's message prints the RAW cd_fast gap, whereas the
                    # stored `dual_gap_` is `raw / n_samples`
                    # (`_coordinate_descent.py:709`), so re-scale for the message.
                    _warn_convergence(dual_gap_j * X.shape[0], tol_scaled_j)
            n_targets = y.shape[1]
            if n_targets == 1:
                # Collapse coef_/n_iter_/dual_gap_ for the single target
                # (`_coordinate_descent.py:1106-1108`), but `intercept_` keeps its
                # `(1,)` shape (or scalar 0.0 when fit_intercept is False).
                self.coef_ = coef_list[0]
                self.n_iter_ = n_iter_list[0]
                self.dual_gap_ = dual_gap_list[0]
            else:
                self.coef_ = np.vstack(coef_list)
                # sklearn's `n_iter_` is the per-target Python list (`:1101`).
                self.n_iter_ = n_iter_list
                self.dual_gap_ = np.array(dual_gap_list)
            # sklearn `_set_intercept` sets a scalar `0.0` when fit_intercept is
            # False, regardless of target count (`_base.py:327`).
            self.intercept_ = (
                np.array(intercept_list) if self.fit_intercept else 0.0
            )
            return self
        X, y = self._validate_data(X, y, dtype="float64", y_numeric=True)
        X, y = _ensure_f64(X), _ensure_f64(y)
        self._rs = _RsLasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        _fit_rust(self._rs, X, y)
        self.coef_ = np.array(self._rs.coef_)
        self.intercept_ = float(self._rs.intercept_)
        self.n_iter_ = int(self._rs.n_iter_)
        self.dual_gap_ = float(self._rs.dual_gap_)
        # ConvergenceWarning when the CD loop hit `max_iter` without the duality
        # gap reaching the scaled tolerance, matching sklearn (`_cd_fast.pyx:
        # 256-269`). `tol_scaled = tol * ||yc||^2` with `yc` the centered target
        # the CD operates on (`y - y.mean()` when `fit_intercept`, else `y`),
        # mirroring `_coordinate_descent.py` `tol *= np.dot(y, y)` on the
        # already-centered target (#2127).
        if self.n_iter_ >= self.max_iter:
            yc = y - y.mean() if self.fit_intercept else y
            tol_scaled = self.tol * float(np.dot(yc, yc))
            # sklearn's message prints the RAW cd_fast gap; the stored `dual_gap_`
            # is `raw / n_samples` (`_coordinate_descent.py:709`), so re-scale.
            _warn_convergence(self.dual_gap_ * X.shape[0], tol_scaled)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        # Multi-output: no single `_rs`; use the stored coefficients so the
        # result is `(n_samples, n_targets)` (`X @ coef_.T + intercept_`) and
        # pickle round-trips work without rebuilding the Rust estimators.
        if np.asarray(self.coef_).ndim == 2:
            return _predict_linear(X, self.coef_, self.intercept_)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        state.pop("_rs_list", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class ElasticNet(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """ElasticNet regression backed by Rust.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    l1_ratio : float, default=0.5
        Mix of L1 vs L2 penalty (0=Ridge, 1=Lasso).
    max_iter : int, default=1000
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(
        self, alpha=1.0, *, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True
    ):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Detect multi-output BEFORE validation. sklearn `ElasticNet`
        # (`class ElasticNet(MultiOutputMixin, RegressorMixin, LinearModel)`)
        # fits each target column INDEPENDENTLY via a per-target
        # coordinate-descent loop
        # (`sklearn/linear_model/_coordinate_descent.py:1071-1101`), producing
        # `coef_` `(n_targets, n_features)`, `intercept_` `(n_targets,)`,
        # `n_iter_` a PER-TARGET LIST, and `dual_gap_` `(n_targets,)`
        # (`:1108-1111`). A 2-D `y` — including a single-column `(n, 1)` — routes
        # here; sklearn verified: multi `coef_` equals the stack of per-target
        # single-output fits (NOT MultiTaskElasticNet). The `MultiOutputMixin`
        # tag means no DataConversionWarning (#2126).
        #
        # n_targets == 1 COLLAPSE (differs from Ridge/LinearRegression): the CD
        # path collapses a single-target fit — INCLUDING a `(n, 1)` `y` — back to
        # 1-D `coef_`, scalar `n_iter_`, scalar `dual_gap_`, and `(n,)` `predict`
        # (`_coordinate_descent.py:1106-1108` `if n_targets == 1`). Only
        # `intercept_` stays a length-1 array `(1,)` from `_set_intercept`
        # (`_base.py:319`). Verified live on the sklearn 1.5.2 oracle (#2126).
        y_arr = np.asarray(y)
        if y_arr.ndim == 2:
            X, y = self._validate_data(
                X, y, dtype="float64", multi_output=True, y_numeric=True
            )
            X, y = _ensure_f64(X), _ensure_f64(y)
            coef_list = []
            intercept_list = []
            n_iter_list = []
            dual_gap_list = []
            self._rs_list = []
            for j in range(y.shape[1]):
                rs_j = _RsElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                )
                _fit_rust(rs_j, X, _ensure_f64(y[:, j]))
                coef_list.append(np.array(rs_j.coef_))
                intercept_list.append(float(rs_j.intercept_))
                n_iter_j = int(rs_j.n_iter_)
                dual_gap_j = float(rs_j.dual_gap_)
                n_iter_list.append(n_iter_j)
                dual_gap_list.append(dual_gap_j)
                self._rs_list.append(rs_j)
                # Per-target ConvergenceWarning: sklearn's CD loop warns once per
                # non-converged target (`_cd_fast.pyx:256-269`), driven here by the
                # real per-target `n_iter_` hitting `max_iter` (#2127).
                if n_iter_j >= self.max_iter:
                    yc_j = y[:, j] - y[:, j].mean() if self.fit_intercept else y[:, j]
                    tol_scaled_j = self.tol * float(np.dot(yc_j, yc_j))
                    # sklearn's message prints the RAW cd_fast gap, whereas the
                    # stored `dual_gap_` is `raw / n_samples`
                    # (`_coordinate_descent.py:709`), so re-scale for the message.
                    _warn_convergence(dual_gap_j * X.shape[0], tol_scaled_j)
            n_targets = y.shape[1]
            if n_targets == 1:
                # Collapse coef_/n_iter_/dual_gap_ for the single target
                # (`_coordinate_descent.py:1106-1108`), but `intercept_` keeps its
                # `(1,)` shape (or scalar 0.0 when fit_intercept is False).
                self.coef_ = coef_list[0]
                self.n_iter_ = n_iter_list[0]
                self.dual_gap_ = dual_gap_list[0]
            else:
                self.coef_ = np.vstack(coef_list)
                # sklearn's `n_iter_` is the per-target Python list (`:1101`).
                self.n_iter_ = n_iter_list
                self.dual_gap_ = np.array(dual_gap_list)
            # sklearn `_set_intercept` sets a scalar `0.0` when fit_intercept is
            # False, regardless of target count (`_base.py:327`).
            self.intercept_ = (
                np.array(intercept_list) if self.fit_intercept else 0.0
            )
            return self
        X, y = self._validate_data(X, y, dtype="float64", y_numeric=True)
        X, y = _ensure_f64(X), _ensure_f64(y)
        self._rs = _RsElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        _fit_rust(self._rs, X, y)
        self.coef_ = np.array(self._rs.coef_)
        self.intercept_ = float(self._rs.intercept_)
        self.n_iter_ = int(self._rs.n_iter_)
        self.dual_gap_ = float(self._rs.dual_gap_)
        # ConvergenceWarning when the CD loop hit `max_iter` without the duality
        # gap reaching the scaled tolerance, matching sklearn (`_cd_fast.pyx:
        # 256-269`). `tol_scaled = tol * ||yc||^2` with `yc` the centered target
        # the CD operates on (`y - y.mean()` when `fit_intercept`, else `y`),
        # mirroring `_coordinate_descent.py` `tol *= np.dot(y, y)` on the
        # already-centered target (#2127).
        if self.n_iter_ >= self.max_iter:
            yc = y - y.mean() if self.fit_intercept else y
            tol_scaled = self.tol * float(np.dot(yc, yc))
            # sklearn's message prints the RAW cd_fast gap; the stored `dual_gap_`
            # is `raw / n_samples` (`_coordinate_descent.py:709`), so re-scale.
            _warn_convergence(self.dual_gap_ * X.shape[0], tol_scaled)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        # Multi-output: no single `_rs`; use the stored coefficients so the
        # result is `(n_samples, n_targets)` (`X @ coef_.T + intercept_`) and
        # pickle round-trips work without rebuilding the Rust estimators.
        if np.asarray(self.coef_).ndim == 2:
            return _predict_linear(X, self.coef_, self.intercept_)
        if hasattr(self, "_rs"):
            return np.array(self._rs.predict(X))
        return _predict_linear(X, self.coef_, self.intercept_)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        state.pop("_rs_list", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
