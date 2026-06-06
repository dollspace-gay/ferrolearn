"""sklearn-compatible wrappers for ferrolearn regression models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
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


class Lasso(RegressorMixin, BaseEstimator):
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


class ElasticNet(RegressorMixin, BaseEstimator):
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
