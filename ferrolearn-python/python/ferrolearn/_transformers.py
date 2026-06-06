"""sklearn-compatible wrappers for ferrolearn transformer models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import _RsPCA, _RsStandardScaler


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _fit_rust(rs, X, y=None):
    """Call rs.fit() and translate Rust errors to sklearn-conforming messages."""
    try:
        if y is not None:
            rs.fit(X, y)
        else:
            rs.fit(X)
    except ValueError as e:
        msg = str(e)
        m = re.search(r"got (\d+)", msg)
        if m and "Insufficient" in msg:
            n = m.group(1)
            raise ValueError(
                f"n_samples={n} is not enough; this estimator needs at least "
                f"as many samples as features. {msg}"
            ) from e
        raise


class StandardScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.

    Backed by Rust.
    """

    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        X = self._validate_data(X, dtype="float64", accept_sparse=False)
        self._rs = _RsStandardScaler(self.with_mean, self.with_std, self.copy)
        _fit_rust(self._rs, _ensure_f64(X))
        # sklearn's attribute-nulling rule (`sklearn/preprocessing/_data.py`):
        #   - `mean_`/`var_` -> None only when with_mean AND with_std are False
        #     (`_data.py:993-995`); otherwise the mean is materialized (it is
        #     used even when with_mean=False for the incremental variance).
        #   - `scale_`/`var_` -> None when with_std is False (`_data.py:1022-1023`).
        self.mean_ = (
            np.array(self._rs.mean_) if (self.with_mean or self.with_std) else None
        )
        self.scale_ = np.array(self._rs.scale_) if self.with_std else None
        # var_ is the TRUE population variance (ddof=0), which is 0.0 on a
        # constant column. It is NOT scale_**2: sklearn's _handle_zeros_in_scale
        # clamps scale_=1.0 on constant columns (so scale_**2=1.0 there), but
        # var_ is the raw variance computed BEFORE that clamp
        # (`sklearn/preprocessing/_data.py:1013-1023`). Read it from the binding,
        # which surfaces `FittedStandardScaler::var()`.
        self.var_ = np.array(self._rs.var_) if self.with_std else None
        self.n_samples_seen_ = X.shape[0]
        self._fit_X = X.copy()
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.transform(X))
        # Fallback using stored attributes
        result = X.copy()
        if self.with_mean:
            result = result - self.mean_
        if self.with_std:
            result = result / np.where(self.scale_ == 0, 1.0, self.scale_)
        return result

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.inverse_transform(X))
        result = X.copy()
        if self.with_std:
            result = result * self.scale_
        if self.with_mean:
            result = result + self.mean_
        return result

    def get_feature_names_out(self, input_features=None):
        # OneToOneFeatureMixin: a 1:1 transformer's output names == its input
        # names; default `x0..x{n_features_in_-1}` when none are provided
        # (sklearn/base.py _OneToOneFeatureMixin.get_feature_names_out).
        check_is_fitted(self)
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=object
            )
        input_features = np.asarray(input_features, dtype=object)
        if input_features.shape[0] != self.n_features_in_:
            raise ValueError(
                f"input_features should have length {self.n_features_in_}, "
                f"got {input_features.shape[0]}"
            )
        return input_features

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct _rs if we have training data
        if hasattr(self, "_fit_X"):
            self._rs = _RsStandardScaler(
                self.with_mean,
                self.with_std,
                getattr(self, "copy", True),
            )
            self._rs.fit(_ensure_f64(self._fit_X))


class PCA(TransformerMixin, BaseEstimator):
    """Principal Component Analysis backed by Rust.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If None, all
        ``min(n_samples, n_features)`` components are kept (resolved at fit),
        matching ``sklearn.decomposition.PCA`` (``sklearn/decomposition/_pca.py:409``).
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = self._validate_data(X, dtype="float64")
        # sklearn stores n_components=None verbatim and resolves it to
        # min(n_samples, n_features) at fit (`_pca.py:523-529`). The Rust
        # binding expects a usize, so resolve None here.
        n_components = (
            self.n_components if self.n_components is not None else min(X.shape)
        )
        self._rs = _RsPCA(n_components=n_components)
        _fit_rust(self._rs, _ensure_f64(X))
        self.components_ = np.array(self._rs.components_)
        self.explained_variance_ = np.array(self._rs.explained_variance_)
        self.explained_variance_ratio_ = np.array(self._rs.explained_variance_ratio_)
        self.mean_ = np.array(self._rs.mean_)
        self.singular_values_ = np.array(self._rs.singular_values_)
        # n_components_ is the resolved component count (row count of components_)
        # and noise_variance_ is the mean of the discarded tail eigenvalues, or
        # 0.0 when all components are kept (sklearn/decomposition/_pca.py:686-691).
        self.n_components_ = int(self._rs.n_components_)
        self.noise_variance_ = float(self._rs.noise_variance_)
        self._fit_X = X.copy()
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.transform(X))
        # Fallback: (X - mean_) @ components_.T
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = np.asarray(X, dtype="float64")
        X = _ensure_f64(X)
        if hasattr(self, "_rs"):
            return np.array(self._rs.inverse_transform(X))
        return X @ self.components_ + self.mean_

    def get_feature_names_out(self, input_features=None):
        # ClassNamePrefixFeaturesOutMixin: PCA emits `pca0..pca{n_components_-1}`
        # (sklearn/decomposition/_base.py / _ClassNamePrefixFeaturesOutMixin).
        check_is_fitted(self)
        n_out = self.components_.shape[0]
        return np.asarray([f"pca{i}" for i in range(n_out)], dtype=object)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_fit_X"):
            n_components = (
                self.n_components
                if self.n_components is not None
                else min(self._fit_X.shape)
            )
            self._rs = _RsPCA(n_components=n_components)
            self._rs.fit(_ensure_f64(self._fit_X))
