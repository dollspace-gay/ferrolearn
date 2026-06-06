"""sklearn-compatible wrappers for ferrolearn transformer models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import _RsPCA, _RsStandardScaler


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _resolve_n_components(nc):
    """Classify the verbatim PCA `n_components` ctor value into binding kwargs.

    Mirrors sklearn's `PCA` `n_components` parameter semantics
    (`sklearn/decomposition/_pca.py:147-166`, `_parameter_constraints` `:386-391`):

      * ``None`` -> ``(None, None)`` — keep ``min(n_samples, n_features)``
        components (resolved by the Rust ``PCA::auto`` / ``NComponents::Auto``,
        `_pca.py:523-525`).
      * ``int >= 1`` -> ``(int(nc), None)`` — an exact component count (Rust
        ``PCA::new`` / ``NComponents::Count``). A ``bool`` is an ``int`` subclass
        and is ACCEPTED here, matching sklearn's ``Interval(Integral, ...)``
        constraint (`_pca.py:386-391`; ``bool`` is ``Integral``): ``True`` -> 1
        component, ``False`` -> 0 (which the Rust ``Count(0)`` validation rejects
        at fit, like sklearn's strictly-positive component count).
      * ``float`` in ``(0, 1)`` -> ``(None, float(nc))`` — select the smallest
        count whose cumulative ``explained_variance_ratio_`` is ``>= nc``, AFTER
        the SVD (Rust ``PCA::with_variance_ratio`` / ``NComponents::Ratio``,
        `_pca.py:659-681`).
      * anything else (float ``>= 1`` or ``<= 0``, or an unsupported type)
        -> ``ValueError`` matching sklearn's constraint-message shape.

    Returns a ``(count, ratio)`` tuple where exactly one is non-None (or both None
    for the Auto case). Validation happens at FIT time (matching sklearn's
    `_validate_params`-at-fit), so ``PCA(n_components=2.0)`` constructs fine and
    only ``.fit()`` raises.
    """
    if nc is None:
        return (None, None)
    # Booleans are an int subclass; sklearn accepts them via the
    # Interval(Integral, ...) constraint (bool is Integral), so True -> 1 and
    # False -> 0. Check the int branch FIRST so a bool routes to Count, not the
    # float path. (False -> Count(0), which the Rust validation rejects at fit,
    # matching sklearn's strictly-positive component count.)
    if isinstance(nc, (int, np.integer)):
        if int(nc) < 1:
            raise ValueError(
                f"The 'n_components' parameter of PCA must be an int in the "
                f"range [0, inf), a float in the range (0, 1), 'mle' or None. "
                f"Got {nc!r} instead."
            )
        return (int(nc), None)
    if isinstance(nc, (float, np.floating)):
        if 0.0 < float(nc) < 1.0:
            return (None, float(nc))
        raise ValueError(
            f"The 'n_components' parameter of PCA must be an int in the range "
            f"[0, inf), a float in the range (0, 1), 'mle' or None. Got {nc!r} "
            f"instead."
        )
    raise ValueError(
        f"The 'n_components' parameter of PCA must be an int in the range "
        f"[0, inf), a float in the range (0, 1), 'mle' or None. Got {nc!r} instead."
    )


def _resolve_random_state(random_state):
    """Classify the verbatim PCA `random_state` ctor value into the binding seed.

    Mirrors sklearn's `PCA(random_state=...)` semantics
    (`sklearn/decomposition/_pca.py:418`, consumed only by the `'auto'`-selected
    `randomized`/`arpack` solvers, `:738`,`:771`):

      * ``None`` -> ``None`` — sklearn's default; the downstream Rust randomized
        branch then uses a fixed reproducible draw (it cannot reproduce numpy's
        GLOBAL RNG state).
      * ``int >= 0`` -> ``int(random_state)`` — the seed fed to
        ``ferray::random::RandomState::new(seed)`` (bit-identical to
        ``numpy.random.RandomState(seed)``), so the randomized spectrum
        reproduces sklearn's for that seed. ``np.random.RandomState`` instances
        and negative ints are not supported by the ``u64`` binding and raise.
    """
    if random_state is None:
        return None
    if isinstance(random_state, (int, np.integer)) and not isinstance(random_state, bool):
        if int(random_state) < 0:
            raise ValueError(
                f"random_state must be a non-negative int or None, got "
                f"{random_state!r}"
            )
        return int(random_state)
    raise ValueError(
        f"random_state must be a non-negative int or None, got {random_state!r}"
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
    whiten : bool, default=False
        When True, the projected components are divided by
        ``sqrt(explained_variance_)`` so the transformed output has unit
        component-wise variance, matching ``sklearn.decomposition.PCA``
        (``sklearn/decomposition/_pca.py:412``; whitening at
        ``sklearn/decomposition/_base.py:157-165``). The whitening math lives in
        the Rust ``PCA::with_whiten`` builder (``ferrolearn-decomp/src/pca.rs:180``).
    """

    def __init__(self, n_components=None, *, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._validate_data(X, dtype="float64")
        # sklearn stores n_components verbatim and validates/resolves it at fit
        # (`_pca.py:523-529`/`:659-681`; `_validate_params`-at-fit). Classify the
        # verbatim value into the binding's (count, ratio) spec — None -> Auto,
        # int -> Count, float in (0,1) -> Ratio (selected after the SVD), else a
        # ValueError matching sklearn.
        count, ratio = _resolve_n_components(self.n_components)
        self._rs = _RsPCA(
            count=count,
            ratio=ratio,
            whiten=self.whiten,
            random_state=_resolve_random_state(self.random_state),
        )
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

    def _ensure_rs(self):
        # Rebuild the Rust handle (dropped on pickle) the same way __setstate__
        # does, threading the resolved n_components + whiten. score/score_samples
        # have no closed-form Python fallback, so _rs must exist.
        if not hasattr(self, "_rs") and hasattr(self, "_fit_X"):
            count, ratio = _resolve_n_components(self.n_components)
            self._rs = _RsPCA(
                count=count,
                ratio=ratio,
                whiten=getattr(self, "whiten", False),
                random_state=_resolve_random_state(getattr(self, "random_state", None)),
            )
            self._rs.fit(_ensure_f64(self._fit_X))

    def score_samples(self, X):
        # Per-sample Gaussian log-likelihood under the fitted model
        # (sklearn/decomposition/_pca.py:805-830). Delegates to the Rust
        # FittedPCA::score_samples (ferrolearn-decomp/src/pca.rs:484, REQ-15).
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        self._ensure_rs()
        return np.asarray(self._rs.score_samples(X))

    def score(self, X, y=None):
        # Average Gaussian log-likelihood of all samples — the mean of
        # score_samples (sklearn/decomposition/_pca.py:832-853). `y` is ignored
        # (sklearn signature `score(self, X, y=None)`). Delegates to the Rust
        # FittedPCA::score (ferrolearn-decomp/src/pca.rs:533, REQ-15).
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        self._ensure_rs()
        return float(self._rs.score(X))

    def get_covariance(self):
        # Data covariance with the generative model:
        #   cov = components_.T * exp_var_diff * components_ + noise_variance_ * I
        # (sklearn/decomposition/_base.py:30-56). No X argument — returns the
        # model's own data covariance. Delegates to the Rust
        # FittedPCA::get_covariance (ferrolearn-decomp/src/pca.rs:328, REQ-14).
        check_is_fitted(self)
        self._ensure_rs()
        return np.array(self._rs.get_covariance())

    def get_precision(self):
        # Data precision matrix (inverse of get_covariance) with the generative
        # model (sklearn/decomposition/_base.py:58-101). No X argument.
        # Delegates to the Rust FittedPCA::get_precision
        # (ferrolearn-decomp/src/pca.rs:390, REQ-14).
        check_is_fitted(self)
        self._ensure_rs()
        return np.array(self._rs.get_precision())

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
            count, ratio = _resolve_n_components(self.n_components)
            self._rs = _RsPCA(
                count=count,
                ratio=ratio,
                whiten=getattr(self, "whiten", False),
                random_state=_resolve_random_state(getattr(self, "random_state", None)),
            )
            self._rs.fit(_ensure_f64(self._fit_X))
