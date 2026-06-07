"""sklearn-compatible Python wrappers for the ~40 extended-surface estimators
bound in extras.rs. Minimal API: __init__, fit, predict or transform. Inherits
sklearn mixins for `.score()` / `fit_transform()`.
"""

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import (
    _RsARDRegression,
    _RsAdaBoostClassifier,
    _RsAgglomerativeClustering,
    _RsBaggingClassifier,
    _RsBayesianRidge,
    _RsBernoulliNB,
    _RsBirch,
    _RsComplementNB,
    _RsDBSCAN,
    _RsDecisionTreeRegressor,
    _RsExtraTreeClassifier,
    _RsExtraTreesClassifier,
    _RsExtraTreesRegressor,
    _RsFactorAnalysis,
    _RsFastICA,
    _RsGaussianMixture,
    _RsGradientBoostingClassifier,
    _RsGradientBoostingRegressor,
    _RsHistGradientBoostingClassifier,
    _RsHistGradientBoostingRegressor,
    _RsHuberRegressor,
    _RsIncrementalPCA,
    _RsKNeighborsRegressor,
    _RsKernelPCA,
    _RsKernelRidge,
    _RsLars,
    _RsLassoLars,
    _RsLinearSVC,
    _RsMaxAbsScaler,
    _RsMinMaxScaler,
    _RsMiniBatchKMeans,
    _RsMultinomialNB,
    _RsNMF,
    _RsNearestCentroid,
    _RsNystroem,
    _RsOrthogonalMatchingPursuit,
    _RsPowerTransformer,
    _RsQDA,
    _RsQuantileRegressor,
    _RsRBFSampler,
    _RsRandomForestRegressor,
    _RsRidgeClassifier,
    _RsRobustScaler,
    _RsSparsePCA,
    _RsTruncatedSVD,
)

# sklearn 1.5.2's full set of recognized neighbors metric names
# (`set(itertools.chain(*sklearn.neighbors.VALID_METRICS.values()))`,
# neighbors/_base.py:401). A metric STRING outside this set is an
# InvalidParameterError (a ValueError subclass) in sklearn; a metric INSIDE it
# that ferrolearn's Euclidean-only core cannot compute (e.g. 'cityblock',
# 'cosine', 'manhattan', 'l1') is an honest NotImplementedError (#876).
_SKLEARN_VALID_KNN_METRICS = frozenset(
    {
        "braycurtis", "canberra", "chebyshev", "cityblock", "correlation",
        "cosine", "dice", "euclidean", "hamming", "haversine", "infinity",
        "jaccard", "l1", "l2", "mahalanobis", "manhattan", "minkowski",
        "nan_euclidean", "p", "precomputed", "pyfunc", "rogerstanimoto",
        "russellrao", "seuclidean", "sokalmichener", "sokalsneath",
        "sqeuclidean", "yule",
    }
)


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def _encode(y):
    classes = np.unique(y)
    enc = np.searchsorted(classes, y).astype(np.int64)
    return enc, classes


# ---------------------------------------------------------------------------
# Regressor wrappers
# ---------------------------------------------------------------------------

class _RegressorWrapper(RegressorMixin, BaseEstimator):
    _RsClass = None

    def fit(self, X, y):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X), _f64(y))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))


class _RegressorPickleMixin:
    """Mixin for pickling regressors whose Rust fitted object is not directly
    picklable (mirrors `_classifiers.py::_ClassifierPickleMixin`). The training
    data is stored on fit and the `_rs` handle is dropped from the pickled
    state and rebuilt by re-fitting on unpickle."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_fit_X") and hasattr(self, "_fit_y"):
            self._rebuild_rs()

    def _store_training_data(self, X, y):
        self._fit_X = X.copy()
        self._fit_y = y.copy()

    def _rebuild_rs(self):
        self._rs = self._make_rs()
        self._rs.fit(self._fit_X, self._fit_y)


class BayesianRidge(RegressorMixin, BaseEstimator):
    """Bayesian ridge regression backed by Rust (#2161).

    Mirrors ``sklearn.linear_model.BayesianRidge``
    (``sklearn/linear_model/_bayes.py:187-202``), surfacing the
    ``compute_score`` keyword-only parameter, the
    ``fit(X, y, sample_weight=None)`` signature, the
    ``predict(X, return_std=False)`` path, and the fitted
    ``coef_``/``intercept_``/``alpha_``/``lambda_``/``sigma_``/``n_iter_``/
    ``scores_`` attributes from the Rust fitted type.

    ``scores_`` (the per-iteration log marginal likelihood, length
    ``n_iter_ + 1``) is populated only when ``compute_score=True``
    (``_bayes.py:198``); otherwise it is an empty array.
    """

    def __init__(self, *, max_iter=300, tol=1e-3, compute_score=False,
                 fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsBayesianRidge(max_iter=self.max_iter, tol=self.tol,
                                compute_score=self.compute_score,
                                fit_intercept=self.fit_intercept)

    def fit(self, X, y, sample_weight=None):
        self._rs = self._make_rs()
        if sample_weight is None:
            self._rs.fit(_f64(X), _f64(y))
        else:
            self._rs.fit(_f64(X), _f64(y), _f64(sample_weight))
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/linear_model/_bayes.py:94-120).
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self.alpha_ = self._rs.alpha_
        self.lambda_ = self._rs.lambda_
        self.sigma_ = np.asarray(self._rs.sigma_)
        self.n_iter_ = int(self._rs.n_iter_)
        # scores_ is empty unless compute_score=True (matches sklearn, which
        # only sets it under that flag).
        self.scores_ = np.asarray(self._rs.scores_)
        return self

    def predict(self, X, return_std=False):
        if not return_std:
            return np.asarray(self._rs.predict(_f64(X)))
        mean, std = self._rs.predict(_f64(X), return_std=True)
        return np.asarray(mean), np.asarray(std)


class ARDRegression(RegressorMixin, BaseEstimator):
    """Bayesian ARD regression backed by Rust (#2163).

    Mirrors ``sklearn.linear_model.ARDRegression``
    (``sklearn/linear_model/_bayes.py:578-603``), surfacing the
    ``compute_score`` keyword-only parameter, the
    ``predict(X, return_std=False)`` path, and the fitted
    ``coef_``/``intercept_``/``alpha_``/``lambda_``/``sigma_``/``n_iter_``/
    ``scores_`` attributes from the Rust fitted type.

    ``scores_`` (the per-iteration ARD objective, length ``n_iter_``) is
    populated only when ``compute_score=True`` (``_bayes.py:587``); otherwise it
    is an empty array. ``sigma_`` is the kept-feature ``(n_kept, n_kept)``
    posterior covariance, exactly as sklearn exposes it (``_bayes.py:727``).
    """

    def __init__(self, *, max_iter=300, tol=1e-3, compute_score=False,
                 fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsARDRegression(max_iter=self.max_iter, tol=self.tol,
                                compute_score=self.compute_score,
                                fit_intercept=self.fit_intercept)

    def fit(self, X, y):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X), _f64(y))
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/linear_model/_bayes.py:488-512).
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self.alpha_ = self._rs.alpha_
        self.lambda_ = np.asarray(self._rs.lambda_)
        self.sigma_ = np.asarray(self._rs.sigma_)
        self.n_iter_ = int(self._rs.n_iter_)
        # scores_ is empty unless compute_score=True (matches sklearn, which
        # only sets it under that flag).
        self.scores_ = np.asarray(self._rs.scores_)
        return self

    def predict(self, X, return_std=False):
        if not return_std:
            return np.asarray(self._rs.predict(_f64(X)))
        mean, std = self._rs.predict(_f64(X), return_std=True)
        return np.asarray(mean), np.asarray(std)


class OrthogonalMatchingPursuit(_RegressorPickleMixin, RegressorMixin,
                                BaseEstimator):
    """Orthogonal Matching Pursuit backed by Rust (#2172).

    Mirrors ``sklearn.linear_model.OrthogonalMatchingPursuit``
    (``sklearn/linear_model/_omp.py:645-753``), surfacing the keyword-only
    constructor ``(n_nonzero_coefs=None, tol=None, fit_intercept=True,
    precompute='auto')`` (sklearn defaults/order, ``_omp.py:742-753``), the
    ``fit(X, y)``/``predict(X)`` contract, and the fitted
    ``coef_``/``intercept_``/``n_features_in_`` attributes from the Rust fitted
    type (``_omp.py:814-815``, ``coef_`` shape ``(n_features,)``).

    ``precompute`` is a Gram-matrix speed knob that does NOT change the OMP
    solution (``_omp.py:791-813``): ``'auto'``/``True``/``False`` all yield the
    identical fit. ferrolearn's core never uses a Gram path, so it accepts any
    ``precompute`` value and ignores it (held only for ``get_params``/``clone``
    round-trip). ``n_iter_``/``n_nonzero_coefs_`` are NOT exposed — the Rust core
    does not compute them (NOT-STARTED, omp.rs REQ-7 #491).

    Pickle: the Rust fitted object is not directly picklable, so the training
    data is stored on fit and ``_rs`` is rebuilt by re-fitting on unpickle
    (``_RegressorPickleMixin``).
    """

    def __init__(self, *, n_nonzero_coefs=None, tol=None, fit_intercept=True,
                 precompute='auto'):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.precompute = precompute

    def _make_rs(self):
        # sklearn's `_validate_params` (`_omp.py:735-740` `_parameter_constraints`)
        # runs at fit and raises `InvalidParameterError` (a `ValueError` subclass)
        # for out-of-range inputs. Mirror it here, Python-side, before the Rust ABI
        # boundary (where 0/negatives would mis-fit or TypeError). bool is Integral,
        # so np.bool_/True/False would pass the int check — but precompute (the only
        # bool-valued param) is handled by its own StrOptions/boolean check below.
        if self.n_nonzero_coefs is not None and (
            not isinstance(self.n_nonzero_coefs, (int, np.integer))
            or isinstance(self.n_nonzero_coefs, bool)
            or self.n_nonzero_coefs < 1
        ):
            raise ValueError(
                f"n_nonzero_coefs == {self.n_nonzero_coefs}, must be >= 1 "
                "(an int in [1, inf)) or None."
            )
        # tol: sklearn constraint `Interval(Real, 0, None, closed='left')`
        # (`_omp.py:737`) -> None or a real >= 0; a negative tol is rejected.
        if self.tol is not None and (
            not isinstance(self.tol, (int, float, np.floating, np.integer))
            or isinstance(self.tol, bool)
            or self.tol < 0
        ):
            raise ValueError(
                f"tol == {self.tol}, must be >= 0 (a real in [0, inf)) or None."
            )
        # precompute: sklearn constraint `[StrOptions({'auto'}), 'boolean']`
        # (`_omp.py:739`) -> 'auto', True, or False; anything else is rejected.
        # It is a Gram-matrix speed knob that does NOT change the OMP solution
        # (`_omp.py:791-813`), so the VALID values are accepted-and-ignored by the
        # Rust core; only the INVALID ones are rejected here.
        if not (self.precompute == "auto"
                or isinstance(self.precompute, (bool, np.bool_))):
            raise ValueError(
                f"precompute == {self.precompute!r}, must be 'auto', True, or False."
            )
        # tol OVERRIDES n_nonzero_coefs (sklearn `_omp.py:786-787`: when `tol is not
        # None`, `n_nonzero_coefs_` is forced to None, so tol is the sole stopping
        # criterion and the support is capped only at n_features, `_omp.py:94`).
        # Pass `n_nonzero_coefs=None` to the core in that case so it does not also
        # cap the support — matching sklearn's tol-only fit.
        n_nonzero_coefs = None if self.tol is not None else self.n_nonzero_coefs
        return _RsOrthogonalMatchingPursuit(
            n_nonzero_coefs=n_nonzero_coefs, tol=self.tol,
            fit_intercept=self.fit_intercept, precompute=self.precompute)

    def fit(self, X, y):
        X = _f64(X)
        y = _f64(y)
        self._rs = self._make_rs()
        self._rs.fit(X, y)
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/linear_model/_omp.py:814-815).
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self._store_training_data(X, y)
        return self

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))


class Lars(_RegressorPickleMixin, RegressorMixin, BaseEstimator):
    """Least Angle Regression backed by Rust (#2174).

    Mirrors ``sklearn.linear_model.Lars``
    (``sklearn/linear_model/_least_angle.py:922-1068``), surfacing the full
    keyword-only constructor ``(fit_intercept=True, verbose=False,
    precompute='auto', n_nonzero_coefs=500, eps=np.finfo(float).eps,
    copy_X=True, fit_path=True, jitter=None, random_state=None)`` (sklearn
    defaults/order, ``_least_angle.py:1047-1058``), the ``fit(X, y)``/
    ``predict(X)`` contract, and the fitted ``coef_`` (shape ``(n_features,)``)/
    ``intercept_``/``n_features_in_`` attributes from the Rust fitted type
    (``_least_angle.py:1125``).

    Of the 9 parameters only ``fit_intercept`` and ``n_nonzero_coefs`` change the
    supported result; they are threaded into the Rust core. The other seven are
    validated to sklearn's ``_parameter_constraints``
    (``_least_angle.py:1032-1042``) and accepted-and-ignored on the supported
    path:

    - ``verbose`` (bool / int), ``copy_X`` (bool) — diagnostics / input-copy
      knobs; never change ``coef_``.
    - ``precompute`` (``'auto'``/``True``/``False``/ndarray/``None``,
      ``_least_angle.py:1035``) — a Gram-matrix speed knob (``_get_gram``,
      ``_least_angle.py:1070-1079``); the LARS path is identical with or without
      it, so every value yields the same fit.
    - ``eps`` (Real >= 0, ``_least_angle.py:1037``) — Cholesky regularization
      floor; numerical-stability only.
    - ``fit_path`` (bool) — when ``False`` sklearn skips storing the full
      ``coef_path_`` but the final ``coef_`` is IDENTICAL
      (``_least_angle.py:1133-1151``).
    - ``random_state`` — consumed only by ``jitter`` (seeds the gaussian noise);
      a no-op when ``jitter is None``.

    ``jitter`` (a non-None Real >= 0) adds scaled SEEDED gaussian noise to ``y``
    before the fit (``_least_angle.py:1170-1175``), which DOES change ``coef_``.
    The Rust core cannot reproduce numpy's seeded normal draws (R-SUBSTRATE-5), so
    a non-None ``jitter`` raises ``NotImplementedError`` (#2174-tracked); the
    ``None`` default is the supported path. ``n_nonzero_coefs`` is an upper bound
    on the active set; sklearn caps it at ``n_features`` (``_least_angle.py``
    lars_path ``max_features``), so the wrapper passes
    ``min(n_nonzero_coefs, n_features)`` to the core (which errors when the bound
    exceeds ``n_features``) — yielding the same support as sklearn's cap.

    Pickle: the Rust fitted object is not directly picklable, so the training
    data is stored on fit and ``_rs`` is rebuilt by re-fitting on unpickle
    (``_RegressorPickleMixin``).
    """

    def __init__(self, *, fit_intercept=True, verbose=False, precompute='auto',
                 n_nonzero_coefs=500, eps=np.finfo(float).eps, copy_X=True,
                 fit_path=True, jitter=None, random_state=None):
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.jitter = jitter
        self.random_state = random_state

    def _validate(self, n_features):
        # sklearn's `_validate_params` runs at fit and raises
        # `InvalidParameterError` (a `ValueError` subclass) for out-of-range
        # inputs. Mirror sklearn's `Lars._parameter_constraints`
        # (`_least_angle.py:1032-1042`), Python-side, before the usize/bool ABI
        # boundary (where 0/negatives would mis-fit or overflow). bool is
        # Integral; n_nonzero_coefs forbids bool (matching sklearn's
        # `Interval(Integral, 1, None)` which rejects bools via its own check).
        if (not isinstance(self.n_nonzero_coefs, (int, np.integer))
                or isinstance(self.n_nonzero_coefs, bool)
                or self.n_nonzero_coefs < 1):
            raise ValueError(
                f"n_nonzero_coefs == {self.n_nonzero_coefs}, must be >= 1 "
                "(an int in [1, inf))."
            )
        # eps: Interval(Real, 0, None, closed='left') (`_least_angle.py:1037`).
        if (not isinstance(self.eps, (int, float, np.floating, np.integer))
                or isinstance(self.eps, bool)
                or self.eps < 0):
            raise ValueError(
                f"eps == {self.eps}, must be >= 0 (a real in [0, inf))."
            )
        # copy_X: ['boolean'] (`_least_angle.py:1039`). An input-copy speed knob
        # that never changes coef_ (accept-and-ignore for valid bools); sklearn's
        # boolean validator accepts bool/np.bool_ and rejects ints/strings, so a
        # non-bool copy_X is an InvalidParameterError (a ValueError subclass) (#2177).
        if not isinstance(self.copy_X, (bool, np.bool_)):
            raise ValueError(
                f"copy_X == {self.copy_X!r}, must be a boolean."
            )
        # precompute: [boolean, StrOptions({'auto'}), np.ndarray, None]
        # (`_least_angle.py:1035`). A Gram-matrix speed knob (no result change);
        # the VALID values are accepted-and-ignored, only INVALID ones rejected.
        if not (self.precompute == "auto"
                or isinstance(self.precompute, (bool, np.bool_, np.ndarray))
                or self.precompute is None):
            raise ValueError(
                f"precompute == {self.precompute!r}, must be 'auto', a boolean, "
                "an array, or None."
            )
        # jitter: [Interval(Real, 0, None, closed='left'), None]
        # (`_least_angle.py:1040`). A non-None jitter adds SEEDED gaussian noise
        # to y (changes coef_) the Rust core cannot reproduce (R-SUBSTRATE-5).
        if self.jitter is not None:
            if (not isinstance(self.jitter, (int, float, np.floating, np.integer))
                    or isinstance(self.jitter, bool)
                    or self.jitter < 0):
                raise ValueError(
                    f"jitter == {self.jitter}, must be >= 0 (a real in [0, inf)) "
                    "or None."
                )
            raise NotImplementedError(
                "jitter != None not supported (adds seeded gaussian noise to y; "
                "needs numpy's RNG stream — NOT-STARTED #2174)."
            )

    def _make_rs(self, n_features):
        self._validate(n_features)
        # n_nonzero_coefs is an upper bound; sklearn caps it at n_features. The
        # Rust core errors when the bound exceeds n_features, so clamp here.
        n_nonzero = min(self.n_nonzero_coefs, n_features)
        return _RsLars(n_nonzero_coefs=n_nonzero,
                       fit_intercept=self.fit_intercept)

    def fit(self, X, y):
        X = _f64(X)
        y = _f64(y)
        self._rs = self._make_rs(X.shape[1])
        self._rs.fit(X, y)
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/linear_model/_least_angle.py:1125).
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self._store_training_data(X, y)
        return self

    def _rebuild_rs(self):
        self._rs = self._make_rs(self._fit_X.shape[1])
        self._rs.fit(self._fit_X, self._fit_y)

    def predict(self, X):
        check_is_fitted(self)
        X = _f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.predict(X))


class LassoLars(_RegressorPickleMixin, RegressorMixin, BaseEstimator):
    """Lasso model fit with Least Angle Regression backed by Rust (#2174).

    Mirrors ``sklearn.linear_model.LassoLars``
    (``sklearn/linear_model/_least_angle.py:1212-1388``), surfacing the full
    constructor ``(alpha=1.0, *, fit_intercept=True, verbose=False,
    precompute='auto', max_iter=500, eps=np.finfo(float).eps, copy_X=True,
    fit_path=True, positive=False, jitter=None, random_state=None)`` — note
    ``alpha`` is positional-or-keyword FIRST, the rest keyword-only
    (``_least_angle.py:1363-1376``) — the ``fit(X, y)``/``predict(X)`` contract,
    and the fitted ``coef_``/``intercept_``/``n_features_in_`` attributes from the
    Rust fitted type.

    ``alpha``, ``max_iter`` and ``fit_intercept`` are threaded into the Rust core.
    The accept-and-ignore params are the same as ``Lars``
    (``verbose``/``precompute``/``eps``/``copy_X``/``fit_path``/``random_state``),
    validated to sklearn's ``_parameter_constraints``
    (``_least_angle.py:1353-1359``). ``jitter != None`` raises
    ``NotImplementedError`` (seeded gaussian noise, R-SUBSTRATE-5; #2174).

    ``positive=True`` (``_least_angle.py:1357``) constrains all coefficients to be
    non-negative — a different optimization the Rust core does NOT implement
    (``lars.rs`` has no ``positive`` builder), so it raises ``NotImplementedError``
    (NOT-STARTED #2174); ``positive=False`` (default) is the supported path.

    Pickle: the Rust fitted object is not directly picklable, so the training
    data is stored on fit and ``_rs`` is rebuilt by re-fitting on unpickle
    (``_RegressorPickleMixin``).
    """

    def __init__(self, alpha=1.0, *, fit_intercept=True, verbose=False,
                 precompute='auto', max_iter=500, eps=np.finfo(float).eps,
                 copy_X=True, fit_path=True, positive=False, jitter=None,
                 random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.precompute = precompute
        self.max_iter = max_iter
        self.eps = eps
        self.copy_X = copy_X
        self.fit_path = fit_path
        self.positive = positive
        self.jitter = jitter
        self.random_state = random_state

    def _validate(self):
        # Mirror sklearn's `LassoLars._parameter_constraints`
        # (`_least_angle.py:1353-1359`: alpha Interval(Real, 0, None),
        # max_iter Interval(Integral, 0, None), positive boolean; inherits
        # Lars's eps/precompute/jitter), Python-side, before the f64/usize/bool
        # ABI boundary.
        if (not isinstance(self.alpha, (int, float, np.floating, np.integer))
                or isinstance(self.alpha, bool)
                or self.alpha < 0):
            raise ValueError(
                f"alpha == {self.alpha}, must be >= 0 (a real in [0, inf))."
            )
        # max_iter: Interval(Integral, 0, None, closed='left')
        # (`_least_angle.py:1356`) -> a non-negative int (bool forbidden).
        if (not isinstance(self.max_iter, (int, np.integer))
                or isinstance(self.max_iter, bool)
                or self.max_iter < 0):
            raise ValueError(
                f"max_iter == {self.max_iter}, must be >= 0 (an int in [0, inf))."
            )
        if (not isinstance(self.eps, (int, float, np.floating, np.integer))
                or isinstance(self.eps, bool)
                or self.eps < 0):
            raise ValueError(
                f"eps == {self.eps}, must be >= 0 (a real in [0, inf))."
            )
        # copy_X: ['boolean'] (`_least_angle.py:1039`, inherited by LassoLars).
        # An input-copy speed knob that never changes coef_; a non-bool copy_X is
        # an InvalidParameterError (a ValueError subclass) in sklearn (#2177).
        if not isinstance(self.copy_X, (bool, np.bool_)):
            raise ValueError(
                f"copy_X == {self.copy_X!r}, must be a boolean."
            )
        if not (self.precompute == "auto"
                or isinstance(self.precompute, (bool, np.bool_, np.ndarray))
                or self.precompute is None):
            raise ValueError(
                f"precompute == {self.precompute!r}, must be 'auto', a boolean, "
                "an array, or None."
            )
        # positive: boolean (`_least_angle.py:1357`). True is a DIFFERENT
        # optimization (non-negative coefficients) the Rust core does not
        # implement (NOT-STARTED #2174).
        if not isinstance(self.positive, (bool, np.bool_)):
            raise ValueError(
                f"positive == {self.positive!r}, must be a boolean."
            )
        if self.positive:
            raise NotImplementedError(
                "positive=True not supported (non-negative coefficient "
                "constraint; the Rust LassoLars core has no `positive` path — "
                "NOT-STARTED #2174)."
            )
        if self.jitter is not None:
            if (not isinstance(self.jitter, (int, float, np.floating, np.integer))
                    or isinstance(self.jitter, bool)
                    or self.jitter < 0):
                raise ValueError(
                    f"jitter == {self.jitter}, must be >= 0 (a real in [0, inf)) "
                    "or None."
                )
            raise NotImplementedError(
                "jitter != None not supported (adds seeded gaussian noise to y; "
                "needs numpy's RNG stream — NOT-STARTED #2174)."
            )

    def _make_rs(self):
        self._validate()
        return _RsLassoLars(alpha=float(self.alpha), max_iter=self.max_iter,
                            fit_intercept=self.fit_intercept)

    def fit(self, X, y):
        X = _f64(X)
        y = _f64(y)
        self._rs = self._make_rs()
        self._rs.fit(X, y)
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/linear_model/_least_angle.py:1125).
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self._store_training_data(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = _f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.predict(X))


class HuberRegressor(_RegressorWrapper):
    """Huber regressor backed by Rust (#500/#501).

    Mirrors ``sklearn.linear_model.HuberRegressor``
    (``sklearn/linear_model/_huber.py:259-274``), including the keyword-only
    ``warm_start`` parameter and the ``fit(X, y, sample_weight=None)`` signature.
    On a warm-start refit the Rust binding reuses the previously fitted
    ``(coef_, intercept_, scale_)`` as the optimizer seed (``_huber.py:308-309``);
    because the Huber objective is convex the converged fit is unchanged and the
    refit converges in fewer iterations. The fitted ``coef_``/``intercept_``/
    ``scale_``/``n_iter_`` attributes are surfaced from the Rust fitted type.
    ``n_iter_`` is the honest ferrolearn L-BFGS iteration count (``_huber.py:342``
    ``self.n_iter_ = opt_res.nit``); R-DEV-7: the Rust core is not scipy's
    L-BFGS-B, so it need not equal sklearn's exactly, but it is a positive int
    ``<= max_iter`` and a warm refit reports fewer iterations than the cold fit.
    """

    def __init__(self, *, epsilon=1.35, alpha=1e-4, max_iter=100, tol=1e-5,
                 warm_start=False, fit_intercept=True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsHuberRegressor(epsilon=self.epsilon, alpha=self.alpha,
                                 max_iter=self.max_iter, tol=self.tol,
                                 fit_intercept=self.fit_intercept,
                                 warm_start=self.warm_start)

    def fit(self, X, y, sample_weight=None):
        # warm_start (sklearn `_huber.py:308`): reuse the prior `_rs` (which
        # holds the previously fitted attributes) so the Rust core seeds the
        # optimizer from them. Otherwise build a fresh handle (cold start).
        if not (self.warm_start and hasattr(self, "_rs")):
            self._rs = self._make_rs()
        if sample_weight is None:
            self._rs.fit(_f64(X), _f64(y))
        else:
            self._rs.fit(_f64(X), _f64(y), _f64(sample_weight))
        self.n_features_in_ = X.shape[1]
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self.scale_ = self._rs.scale_
        # sklearn `n_iter_` (`_huber.py:342` `self.n_iter_ = opt_res.nit`): the
        # number of optimizer iterations the fit ran for. R-DEV-7: the Rust core
        # is not scipy's L-BFGS-B, so this is the honest ferrolearn iteration
        # count (positive int <= max_iter; a warm refit takes fewer than cold).
        self.n_iter_ = int(self._rs.n_iter_)
        return self


class QuantileRegressor(_RegressorWrapper):
    """Quantile regression backed by Rust (#507/#508).

    Mirrors ``sklearn.linear_model.QuantileRegressor``
    (``sklearn/linear_model/_quantile.py:128-141``), surfacing the keyword-only
    ``solver`` (default ``"highs"``) and ``solver_options`` (default ``None``)
    constructor parameters and the fitted ``n_iter_`` attribute
    (``_quantile.py:300`` ``self.n_iter_ = result.nit``).

    ``solver`` accepts sklearn's ``_parameter_constraints`` set
    (``_quantile.py:114-124``): ``"highs"``/``"highs-ds"``/``"highs-ipm"`` and
    ``"revised simplex"`` all reach the SAME unique LP vertex, so they map onto
    ferrolearn's two-phase primal simplex and give identical ``coef_``.
    ``"interior-point"`` raises ``ValueError`` (it was removed in SciPy 1.11.0,
    ``_quantile.py:196-199``); any other string is an invalid parameter
    (``ValueError``, matching sklearn's ``InvalidParameterError``).

    ``solver_options`` accepts ``None`` and an empty dict (HiGHS tuning knobs
    that do not change the LP optimum, ``_quantile.py:206``); a non-empty dict
    raises ``NotImplementedError`` (ferrolearn's simplex has no such tuning
    surface; #508-tracked).

    ``n_iter_`` is the honest ferrolearn two-phase-simplex PIVOT count
    (``_quantile.py:300``); R-DEV-7: ferrolearn's solver is NOT scipy's HiGHS,
    so this need not equal sklearn's ``n_iter_``, but it is a positive int
    ``<= max_iter`` and deterministic for a given dataset.
    """

    def __init__(self, *, quantile=0.5, alpha=1.0, max_iter=10000,
                 tol=1e-6, fit_intercept=True, solver="highs",
                 solver_options=None):
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_options = solver_options

    def _make_rs(self):
        return _RsQuantileRegressor(quantile=self.quantile, alpha=self.alpha,
                                    max_iter=self.max_iter, tol=self.tol,
                                    fit_intercept=self.fit_intercept,
                                    solver=self.solver,
                                    solver_options=self.solver_options)

    def fit(self, X, y):
        super().fit(X, y)
        # sklearn `n_iter_` (`_quantile.py:300`): surfaced from the Rust fitted
        # type (the two-phase-simplex pivot count; R-DEV-7 honest count).
        self.n_iter_ = int(self._rs.n_iter_)
        return self


class DecisionTreeRegressor(_RegressorWrapper):
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _make_rs(self):
        return _RsDecisionTreeRegressor(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf)


class RandomForestRegressor(_RegressorWrapper):
    def __init__(self, n_estimators=100, *, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make_rs(self):
        return _RsRandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state)


class ExtraTreesRegressor(_RegressorWrapper):
    def __init__(self, n_estimators=100, *, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsExtraTreesRegressor(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth,
                                      random_state=self.random_state)


class GradientBoostingRegressor(_RegressorWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsGradientBoostingRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class HistGradientBoostingRegressor(_RegressorWrapper):
    def __init__(self, *, max_iter=100, learning_rate=0.1,
                 max_depth=None, random_state=None):
        # sklearn HistGradientBoostingRegressor uses `max_iter` (the number of
        # boosting iterations), NOT `n_estimators`
        # (sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py).
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        # The Rust binding's `n_estimators` IS the boosting-iteration count,
        # i.e. sklearn's `max_iter`.
        return _RsHistGradientBoostingRegressor(
            n_estimators=self.max_iter, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class KNeighborsRegressor(_RegressorPickleMixin, RegressorMixin, BaseEstimator):
    """K-Nearest Neighbors Regressor backed by Rust (#2147).

    Mirrors ``sklearn.neighbors.KNeighborsRegressor``
    (``sklearn/neighbors/_regression.py:178-189``). ``n_neighbors`` is
    positional-or-keyword; the rest are keyword-only.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. 'uniform' averages the neighbor
        targets equally; 'distance' weights each neighbor target by the inverse
        of its distance (``sklearn/neighbors/_regression.py:43-45``) — this
        CHANGES the prediction. Callable weights are NOT supported
        (NotImplementedError, #876).
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Nearest-neighbor search strategy only — all values give identical
        predict. 'ball_tree' maps to 'auto' (no distinct regressor variant).
    leaf_size : int, default=30
        ABI-only: affects tree-build performance, not the result.
    p : float, default=2
        Minkowski power. Only ``p=2`` (Euclidean) is supported; any other value
        raises NotImplementedError (#876).
    metric : str, default='minkowski'
        Only 'minkowski' (with p=2) and 'euclidean' are supported; any other
        value raises NotImplementedError (#876).
    metric_params : dict or None, default=None
        Only ``None`` is supported (no custom metric); a non-None value raises
        NotImplementedError (#876).
    n_jobs : int or None, default=None
        ABI-only: a threading knob, does not affect the result.
    """

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _make_rs(self):
        # sklearn validates n_neighbors in [1, inf) and leaf_size in [1, inf)
        # (Integral) at fit, raising InvalidParameterError (a ValueError
        # subclass). Mirror that here — Python-side, before the usize ABI
        # boundary that would otherwise raise OverflowError for negatives (#2152).
        if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError(
                f"n_neighbors == {self.n_neighbors}, must be >= 1 (an int in [1, inf))."
            )
        if not isinstance(self.leaf_size, (int, np.integer)) or self.leaf_size < 1:
            raise ValueError(
                f"leaf_size == {self.leaf_size}, must be >= 1 (an int in [1, inf))."
            )
        # n_jobs: sklearn constraint [Integral, None] (neighbors/_base.py:404).
        # A non-int (e.g. 1.5/'x') is an InvalidParameterError (ValueError);
        # mirror that before the Option<i64> ABI boundary that would TypeError
        # (#2152). bool is Integral, so it is accepted (sklearn does too).
        if self.n_jobs is not None and not isinstance(self.n_jobs, (int, np.integer)):
            raise ValueError(f"n_jobs must be an int or None, got {self.n_jobs!r}.")
        # weights: sklearn accepts {'uniform','distance'}, a callable, or None;
        # None behaves identically to 'uniform' (neighbors/_regression.py:173).
        # callable is NOT supported (#876). An invalid string is rejected
        # (ValueError) at the Rust boundary.
        weights = self.weights
        if weights is None:
            weights = "uniform"
        elif callable(weights):
            raise NotImplementedError(
                "callable weights not supported (only 'uniform'/'distance'; "
                "NOT-STARTED #876)"
            )
        elif not isinstance(weights, str):
            raise ValueError(
                "weights must be 'uniform', 'distance', a callable, or None, got "
                f"{weights!r}."
            )
        # algorithm/metric: sklearn requires a str (metric also accepts a
        # callable). A non-str (e.g. None/int) is an InvalidParameterError
        # (ValueError); mirror that before the Rust String boundary that would
        # otherwise TypeError (#2156). An invalid str value is rejected
        # (ValueError / NotImplementedError) inside the Rust binding.
        if not isinstance(self.algorithm, str):
            raise ValueError(
                "algorithm must be one of 'auto', 'ball_tree', 'kd_tree', "
                f"'brute', got {self.algorithm!r}."
            )
        if callable(self.metric):
            raise NotImplementedError(
                "callable metric not supported (no custom metric; NOT-STARTED #876)"
            )
        if not isinstance(self.metric, str):
            raise ValueError(f"metric must be a str or callable, got {self.metric!r}.")
        # A metric string OUTSIDE sklearn's recognized set is an invalid
        # parameter (ValueError); reject it here so it is NOT conflated with a
        # valid-but-unimplemented metric (which the Rust core reports as
        # NotImplementedError #876).
        if self.metric not in _SKLEARN_VALID_KNN_METRICS:
            raise ValueError(
                f"metric={self.metric!r} is not a valid metric "
                "(not among sklearn's recognized metric names)."
            )
        # metric_params: sklearn constraint [dict, None] (neighbors/_base.py:402).
        # An EMPTY dict is valid (no custom params == None) and is accepted; a
        # non-dict (e.g. a list) is a ValueError; a NON-EMPTY dict needs a custom
        # metric the Rust core lacks (NOT-STARTED #876).
        if self.metric_params is not None:
            if not isinstance(self.metric_params, dict):
                raise ValueError(
                    "metric_params must be a dict or None, got "
                    f"{type(self.metric_params).__name__}."
                )
            if len(self.metric_params) > 0:
                raise NotImplementedError(
                    f"metric_params={self.metric_params!r} not supported "
                    "(no custom metric; NOT-STARTED #876)"
                )
        return _RsKNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=float(self.p),
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y):
        X = _f64(X)
        y = _f64(y)
        self._rs = self._make_rs()
        self._rs.fit(X, y)
        self.n_features_in_ = X.shape[1]
        self._store_training_data(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = _f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.predict(X))


class KernelRidge(_RegressorWrapper):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def _make_rs(self):
        return _RsKernelRidge(alpha=self.alpha)


# ---------------------------------------------------------------------------
# Classifier wrappers
# ---------------------------------------------------------------------------

class _ClassifierWrapper(ClassifierMixin, BaseEstimator):
    _preprocess_X = staticmethod(_f64)

    def fit(self, X, y):
        Xp = self._preprocess_X(X)
        y_enc, self.classes_ = _encode(y)
        self._rs = self._make_rs()
        self._rs.fit(Xp, y_enc)
        self.n_features_in_ = Xp.shape[1]
        return self

    def predict(self, X):
        Xp = self._preprocess_X(X)
        y_enc = np.asarray(self._rs.predict(Xp))
        return self.classes_[y_enc]


class RidgeClassifier(_ClassifierWrapper):
    def __init__(self, alpha=1.0, *, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsRidgeClassifier(alpha=self.alpha,
                                  fit_intercept=self.fit_intercept)


class LinearSVC(_ClassifierWrapper):
    def __init__(self, *, C=1.0, max_iter=1000, tol=1e-4):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def _make_rs(self):
        return _RsLinearSVC(c=float(self.C), max_iter=self.max_iter,
                            tol=self.tol)


class QuadraticDiscriminantAnalysis(_ClassifierWrapper):
    def __init__(self, *, reg_param=0.0):
        self.reg_param = reg_param

    def _make_rs(self):
        return _RsQDA(reg_param=self.reg_param)


class _DiscreteNBWrapper(_ClassifierWrapper):
    """Discrete naive-Bayes wrapper (#2103): in addition to the base
    `_ClassifierWrapper` `classes_`/`n_features_in_`, surfaces the four
    `_BaseDiscreteNB` fitted attributes sklearn exposes
    (`feature_log_prob_`/`class_log_prior_`/`feature_count_`/`class_count_`,
    sklearn/naive_bayes.py). The values are computed by the Rust fitted types
    and read off the `_Rs*` getters added in extras.rs."""

    def fit(self, X, y):
        super().fit(X, y)
        self.feature_log_prob_ = np.array(self._rs.feature_log_prob_)
        self.class_log_prior_ = np.array(self._rs.class_log_prior_)
        self.feature_count_ = np.array(self._rs.feature_count_)
        self.class_count_ = np.array(self._rs.class_count_)
        return self


class MultinomialNB(_DiscreteNBWrapper):
    def __init__(self, *, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def _make_rs(self):
        return _RsMultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior)


class BernoulliNB(_DiscreteNBWrapper):
    def __init__(self, *, alpha=1.0, fit_prior=True, binarize=0.0):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.binarize = binarize

    def _make_rs(self):
        return _RsBernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior,
                              binarize=self.binarize)


class ComplementNB(_DiscreteNBWrapper):
    def __init__(self, *, alpha=1.0, fit_prior=True, norm=False):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.norm = norm

    def _make_rs(self):
        return _RsComplementNB(alpha=self.alpha, fit_prior=self.fit_prior,
                               norm=self.norm)


class ExtraTreeClassifier(_ClassifierWrapper):
    def __init__(self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _make_rs(self):
        return _RsExtraTreeClassifier(max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf)


class ExtraTreesClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsExtraTreesClassifier(n_estimators=self.n_estimators,
                                       max_depth=self.max_depth,
                                       random_state=self.random_state)


class AdaBoostClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _make_rs(self):
        return _RsAdaBoostClassifier(n_estimators=self.n_estimators,
                                     learning_rate=self.learning_rate,
                                     random_state=self.random_state)


class GradientBoostingClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsGradientBoostingClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class HistGradientBoostingClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=100, learning_rate=0.1,
                 max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_rs(self):
        return _RsHistGradientBoostingClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, random_state=self.random_state)


class BaggingClassifier(_ClassifierWrapper):
    def __init__(self, *, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _make_rs(self):
        return _RsBaggingClassifier(n_estimators=self.n_estimators,
                                    random_state=self.random_state)


class NearestCentroid(_ClassifierWrapper):
    def __init__(self):
        """No tunable hyperparameters; defaults match sklearn."""

    def _make_rs(self):
        return _RsNearestCentroid()


# ---------------------------------------------------------------------------
# Cluster wrappers
# ---------------------------------------------------------------------------

class _ClusterWrapper(ClusterMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X))
        self.labels_ = np.asarray(self._rs.labels_)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class MiniBatchKMeans(_ClusterWrapper):
    def __init__(self, n_clusters=8, *, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _make_rs(self):
        return _RsMiniBatchKMeans(n_clusters=self.n_clusters,
                                  max_iter=self.max_iter,
                                  random_state=self.random_state)

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))


class DBSCAN(_ClusterWrapper):
    def __init__(self, eps=0.5, *, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def _make_rs(self):
        return _RsDBSCAN(eps=self.eps, min_samples=self.min_samples)


class AgglomerativeClustering(_ClusterWrapper):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def _make_rs(self):
        return _RsAgglomerativeClustering(n_clusters=self.n_clusters)


class Birch(_ClusterWrapper):
    def __init__(self, *, n_clusters=None, threshold=0.5):
        self.n_clusters = n_clusters
        self.threshold = threshold

    def _make_rs(self):
        return _RsBirch(n_clusters=self.n_clusters, threshold=self.threshold)


class GaussianMixture(BaseEstimator):
    """sklearn places GaussianMixture in `sklearn.mixture` (not cluster);
    we mirror that style — fit/predict/labels_."""

    def __init__(self, n_components=1, *, max_iter=100, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        self._rs = _RsGaussianMixture(n_components=self.n_components,
                                      max_iter=self.max_iter,
                                      random_state=self.random_state)
        self._rs.fit(_f64(X))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(self._rs.predict(_f64(X)))

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)


# ---------------------------------------------------------------------------
# Decomp / preprocess transformer wrappers
# ---------------------------------------------------------------------------

class _TransformerWrapper(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(self._rs.transform(_f64(X)))


class IncrementalPCA(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsIncrementalPCA(n_components=self.n_components)


class TruncatedSVD(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsTruncatedSVD(n_components=self.n_components)


class FastICA(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsFastICA(n_components=self.n_components)


class NMF(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsNMF(n_components=self.n_components)


class KernelPCA(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsKernelPCA(n_components=self.n_components)


class SparsePCA(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsSparsePCA(n_components=self.n_components)


class FactorAnalysis(_TransformerWrapper):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def _make_rs(self):
        return _RsFactorAnalysis(n_components=self.n_components)


class MinMaxScaler(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (feature_range=(0, 1))."""

    def _make_rs(self):
        return _RsMinMaxScaler()


class MaxAbsScaler(_TransformerWrapper):
    def __init__(self):
        """No tunable hyperparameters."""

    def _make_rs(self):
        return _RsMaxAbsScaler()


class RobustScaler(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (quantile_range=(25.0, 75.0))."""

    def _make_rs(self):
        return _RsRobustScaler()


class PowerTransformer(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (method='yeo-johnson')."""

    def _make_rs(self):
        return _RsPowerTransformer()


class Nystroem(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (kernel='rbf', n_components=100)."""

    def _make_rs(self):
        return _RsNystroem()


class RBFSampler(_TransformerWrapper):
    def __init__(self):
        """Defaults match sklearn (gamma=1.0, n_components=100)."""

    def _make_rs(self):
        return _RsRBFSampler()
