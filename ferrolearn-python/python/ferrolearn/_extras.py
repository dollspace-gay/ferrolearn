"""sklearn-compatible Python wrappers for the ~40 extended-surface estimators
bound in extras.rs. Minimal API: __init__, fit, predict or transform. Inherits
sklearn mixins for `.score()` / `fit_transform()`.
"""

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils.deprecation import _deprecate_Xt_in_inverse_transform
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import (
    _RsARDRegression,
    _RsAdaBoostClassifier,
    _RsAgglomerativeClustering,
    _RsBaggingClassifier,
    _RsBayesianRidge,
    _RsBernoulliNB,
    _RsBinarizer,
    _RsBirch,
    _RsComplementNB,
    _RsDBSCAN,
    _RsDecisionTreeRegressor,
    _RsExtraTreeClassifier,
    _RsExtraTreesClassifier,
    _RsExtraTreesRegressor,
    _RsFactorAnalysis,
    _RsFastICA,
    _RsFeatureAgglomeration,
    _RsGaussianMixture,
    _RsGradientBoostingClassifier,
    _RsGradientBoostingRegressor,
    _RsHistGradientBoostingClassifier,
    _RsHistGradientBoostingRegressor,
    _RsHuberRegressor,
    _RsIncrementalPCA,
    _RsIsolationForest,
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
    _RsNormalizer,
    _RsNystroem,
    _RsOPTICS,
    _RsOrthogonalMatchingPursuit,
    _RsPowerTransformer,
    _RsQDA,
    _RsQuantileRegressor,
    _RsRANSACRegressor,
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


class RANSACRegressor(_RegressorPickleMixin, RegressorMixin, BaseEstimator):
    """RANSAC robust regression backed by Rust (#2178).

    Mirrors ``sklearn.linear_model.RANSACRegressor``
    (``sklearn/linear_model/_ransac.py:288-315``) over the DEFAULT
    ``LinearRegression`` base (``estimator=None`` -> ``LinearRegression()``,
    ``_ransac.py:380``), surfacing the full constructor
    ``(estimator=None, *, min_samples=None, residual_threshold=None,
    max_trials=100, max_skips=np.inf, stop_n_inliers=np.inf, stop_score=np.inf,
    stop_probability=0.99, loss='absolute_error', random_state=None)`` (sklearn
    defaults/order), the ``fit(X, y)``/``predict(X)`` contract, and the fitted
    ``coef_``/``intercept_`` (from the refit base estimator, ``_ransac.py:602-605``)/
    ``inlier_mask_`` (``_ransac.py:201``)/``n_features_in_`` attributes.

    Of the constructor surface, ``min_samples``/``residual_threshold``/
    ``max_trials``/``random_state`` are threaded into the Rust core. The rest are
    validated to sklearn's ``_parameter_constraints`` (``_ransac.py:260-286``):

    - ``estimator``: ``None`` -> the default ``LinearRegression`` base. A non-None
      ``estimator`` that is NOT a ``LinearRegression`` instance raises
      ``NotImplementedError`` (the Rust binding has no estimator pluggability; only
      the default ``fit_intercept=True`` LinearRegression base is supported). A
      ``LinearRegression`` instance with non-default params (``fit_intercept=False``
      etc.) is also rejected (only sklearn's default-equivalent base is bound).
    - ``min_samples``: sklearn ``Interval(Integral, 1, None)`` or
      ``Interval(RealNotInt, 0, 1)`` or ``None`` (``_ransac.py:262-266``). Resolved
      at FIT time (we have ``X`` then): ``None`` -> ``n_features + 1``
      (``_ransac.py:388``); an int ``>= 1`` -> as-is; a float in ``[0, 1]`` ->
      ``ceil(min_samples * n_samples)`` (``_ransac.py:389-392``); out of range ->
      ``ValueError``; ``> n_samples`` -> ``ValueError`` (``_ransac.py:393-397``).
    - ``residual_threshold``: ``None`` (the core's MAD-of-y default, ransac.rs
      REQ-2) or a real ``>= 0``; negative -> ``ValueError``.
    - ``max_trials``: an int ``>= 0`` (or ``np.inf``); other values -> ``ValueError``.
    - ``loss``: only the default ``'absolute_error'`` is supported; the Rust core
      hardcodes absolute-error residuals (ransac.rs REQ-8 NOT-STARTED #516), so
      ``'squared_error'`` (which DOES change the inlier set) and any callable raise
      ``NotImplementedError``.
    - ``max_skips``/``stop_n_inliers``/``stop_score``/``stop_probability``: only the
      sklearn DEFAULTS (``inf``/``inf``/``inf``/``0.99``) are supported — the Rust
      core runs a FIXED ``max_trials`` loop with no dynamic-stop / skip tracking
      (ransac.rs REQ-7 NOT-STARTED #515); a non-default value (which would change
      the trial count / result) raises ``NotImplementedError``.

    RNG-substrate caveat (#2118): the Rust core's subset RNG is a Fisher-Yates
    ``StdRng``, NOT numpy's MT19937, so for a GIVEN ``random_state`` the drawn
    subsets differ from sklearn. On WELL-SEPARATED data (clear inliers + a few far
    outliers) the best inlier set is UNIQUE, so the final refit ``coef_``/
    ``intercept_`` converge to sklearn's regardless of RNG given sufficient
    ``max_trials``; on overlapping data the chosen consensus set (and thus
    ``coef_``) may differ from sklearn's for the same seed.

    ``n_trials_``/``n_skips_*``/``estimator_`` are NOT exposed: the Rust core does
    not track the trial count (ransac.rs REQ-7 #515) nor surface the refit base as
    a wrapped fitted type (REQ-10 #518), so none is faked.

    Pickle: the Rust fitted object is not directly picklable, so the training data
    is stored on fit and ``_rs`` is rebuilt by re-fitting on unpickle
    (``_RegressorPickleMixin``).
    """

    def __init__(self, estimator=None, *, min_samples=None,
                 residual_threshold=None, is_data_valid=None,
                 is_model_valid=None, max_trials=100, max_skips=np.inf,
                 stop_n_inliers=np.inf, stop_score=np.inf, stop_probability=0.99,
                 loss='absolute_error', random_state=None):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        # is_data_valid/is_model_valid (`_ransac.py:294-295`): per-trial callable
        # rejection hooks. The Rust core has no such hooks (REQ-11 NOT-STARTED
        # #519); held for get_params/clone parity, a non-None value is rejected in
        # `_validate`.
        self.is_data_valid = is_data_valid
        self.is_model_valid = is_model_valid
        self.max_trials = max_trials
        self.max_skips = max_skips
        self.stop_n_inliers = stop_n_inliers
        self.stop_score = stop_score
        self.stop_probability = stop_probability
        self.loss = loss
        self.random_state = random_state

    def _validate(self, n_samples, n_features):
        # Mirror sklearn's `RANSACRegressor._parameter_constraints`
        # (`_ransac.py:260-286`) + the `fit`-time resolution (`_ransac.py:377-403`),
        # Python-side, before the usize/f64 ABI boundary. Returns the resolved
        # integer `min_samples` (or None to let the core default to n_features+1).

        # estimator: None -> default LinearRegression base. A non-None estimator
        # must be a *default-equivalent* LinearRegression instance; anything else
        # has no Rust binding (no estimator pluggability) -> NotImplementedError.
        from sklearn.linear_model import LinearRegression as _SkLR
        if self.estimator is not None:
            if not isinstance(self.estimator, _SkLR):
                raise NotImplementedError(
                    "RANSACRegressor only supports the default LinearRegression "
                    "base estimator (no estimator pluggability across the Rust "
                    "ABI; NOT-STARTED)."
                )
            # Only sklearn's default-equivalent LinearRegression is bound; a
            # non-default base (fit_intercept=False / positive=True / ...) is a
            # different fit the binding cannot reproduce.
            defaults = _SkLR()
            for p in ("fit_intercept", "copy_X", "n_jobs", "positive"):
                if getattr(self.estimator, p) != getattr(defaults, p):
                    raise NotImplementedError(
                        "RANSACRegressor only supports a default-parameter "
                        f"LinearRegression base; {p}="
                        f"{getattr(self.estimator, p)!r} differs from the "
                        "default (NOT-STARTED)."
                    )

        # is_data_valid / is_model_valid: per-trial callable rejection hooks
        # (`_ransac.py:294-295`); the Rust core has no such hooks (REQ-11 #519).
        if self.is_data_valid is not None:
            raise NotImplementedError(
                "is_data_valid not supported (no per-trial data-validation hook "
                "in the Rust core; NOT-STARTED #519)."
            )
        if self.is_model_valid is not None:
            raise NotImplementedError(
                "is_model_valid not supported (no per-trial model-validation hook "
                "in the Rust core; NOT-STARTED #519)."
            )

        # loss: only the default 'absolute_error' (ransac.rs hardcodes absolute
        # residuals, REQ-8 #516). 'squared_error' / a callable change the inlier
        # set, so they are NotImplementedError.
        if self.loss != "absolute_error":
            if callable(self.loss):
                raise NotImplementedError(
                    "loss=<callable> not supported (the Rust core hardcodes "
                    "absolute-error residuals; NOT-STARTED #516)."
                )
            if self.loss == "squared_error":
                raise NotImplementedError(
                    "loss='squared_error' not supported (the Rust core hardcodes "
                    "absolute-error residuals, which select a different inlier "
                    "set; NOT-STARTED #516)."
                )
            raise ValueError(
                f"loss == {self.loss!r}, must be 'absolute_error', "
                "'squared_error', or a callable."
            )

        # max_skips / stop_*: only the sklearn DEFAULTS are supported (the Rust
        # core runs a fixed max_trials loop with no dynamic-stop / skip tracking,
        # REQ-7 #515). A non-default value changes the trial count / result.
        if not (self.max_skips == np.inf):
            raise NotImplementedError(
                "max_skips != np.inf not supported (the Rust core does not track "
                "skips; NOT-STARTED #515)."
            )
        if not (self.stop_n_inliers == np.inf):
            raise NotImplementedError(
                "stop_n_inliers != np.inf not supported (no early-stop in the "
                "Rust core; NOT-STARTED #515)."
            )
        if not (self.stop_score == np.inf):
            raise NotImplementedError(
                "stop_score != np.inf not supported (no early-stop in the Rust "
                "core; NOT-STARTED #515)."
            )
        if not (self.stop_probability == 0.99):
            raise NotImplementedError(
                "stop_probability != 0.99 not supported (no dynamic max_trials in "
                "the Rust core; NOT-STARTED #515)."
            )

        # max_trials: Interval(Integral, 0, None) or {np.inf} (`_ransac.py:270`).
        # The Rust core takes a usize; np.inf has no usize image, so reject it
        # (a fixed loop cannot run inf trials — NOT-STARTED #515).
        if self.max_trials == np.inf:
            raise NotImplementedError(
                "max_trials=np.inf not supported (the Rust core runs a fixed "
                "max_trials loop; NOT-STARTED #515)."
            )
        if (not isinstance(self.max_trials, (int, np.integer))
                or isinstance(self.max_trials, bool)
                or self.max_trials < 0):
            raise ValueError(
                f"max_trials == {self.max_trials}, must be >= 0 (an int in "
                "[0, inf)) or np.inf."
            )

        # residual_threshold: Interval(Real, 0, None) or None (`_ransac.py:267`).
        if self.residual_threshold is not None and (
            not isinstance(self.residual_threshold,
                           (int, float, np.floating, np.integer))
            or isinstance(self.residual_threshold, bool)
            or self.residual_threshold < 0
        ):
            raise ValueError(
                f"residual_threshold == {self.residual_threshold}, must be >= 0 "
                "(a real in [0, inf)) or None."
            )

        # min_samples resolution (`_ransac.py:389-397`), mirroring sklearn's
        # fit-time branches EXACTLY (verified against the live 1.5.2 oracle, #2179):
        #   None        -> n_features + 1                            (:388)
        #   0 < ms < 1  -> ceil(ms * n_samples)   (STRICT both ends) (:389-390)
        #   ms >= 1     -> int(ms) if integer-valued, else ValueError (:391-394)
        # sklearn's branches are STRICT, so the float endpoints 0.0 and 1.0 do
        # NOT take the ceil branch: 1.0 falls into the `>= 1` branch and resolves
        # to a size-1 subset, while 0.0 satisfies NEITHER branch and leaves
        # `min_samples` unbound -> the live oracle raises (mirrored here as a
        # ValueError). bool is Integral and is handled by these numeric branches
        # with NO special-casing: True (== 1) is `>= 1` and integer-valued ->
        # int(True) == 1 (ACCEPTED); False (== 0) satisfies neither branch ->
        # rejected, matching sklearn.
        if self.min_samples is None:
            min_samples = n_features + 1
        elif not isinstance(self.min_samples,
                            (int, float, np.integer, np.floating)):
            # sklearn's `_parameter_constraints` rejects non-real types (e.g. a
            # string) as InvalidParameterError (a ValueError) before fit.
            raise ValueError(
                f"min_samples == {self.min_samples!r}, must be an int >= 1, a "
                "float in (0, 1), or None."
            )
        elif 0 < self.min_samples < 1:
            min_samples = int(np.ceil(self.min_samples * n_samples))
        elif self.min_samples >= 1:
            # A non-integer real >= 1 (e.g. 2.5) is rejected by sklearn's fit-time
            # `min_samples % 1 != 0` check (`_ransac.py:393-394`).
            if self.min_samples % 1 != 0:
                raise ValueError(
                    f"min_samples == {self.min_samples}, must be an integer "
                    "value when >= 1, a float in (0, 1), or None."
                )
            min_samples = int(self.min_samples)
        else:
            # Falls through every valid branch (e.g. 0.0, 0, False, negatives):
            # below 1 with no valid branch -> error (matches the live oracle).
            raise ValueError(
                f"min_samples == {self.min_samples}, must be an int >= 1, a "
                "float in (0, 1), or None."
            )

        if min_samples > n_samples:
            raise ValueError(
                "`min_samples` may not be larger than number of samples: "
                f"n_samples = {n_samples}."
            )
        return min_samples

    def _make_rs(self, n_samples, n_features):
        min_samples = self._validate(n_samples, n_features)
        rt = (None if self.residual_threshold is None
              else float(self.residual_threshold))
        rs = (None if self.random_state is None else int(self.random_state))
        return _RsRANSACRegressor(
            min_samples=min_samples, residual_threshold=rt,
            max_trials=int(self.max_trials), random_state=rs)

    def fit(self, X, y):
        X = _f64(X)
        y = _f64(y)
        self._rs = self._make_rs(X.shape[0], X.shape[1])
        self._rs.fit(X, y)
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/linear_model/_ransac.py:201/602-605).
        self.coef_ = np.asarray(self._rs.coef_)
        self.intercept_ = self._rs.intercept_
        self.inlier_mask_ = np.asarray(self._rs.inlier_mask_)
        self._store_training_data(X, y)
        return self

    def _rebuild_rs(self):
        self._rs = self._make_rs(self._fit_X.shape[0], self._fit_X.shape[1])
        self._rs.fit(self._fit_X, self._fit_y)

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
    """DBSCAN density-based clustering backed by Rust (#2190, #2193).

    Mirrors ``sklearn.cluster.DBSCAN`` (``sklearn/cluster/_dbscan.py:345-363``),
    surfacing the constructor ``(eps=0.5, *, min_samples=5, metric='euclidean',
    p=None)`` (sklearn defaults/order), the ``fit(X, y=None, sample_weight=None)``
    signature (``_dbscan.py:370``) and the fitted ``labels_`` (``_dbscan.py:439``),
    ``core_sample_indices_`` (``np.where(core_samples)[0]``, ``_dbscan.py:438``),
    and ``components_`` (``X[core_sample_indices_].copy()``,
    ``_dbscan.py:441-446``) attributes from the Rust fitted type.

    ``sample_weight`` (``_dbscan.py:384-388``): weight of each sample, such that
    a sample with a weight of at least ``min_samples`` is by itself a core
    sample; a sample with a negative weight may inhibit its eps-neighbor from
    being core. Weights are absolute and default to 1 (the ``None`` default
    reproduces the unweighted ``len >= min_samples`` core path EXACTLY). A
    wrong-length ``sample_weight`` raises ``ValueError`` (the Rust core's
    ``ShapeMismatch`` -> ``PyValueError``, mirroring ``_check_sample_weight``).

    ``metric`` (``_dbscan.py:350``, default ``'euclidean'``) selects the neighbor
    distance metric. The supported strings mirror
    ``sklearn.metrics.pairwise._VALID_METRICS``:
    ``'euclidean'``/``'l2'``, ``'manhattan'``/``'l1'``/``'cityblock'``,
    ``'chebyshev'``, and ``'minkowski'``. An unknown metric raises ``ValueError``
    (matching sklearn's ``InvalidParameterError``). ``p`` (``_dbscan.py:354``,
    default ``None``) is the Minkowski order; it is used ONLY by
    ``metric='minkowski'`` (``p=None`` -> ``p=2``, i.e. Euclidean) and is IGNORED
    by every other metric (``_dbscan.py:411-418``, #2192) — so
    ``DBSCAN(metric='euclidean', p=3)`` is plain Euclidean. A non-positive /
    ``NaN`` Minkowski ``p`` raises ``ValueError`` (the Rust core's
    ``InvalidParameter`` -> ``PyValueError``, matching ``NearestNeighbors``'
    ``p in (0, inf]``). ``metric_params``, ``metric='precomputed'``, and callable
    metrics stay NOT-STARTED (no ferrolearn surface). ``algorithm``/``leaf_size``/
    ``n_jobs`` (neighbor-search knobs, identical result) are not exposed.
    """

    def __init__(self, eps=0.5, *, min_samples=5, metric="euclidean", p=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.p = p

    def _make_rs(self):
        return _RsDBSCAN(eps=self.eps, min_samples=self.min_samples,
                         metric=self.metric, p=self.p)

    def fit(self, X, y=None, sample_weight=None):
        self._rs = self._make_rs()
        # sklearn validates X via check_array(dtype="numeric")
        # (`_dbscan.py:395`): a 2-D array is REQUIRED (1-D -> ValueError, not
        # TypeError, #2191), and float32/float64 are PRESERVED (other dtypes
        # cast to float64) — DBSCAN does NOT upcast float32.
        Xa = np.asarray(X)
        if Xa.ndim != 2:
            raise ValueError(f"Expected 2D array, got {Xa.ndim}D array instead.")
        if Xa.dtype not in (np.float32, np.float64):
            Xa = Xa.astype(np.float64)
        # sklearn `DBSCAN.fit(X, y=None, sample_weight=None)`
        # (`_dbscan.py:370`): thread `sample_weight` into the Rust core (the
        # weighted core-determination path). `None` is the unweighted path,
        # byte-identical to today.
        if sample_weight is None:
            self._rs.fit(_f64(Xa))
        else:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if sw.ndim == 0:
                # _check_sample_weight broadcasts a scalar to all n samples.
                sw = np.full(Xa.shape[0], float(sw))
            elif sw.ndim > 1:
                # _check_sample_weight: "Sample weights must be 1D array or
                # scalar" (ValueError, not TypeError, #2191).
                raise ValueError("Sample weights must be 1D array or scalar")
            self._rs.fit(_f64(Xa), sample_weight=sw)
        self.labels_ = np.asarray(self._rs.labels_)
        self.core_sample_indices_ = np.asarray(self._rs.core_sample_indices_)
        # sklearn `self.components_ = X[self.core_sample_indices_].copy()`
        # (`_dbscan.py:442`) — taken from the VALIDATED input, preserving its
        # dtype (float32 stays float32), NOT the f64 Rust core copy (#2191).
        self.components_ = Xa[self.core_sample_indices_].copy()
        self.n_features_in_ = Xa.shape[1]
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        # sklearn `DBSCAN.fit_predict(X, y=None, sample_weight=None)`
        # (`_dbscan.py:450`) threads sample_weight through fit.
        return self.fit(X, sample_weight=sample_weight).labels_


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


class OPTICS(_ClusterWrapper):
    """OPTICS density-based clustering backed by Rust (#1090, REQ-12).

    Mirrors ``sklearn.cluster.OPTICS`` (``sklearn/cluster/_optics.py:266-297``):
    Ordering Points To Identify the Clustering Structure. Computes a reachability
    ordering of the data and extracts ``labels_`` via the Xi-steep method (the
    default) or a DBSCAN-like cut at a fixed ``eps``. Exposes the keyword-only
    constructor ``(*, min_samples=5, max_eps=np.inf, metric='minkowski', p=2,
    metric_params=None, cluster_method='xi', eps=None, xi=0.05,
    predecessor_correction=True, min_cluster_size=None, algorithm='auto',
    leaf_size=30, memory=None, n_jobs=None)`` (sklearn defaults/order), the
    ``fit(X, y=None)`` contract, the inherited ``fit_predict``, and the fitted
    attributes ``labels_``/``reachability_``/``ordering_``/``core_distances_``/
    ``predecessor_``/``cluster_hierarchy_``/``n_features_in_``
    (``_optics.py:175-209``).

    Of the constructor surface, ``min_samples``/``max_eps``/``xi``/
    ``min_cluster_size``/``cluster_method``/``eps``/``predecessor_correction`` are
    threaded into the Rust core (``ferrolearn_cluster::OPTICS<f64>``, optics.rs
    REQ-1..11 SHIPPED, bit-exact vs the live sklearn 1.5.2 oracle):

    - ``cluster_method`` (``_optics.py:274``, ``StrOptions({"xi","dbscan"})``):
      ``'xi'`` (default) runs the automatic Xi-steep extraction; ``'dbscan'`` runs
      a DBSCAN-like cut at ``eps`` (assuming ``max_eps`` when ``eps is None``,
      ``_optics.py:375-378``). An unknown string raises ``ValueError`` (sklearn's
      ``InvalidParameterError`` is a ``ValueError`` subclass).
    - ``min_cluster_size`` (``_optics.py:278``): ``None`` (default) means use
      ``min_samples``; an int is the minimum number of samples in a Xi cluster
      (``_optics.py:902-903``).
    - ``eps`` (``_optics.py:275``): the DBSCAN cut radius, used ONLY by
      ``cluster_method='dbscan'``; ``eps > max_eps`` raises ``ValueError``
      (``_optics.py:380-383``).

    The Rust OPTICS core supports only the Euclidean (``metric='minkowski', p=2``)
    brute-force path (optics.rs ``core_distance``/``get_neighbors`` compute
    Euclidean distances), so the params that would CHANGE the result — a
    non-Euclidean ``metric``/``p``, ``metric_params``, a precomputed/callable
    ``metric``, an ``algorithm`` other than ``'auto'``/``'brute'`` — raise
    ``NotImplementedError`` when set NON-default, honestly surfacing the gap rather
    than silently mis-resolving (optics.rs REQ-10 NOT-STARTED, #1088). A FLOAT
    ``min_samples`` (sklearn accepts a ``(0, 1]`` fraction, ``_optics.py:245``) is
    likewise REQ-10 NOT-STARTED: it raises ``NotImplementedError`` rather than
    truncating. The no-op knobs ``leaf_size``/``memory``/``n_jobs`` (neighbor-search
    / caching / threading; identical result) are accepted-and-ignored (held only
    for ``get_params``/``clone`` round-trip).
    """

    def __init__(self, *, min_samples=5, max_eps=np.inf, metric="minkowski",
                 p=2, metric_params=None, cluster_method="xi", eps=None,
                 xi=0.05, predecessor_correction=True, min_cluster_size=None,
                 algorithm="auto", leaf_size=30, memory=None, n_jobs=None):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.cluster_method = cluster_method
        self.eps = eps
        self.xi = xi
        self.predecessor_correction = predecessor_correction
        self.min_cluster_size = min_cluster_size
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.memory = memory
        self.n_jobs = n_jobs

    def _validate_unsupported(self):
        # The Rust OPTICS core computes Euclidean (minkowski p=2) distances by
        # brute force over the full sample set. Params that would CHANGE the
        # reachability graph raise NotImplementedError when non-default; the no-op
        # knobs (leaf_size, memory, n_jobs) are accepted silently. REQ-10 #1088.
        if self.metric != "minkowski":
            raise NotImplementedError(
                f"metric == {self.metric!r} not supported (the Rust OPTICS core "
                "computes only Euclidean = minkowski(p=2) distances; "
                "REQ-10 NOT-STARTED #1088)."
            )
        if self.p != 2:
            raise NotImplementedError(
                f"p == {self.p!r} not supported (the Rust OPTICS core computes "
                "only minkowski(p=2) = Euclidean distances; REQ-10 NOT-STARTED "
                "#1088)."
            )
        if self.metric_params is not None:
            raise NotImplementedError(
                "metric_params != None not supported (the Rust OPTICS core "
                "computes only Euclidean distances; REQ-10 NOT-STARTED #1088)."
            )
        if self.algorithm not in ("auto", "brute"):
            raise NotImplementedError(
                f"algorithm == {self.algorithm!r} not supported (the Rust OPTICS "
                "core uses a brute-force neighbor search; pass 'auto'/'brute'; "
                "REQ-10 NOT-STARTED #1088)."
            )
        # A float min_samples is sklearn's (0, 1] fraction (_optics.py:245); the
        # Rust core takes a usize, so reject the float branch (REQ-10 #1088)
        # rather than silently truncating. A bool is an int subclass in Python; we
        # accept genuine ints (and bool, though min_samples<2 then errors).
        if isinstance(self.min_samples, float):
            raise NotImplementedError(
                "a float min_samples (the (0, 1] fraction of n_samples, "
                "sklearn/cluster/_optics.py:245) is not supported (the Rust "
                "OPTICS core takes an integer min_samples; REQ-10 NOT-STARTED "
                "#1088)."
            )
        if self.min_cluster_size is not None and isinstance(
            self.min_cluster_size, float
        ):
            raise NotImplementedError(
                "a float min_cluster_size (the (0, 1] fraction of n_samples, "
                "sklearn/cluster/_optics.py:906) is not supported (the Rust "
                "OPTICS core takes an integer min_cluster_size; REQ-10 "
                "NOT-STARTED #1088)."
            )
        # An INTEGER min_cluster_size must be >= 2 (sklearn
        # `Interval(Integral, 2, None, closed="left")`, `_optics.py:255-256` ->
        # InvalidParameterError). ferrolearn previously threaded an int < 2
        # straight to the core and fit silently (#2199). (bool is an int
        # subclass; True==1/False==0 are likewise rejected, matching sklearn.)
        if (
            self.min_cluster_size is not None
            and not isinstance(self.min_cluster_size, float)
            and self.min_cluster_size < 2
        ):
            raise ValueError(
                "The 'min_cluster_size' parameter of OPTICS must be an int in "
                "the range [2, inf) or a float in the range (0, 1). Got "
                f"{self.min_cluster_size!r} instead."
            )

    def _make_rs(self):
        self._validate_unsupported()
        return _RsOPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            xi=self.xi,
            min_cluster_size=self.min_cluster_size,
            cluster_method=self.cluster_method,
            eps=self.eps,
            predecessor_correction=self.predecessor_correction,
        )

    def fit(self, X, y=None):
        # sklearn `OPTICS.fit(X, y=None)` (`_optics.py:303`): `y` ignored. The
        # supported-param gate runs in `_make_rs` BEFORE the ABI; the Rust core
        # validates min_samples/max_eps/xi/eps bounds (FerroError -> ValueError).
        self._rs = self._make_rs()
        self._rs.fit(_f64(X))
        # Fitted attrs (`_optics.py:175-209`), indexed by original sample order
        # (`ordering_` lists the reachability-plot order).
        self.labels_ = np.asarray(self._rs.labels_)
        self.reachability_ = np.asarray(self._rs.reachability_)
        self.ordering_ = np.asarray(self._rs.ordering_)
        self.core_distances_ = np.asarray(self._rs.core_distances_)
        self.predecessor_ = np.asarray(self._rs.predecessor_)
        # sklearn only sets `cluster_hierarchy_` on the Xi branch (`_optics.py:373`);
        # the dbscan branch leaves the attribute UNSET (the Rust core returns an
        # empty (0, 2) array there). Mirror sklearn's `hasattr` contract: expose
        # the attr ONLY for the Xi method.
        if self.cluster_method == "xi":
            self.cluster_hierarchy_ = np.asarray(self._rs.cluster_hierarchy_)
        self.n_features_in_ = X.shape[1]
        return self


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
# Ensemble outlier-detection wrappers
# ---------------------------------------------------------------------------

class IsolationForest(_RegressorPickleMixin, OutlierMixin, BaseEstimator):
    """Isolation Forest anomaly detector backed by Rust (#2180).

    Mirrors ``sklearn.ensemble.IsolationForest``
    (``sklearn/ensemble/_iforest.py:221-248``), surfacing the keyword-only
    constructor ``(n_estimators=100, max_samples='auto', contamination='auto',
    max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0,
    warm_start=False)`` (sklearn defaults/order), the ``fit(X, y=None)`` contract,
    and the ``predict``/``score_samples``/``decision_function``/``fit_predict``
    ``OutlierMixin`` method surface plus the fitted ``offset_``/``n_features_in_``
    attributes from the Rust fitted type.

    Of the constructor surface, ``n_estimators``/``max_samples``/``contamination``/
    ``random_state`` are threaded into the Rust core. ``max_samples`` and
    ``contamination`` are resolved to sklearn's effective values BEFORE the Rust
    ABI (which takes an integer ``max_samples`` and an ``'auto'``-or-float
    contamination):

    - ``max_samples`` (sklearn ``_iforest.py:303-318``): ``'auto'`` →
      ``min(256, n_samples)``; an int → as-is (the core re-clamps to
      ``n_samples``, matching sklearn's warn-and-clamp ``:307-314``); a float in
      ``(0, 1]`` → ``int(max_samples * n_samples)`` (sklearn truncates, ``:318``).
    - ``contamination`` (sklearn ``_iforest.py:341-353``): ``'auto'`` → the
      ``offset_ = -0.5`` paper path; a float in ``(0, 0.5]`` → the
      ``offset_ = percentile`` path.

    Validation mirrors sklearn's ``_parameter_constraints``
    (``_iforest.py:199-219``), raising ``ValueError`` (sklearn's
    ``InvalidParameterError`` ⊂ ``ValueError``) for ``n_estimators < 1``, a
    ``max_samples`` int ``< 1`` / float outside ``(0, 1]`` / invalid string, or a
    ``contamination`` float outside ``(0, 0.5]``.

    The Rust core implements only the full-feature, with-replacement subsample
    path (isolation_forest.rs REQ-7a/7b NOT-STARTED #728/#729), so the
    non-default ``max_features != 1.0`` / ``bootstrap=True`` / ``warm_start=True``
    raise ``NotImplementedError``; ``n_jobs`` (a threading knob) and ``verbose`` (a
    diagnostics knob) are accepted-and-ignored (held only for ``get_params``/
    ``clone`` round-trip).

    RNG-substrate caveat (#2118 / isolation_forest.rs RNG-boundary #730): the Rust
    core's subsample + split draws use ``StdRng``, NOT numpy's MT19937, so for a
    GIVEN ``random_state`` the trees — and thus the exact ``score_samples`` values
    — DIFFER from sklearn. The STRUCTURAL contract holds and matches sklearn:
    ``predict ∈ {-1, +1}``, ``score_samples ∈ [-1, 0]``,
    ``decision_function == score_samples - offset_``,
    ``predict == where(decision_function < 0, -1, 1)``, ``offset_ == -0.5`` for
    ``contamination='auto'``, and the anomaly ranking (injected far outliers score
    lower than inliers). Exact score parity is the RNG-substrate limitation, not a
    binding bug.

    Pickle: the Rust fitted object is not directly picklable, so the training data
    is stored on fit and ``_rs`` is rebuilt by re-fitting on unpickle
    (``_RegressorPickleMixin``).
    """

    def __init__(self, *, n_estimators=100, max_samples="auto",
                 contamination="auto", max_features=1.0, bootstrap=False,
                 n_jobs=None, random_state=None, verbose=0, warm_start=False):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

    def _validate_static(self):
        # The n_samples-independent half of sklearn's `_parameter_constraints`
        # (`_iforest.py:199-219`), runnable before we have X. n_estimators and
        # contamination, plus the unsupported-knob NotImplementedError gates.
        # n_estimators: Interval(Integral, 1, None, closed='left') (`:200`).
        # Python `bool` is `numbers.Integral`, so sklearn accepts True (== 1 >= 1,
        # fits 1 estimator) and rejects False (== 0 < 1) via the >= 1 bound; mirror
        # that by NOT special-casing bool.
        if (not isinstance(self.n_estimators, (int, np.integer))
                or self.n_estimators < 1):
            raise ValueError(
                f"n_estimators == {self.n_estimators}, must be >= 1 "
                "(an int in [1, inf))."
            )
        # contamination: StrOptions({'auto'}) or Interval(Real, 0, 0.5,
        # closed='right') (`:206-209`).
        if isinstance(self.contamination, str):
            if self.contamination != "auto":
                raise ValueError(
                    f"contamination == {self.contamination!r}, must be 'auto' or "
                    "a float in (0, 0.5]."
                )
        elif (not isinstance(self.contamination,
                             (int, float, np.floating, np.integer))
              or isinstance(self.contamination, bool)
              or not (0 < self.contamination <= 0.5)):
            raise ValueError(
                f"contamination == {self.contamination}, must be 'auto' or a "
                "float in (0, 0.5]."
            )
        # max_features: only the default 1.0 (full feature set) is supported; the
        # Rust core always considers every feature (isolation_forest.rs REQ-7a
        # NOT-STARTED #728).
        if self.max_features != 1.0:
            raise NotImplementedError(
                "max_features != 1.0 not supported (the Rust IsolationForest core "
                "always considers the full feature set; NOT-STARTED #728)."
            )
        # bootstrap: only the default False (subsample WITHOUT replacement is the
        # core's only path; bootstrap=True would draw WITH replacement,
        # isolation_forest.rs REQ-7b NOT-STARTED #729).
        if self.bootstrap:
            raise NotImplementedError(
                "bootstrap=True not supported (the Rust IsolationForest core does "
                "not implement bootstrap resampling; NOT-STARTED #729)."
            )
        # warm_start: only the default False (no incremental tree-adding path in
        # the Rust core).
        if self.warm_start:
            raise NotImplementedError(
                "warm_start=True not supported (the Rust IsolationForest core does "
                "not implement incremental tree-adding; NOT-STARTED #728)."
            )

    def _resolve_max_samples(self, n_samples):
        # sklearn `_iforest.py:303-318`: resolve `max_samples` to an int given
        # n_samples. 'auto' -> min(256, n); int -> as-is (the core re-clamps to
        # n_samples, matching sklearn's warn-and-clamp :307-314); float in (0, 1]
        # -> int(f * n) (sklearn truncates, :318).
        ms = self.max_samples
        if isinstance(ms, str):
            if ms != "auto":
                raise ValueError(
                    f"max_samples == {ms!r}, must be 'auto', an int >= 1, or a "
                    "float in (0, 1]."
                )
            return min(256, n_samples)
        # bool is `numbers.Integral` in sklearn, so True falls through the Integral
        # branch (True == 1 >= 1 -> max_samples_ == 1) and False (== 0 < 1) is
        # rejected by that branch's >= 1 bound. Do NOT special-case bool.
        if isinstance(ms, (int, np.integer)):
            # Interval(Integral, 1, None, closed='left') (`_iforest.py:203`).
            if ms < 1:
                raise ValueError(
                    f"max_samples == {ms}, must be >= 1 (an int in [1, inf)) when "
                    "an integer."
                )
            return int(ms)
        if isinstance(ms, (float, np.floating)):
            # Interval(RealNotInt, 0, 1, closed='right') (`_iforest.py:204`):
            # a float must be in (0, 1].
            if not (0 < ms <= 1):
                raise ValueError(
                    f"max_samples == {ms}, must be in (0, 1] when a float."
                )
            # sklearn truncates: int(max_samples * n_samples) (`:318`), with NO
            # guard. A tiny float that truncates to 0 (e.g. 0.01 * 45 -> 0) makes
            # sklearn's fit raise ValueError (zero-size subsample); mirror that
            # observable raise instead of clamping up to 1.
            resolved = int(ms * n_samples)
            if resolved < 1:
                raise ValueError(
                    f"max_samples == {ms} resolves to int({ms} * {n_samples}) == "
                    f"{resolved}, which is an invalid (zero-size) subsample."
                )
            return resolved
        raise ValueError(
            f"max_samples == {ms!r}, must be 'auto', an int >= 1, or a float in "
            "(0, 1]."
        )

    def _make_rs(self, n_samples):
        self._validate_static()
        max_samples = self._resolve_max_samples(n_samples)
        # contamination: None == sklearn 'auto' across the ABI; a float takes the
        # percentile path.
        contamination = (None if isinstance(self.contamination, str)
                         else float(self.contamination))
        rs = (None if self.random_state is None else int(self.random_state))
        return _RsIsolationForest(
            n_estimators=self.n_estimators, max_samples=max_samples,
            contamination=contamination, random_state=rs)

    def fit(self, X, y=None, sample_weight=None):
        X = _f64(X)
        self._rs = self._make_rs(X.shape[0])
        self._rs.fit(X)
        self.n_features_in_ = X.shape[1]
        # Fitted attributes surfaced from the Rust fitted type
        # (sklearn/ensemble/_iforest.py:344/353).
        self.offset_ = self._rs.offset_
        # _store_training_data expects (X, y); y is ignored by IsolationForest, so
        # store a placeholder for the pickle-refit path.
        self._fit_X = X.copy()
        self._fit_y = None
        return self

    def _rebuild_rs(self):
        self._rs = self._make_rs(self._fit_X.shape[0])
        self._rs.fit(self._fit_X)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_fit_X"):
            self._rebuild_rs()

    def predict(self, X):
        check_is_fitted(self)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.predict(_f64(X)))

    def score_samples(self, X):
        check_is_fitted(self)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.score_samples(_f64(X)))

    def decision_function(self, X):
        check_is_fitted(self)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.decision_function(_f64(X)))

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


class FeatureAgglomeration(_TransformerWrapper):
    """Feature Agglomeration backed by Rust (#943).

    Mirrors ``sklearn.cluster.FeatureAgglomeration``
    (``sklearn/cluster/_agglomerative.py:1296-1346`` +
    ``sklearn/cluster/_feature_agglomeration.py:22-92``): hierarchical clustering
    of the FEATURES (columns) followed by per-cluster pooling, exposing the
    ``TransformerMixin`` ``fit``/``transform``/``fit_transform`` surface plus
    ``inverse_transform`` and the dendrogram fitted attributes
    ``labels_``/``n_clusters_``/``children_``/``distances_``/``n_leaves_``/
    ``n_connected_components_`` delegated from the inner ``AgglomerativeClustering``
    over ``X.T`` (``_agglomerative.py:1338-1340``).

    The keyword-only constructor mirrors sklearn's signature
    ``(n_clusters=2, *, metric=None, memory=None, connectivity=None,
    compute_full_tree='auto', linkage='ward', pooling_func=np.mean,
    distance_threshold=None, compute_distances=False)`` (``_agglomerative.py:1296``).
    Of these, ``n_clusters``/``linkage``/``pooling_func``/``compute_distances`` are
    threaded into the Rust core:

    - ``linkage`` (``_agglomerative.py:1290``, ``StrOptions(_TREE_BUILDERS)``):
      one of ``'ward'``/``'complete'``/``'average'``/``'single'`` (default
      ``'ward'``). The Rust core supports all four (feature_agglomeration.rs
      ``map_linkage``).
    - ``pooling_func`` (``_agglomerative.py:1291``, ``[callable]``, default
      ``np.mean``): the Rust core offers only the closed
      ``PoolingFunc::{Mean, Max}`` enum, so this wrapper accepts the strings
      ``'mean'``/``'max'`` OR the numpy callables ``np.mean``/``np.max`` and maps
      them to the enum. ANY OTHER callable (an arbitrary pooling function, which
      sklearn permits) raises ``NotImplementedError`` (feature_agglomeration.rs
      REQ-7 NOT-STARTED #941); a non-callable / unknown string raises
      ``ValueError`` (sklearn ``InvalidParameterError ⊂ ValueError``).
    - ``compute_distances`` (``_agglomerative.py:1307/1319``, default ``False``):
      when ``True``, ``distances_`` is computed; else it is ``None``.

    The params the Rust core does NOT support — ``metric``/``memory``/
    ``connectivity``/``compute_full_tree``/``distance_threshold`` — are accepted
    (held for ``get_params``/``clone`` round-trip) but raise ``NotImplementedError``
    if set to a NON-default value, honestly surfacing the gap rather than silently
    ignoring it (esp. ``distance_threshold``, which would change the cut, REQ-6
    NOT-STARTED #941).
    """

    def __init__(self, n_clusters=2, *, metric=None, memory=None,
                 connectivity=None, compute_full_tree="auto", linkage="ward",
                 pooling_func=np.mean, distance_threshold=None,
                 compute_distances=False):
        self.n_clusters = n_clusters
        self.metric = metric
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.pooling_func = pooling_func
        self.distance_threshold = distance_threshold
        self.compute_distances = compute_distances

    def _resolve_pooling(self):
        # Map the sklearn `pooling_func` (string or numpy callable) onto the Rust
        # core's closed PoolingFunc enum (passed as a string across the ABI).
        pf = self.pooling_func
        if isinstance(pf, str):
            if pf not in ("mean", "max"):
                raise ValueError(
                    f"pooling_func == {pf!r}, must be 'mean'/np.mean or "
                    "'max'/np.max (the Rust FeatureAgglomeration core supports "
                    "only mean/max pooling; arbitrary callables NOT-STARTED #941)."
                )
            return pf
        if pf is np.mean:
            return "mean"
        if pf is np.max:
            return "max"
        if callable(pf):
            raise NotImplementedError(
                "pooling_func as an arbitrary callable is not supported (the Rust "
                "FeatureAgglomeration core offers only PoolingFunc::{Mean, Max}; "
                "pass 'mean'/np.mean or 'max'/np.max; REQ-7 NOT-STARTED #941)."
            )
        raise ValueError(
            f"pooling_func == {pf!r}, must be 'mean'/np.mean or 'max'/np.max."
        )

    def _validate_unsupported(self):
        # The Rust core implements only the unstructured, euclidean, n_clusters
        # cut. Params that would CHANGE that result raise NotImplementedError when
        # non-default; the no-op knobs (metric default, memory, compute_full_tree
        # 'auto') are accepted silently.
        if self.metric is not None:
            raise NotImplementedError(
                "metric != None not supported (the Rust FeatureAgglomeration core "
                "uses the euclidean/ward default only; REQ-6 NOT-STARTED #941)."
            )
        if self.connectivity is not None:
            raise NotImplementedError(
                "connectivity != None not supported (the Rust FeatureAgglomeration "
                "core implements only the unstructured path; REQ-6 NOT-STARTED "
                "#941)."
            )
        if self.distance_threshold is not None:
            raise NotImplementedError(
                "distance_threshold != None not supported (the Rust "
                "FeatureAgglomeration core cuts by n_clusters only; "
                "REQ-6 NOT-STARTED #941)."
            )

    def _make_rs(self):
        self._validate_unsupported()
        return _RsFeatureAgglomeration(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            pooling=self._resolve_pooling(),
            compute_distances=self.compute_distances,
        )

    def fit(self, X, y=None):
        self._rs = self._make_rs()
        self._rs.fit(_f64(X))
        self.n_features_in_ = X.shape[1]
        # Dendrogram fitted attributes delegated from the inner clustering over
        # X.T (sklearn/cluster/_agglomerative.py:1083-1095/1339).
        self.labels_ = np.asarray(self._rs.labels_)
        self.n_clusters_ = self._rs.n_clusters_
        self.children_ = np.asarray(self._rs.children_)
        self.distances_ = (
            None if self._rs.distances_ is None
            else np.asarray(self._rs.distances_)
        )
        self.n_leaves_ = self._rs.n_leaves_
        self.n_connected_components_ = self._rs.n_connected_components_
        return self

    def inverse_transform(self, X=None, *, Xt=None):
        # sklearn FeatureAgglomeration.inverse_transform(self, X=None, *, Xt=None)
        # routes the deprecated `Xt=` kwarg through `_deprecate_Xt_in_inverse_transform`
        # (FutureWarning), and the canonical positional param is named `X`
        # (_feature_agglomeration.py:66-87, #2188).
        X = _deprecate_Xt_in_inverse_transform(X, Xt)
        check_is_fitted(self)
        return np.asarray(self._rs.inverse_transform(_f64(X)))


class Binarizer(_TransformerWrapper):
    """Binarize features to 0/1 by a threshold, backed by Rust (#1131).

    Mirrors ``sklearn.preprocessing.Binarizer``
    (``sklearn/preprocessing/_data.py:2177-2306``): values strictly greater than
    ``threshold`` map to 1, all others to 0 (default ``threshold=0.0`` -> only
    positive values map to 1). The keyword-only constructor mirrors sklearn's
    ``__init__(self, *, threshold=0.0, copy=True)`` (``_data.py:2253``); both
    ``threshold`` and ``copy`` are threaded into the Rust core via ``_make_rs`` so
    ``get_params``/``set_params``/``clone`` round-trip them.

    ``copy`` is an accept-and-document no-op (ferrolearn's ``Transform`` always
    returns a freshly allocated array, so ``copy=False`` produces identical
    output; binarizer.rs REQ-2). A non-finite ``threshold`` is ACCEPTED at ``fit``
    (sklearn ``_parameter_constraints {threshold: [Real]}`` is a bare type check,
    binarizer.rs #2209) but REJECTED at ``transform`` (sklearn's free ``binarize``
    ``@validate_params`` open interval ``(-inf, inf)``, ``_data.py:2114``, #2208),
    surfacing as ``ValueError`` from the Rust core — matching sklearn. NaN/+-inf
    INPUT also raises ``ValueError`` (``check_array(force_all_finite=True)``,
    binarizer.rs REQ-9). Inherits ``fit``/``transform``/``fit_transform`` from
    ``_TransformerWrapper``.
    """

    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def _make_rs(self):
        return _RsBinarizer(threshold=self.threshold, copy=self.copy)

    def transform(self, X):
        # sklearn Binarizer is STATELESS (`_more_tags() -> {"stateless": True}`,
        # `_data.py:2308`; "does not need to be fitted"): `transform` works
        # WITHOUT a prior `fit` (#2213). Build the Rust core on demand if `fit`
        # was never called — `binarize` reads only `threshold`, no fitted state.
        if not hasattr(self, "_rs"):
            self._rs = self._make_rs()
            self._rs.fit(_f64(X))
        out = np.asarray(self._rs.transform(_f64(X)))
        # sklearn `binarize` -> `check_array(dtype="numeric")` PRESERVES the input
        # dtype (int64->int64, float32->float32; `_data.py:2160`), but the f64
        # Rust ABI always returns float64. Cast back (#2214) — 0.0/1.0 are exact
        # in every numeric dtype, so the values are unchanged.
        in_dtype = np.asarray(X).dtype
        if np.issubdtype(in_dtype, np.number) and out.dtype != in_dtype:
            out = out.astype(in_dtype, copy=False)
        return out


class Normalizer(_TransformerWrapper):
    """Normalize samples (rows) to unit norm, backed by Rust (#1146).

    Mirrors ``sklearn.preprocessing.Normalizer``
    (``sklearn/preprocessing/_data.py:1980-2110``): each sample (row) with at
    least one non-zero component is scaled so its chosen ``norm`` (l1/l2/max)
    equals 1; a zero-norm row is left unchanged. The constructor mirrors sklearn's
    signature ``__init__(self, norm="l2", *, copy=True)`` (``_data.py:2058``) —
    ``norm`` is POSITIONAL-OR-KEYWORD, ``copy`` is keyword-only. Both are threaded
    into the Rust core via ``_make_rs`` so ``get_params``/``set_params``/``clone``
    round-trip them.

    ``copy`` is an accept-and-document no-op (ferrolearn's ``Transform`` always
    returns a freshly allocated array, so ``copy=False`` produces identical
    output; normalizer.rs REQ-5). A bad ``norm`` string (anything other than
    ``'l1'``/``'l2'``/``'max'``) raises ``ValueError`` (sklearn
    ``_parameter_constraints {norm: StrOptions({"l1","l2","max"})}``,
    ``_data.py:2055``, an ``InvalidParameterError`` ⊂ ValueError), surfaced from
    the Rust core's ``fit``. NaN/+-inf INPUT also raises ``ValueError``
    (``check_array(force_all_finite=True)``, normalizer.rs REQ-2/REQ-3).

    sklearn is STATELESS (``_more_tags() -> {"stateless": True}``,
    ``_data.py:2110``): ``transform`` works WITHOUT a prior ``fit`` (#2213). The
    overridden ``transform`` below builds the Rust core on demand.
    """

    def __init__(self, norm="l2", *, copy=True):
        self.norm = norm
        self.copy = copy

    def _make_rs(self):
        return _RsNormalizer(norm=self.norm, copy=self.copy)

    def transform(self, X):
        # sklearn Normalizer is STATELESS (`_more_tags() -> {"stateless": True}`,
        # `_data.py:2110`; "does not need to be fitted"): `transform` works
        # WITHOUT a prior `fit` (#2213). Build the Rust core on demand if `fit`
        # was never called — `normalize` reads only `norm`, no fitted state. The
        # on-demand `fit` also surfaces the bad-`norm`-string ValueError, matching
        # sklearn's parameter validation.
        if not hasattr(self, "_rs"):
            self._rs = self._make_rs()
            self._rs.fit(_f64(X))
        out = np.asarray(self._rs.transform(_f64(X)))
        # sklearn `normalize` -> `check_array(dtype=FLOAT_DTYPES)` (`_data.py:2104`
        # -> `_data.py:1933`): a FLOATING input dtype is PRESERVED (float32->float32,
        # float64->float64), but an INTEGER input is UPCAST to float64 (NOT
        # int-preserved, unlike Binarizer #2214). The f64 Rust ABI always returns
        # float64, so cast back ONLY when the input was a floating dtype; leave
        # integer input as float64.
        in_dtype = np.asarray(X).dtype
        if np.issubdtype(in_dtype, np.floating) and out.dtype != in_dtype:
            out = out.astype(in_dtype, copy=False)
        return out


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
