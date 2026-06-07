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
    _RsLinearSVC,
    _RsMaxAbsScaler,
    _RsMinMaxScaler,
    _RsMiniBatchKMeans,
    _RsMultinomialNB,
    _RsNMF,
    _RsNearestCentroid,
    _RsNystroem,
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


class ARDRegression(_RegressorWrapper):
    def __init__(self, *, max_iter=300, tol=1e-3, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsARDRegression(max_iter=self.max_iter, tol=self.tol,
                                fit_intercept=self.fit_intercept)


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
    def __init__(self, *, quantile=0.5, alpha=1.0, max_iter=10000,
                 tol=1e-6, fit_intercept=True):
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _make_rs(self):
        return _RsQuantileRegressor(quantile=self.quantile, alpha=self.alpha,
                                    max_iter=self.max_iter, tol=self.tol,
                                    fit_intercept=self.fit_intercept)


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
