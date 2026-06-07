"""sklearn-compatible wrappers for ferrolearn classification models."""

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import (
    _RsDecisionTreeClassifier,
    _RsGaussianNB,
    _RsKNeighborsClassifier,
    _RsLogisticRegression,
    _RsRandomForestClassifier,
)


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _encode_labels(y):
    """Encode arbitrary labels to contiguous integers and return the mapping."""
    classes = np.unique(y)
    label_map = {c: i for i, c in enumerate(classes)}
    y_encoded = np.array([label_map[v] for v in y], dtype=np.int64)
    return y_encoded, classes


def _decode_labels(y_encoded, classes):
    """Decode integer labels back to original labels."""
    return classes[y_encoded]


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


def _check_classification_target(y):
    """Raise ValueError if y is continuous (not a classification target)."""
    y_type = type_of_target(y)
    if y_type in ("continuous", "continuous-multioutput"):
        raise ValueError(
            f"Unknown label type: {y_type!r}. Maybe you are trying to fit a "
            "classifier on a regression target with continuous values."
        )


class _ClassifierPickleMixin:
    """Mixin for pickling classifiers. Stores training data for re-fit on unpickle."""

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reconstruct _rs by re-fitting if we have stored training data
        if hasattr(self, "_fit_X") and hasattr(self, "_fit_y_encoded"):
            self._rebuild_rs()

    def _store_training_data(self, X, y_encoded):
        """Store training data for pickle reconstruction."""
        self._fit_X = X.copy()
        self._fit_y_encoded = y_encoded.copy()


class LogisticRegression(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Logistic Regression backed by Rust.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, *, C=1.0, max_iter=100, tol=1e-4, fit_intercept=True):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = self._validate_data(X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsLogisticRegression(
            c=float(self.C),
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        _fit_rust(self._rs, X, y_encoded)
        self.coef_ = np.array(self._rs.coef_).reshape(1, -1)
        self.intercept_ = np.array([float(self._rs.intercept_)])
        self.n_iter_ = self.max_iter
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsLogisticRegression(
            c=float(self.C),
            max_iter=self.max_iter,
            tol=self.tol,
            fit_intercept=self.fit_intercept,
        )
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))

    def predict_log_proba(self, X):
        # sklearn LogisticRegression.predict_log_proba is log_softmax, i.e.
        # log(predict_proba) (sklearn/linear_model/_logistic.py).
        return np.log(self.predict_proba(X))


class DecisionTreeClassifier(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Decision Tree Classifier backed by Rust.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    """

    def __init__(
        self, *, max_depth=None, min_samples_split=2, min_samples_leaf=1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X, y = self._validate_data(X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsDecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        _fit_rust(self._rs, X, y_encoded)
        self.feature_importances_ = np.array(self._rs.feature_importances_)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsDecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))

    def predict_log_proba(self, X):
        # sklearn DecisionTreeClassifier.predict_log_proba returns
        # np.log(predict_proba) (sklearn/tree/_classes.py).
        return np.log(self.predict_proba(X))


class RandomForestClassifier(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Random Forest Classifier backed by Rust.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of the trees.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=100,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X, y = self._validate_data(X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        _fit_rust(self._rs, X, y_encoded)
        self.feature_importances_ = np.array(self._rs.feature_importances_)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))


class KNeighborsClassifier(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """K-Nearest Neighbors Classifier backed by Rust.

    Mirrors ``sklearn.neighbors.KNeighborsClassifier``
    (``sklearn/neighbors/_classification.py:193``). ``n_neighbors`` is
    positional-or-keyword; the rest are keyword-only.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction. 'uniform' weights all neighbors
        equally; 'distance' weights by the inverse of distance. Callable
        weights are NOT supported (NotImplementedError, #876).
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors. This is a search
        strategy only — all values give identical predict/predict_proba.
        'ball_tree' maps to 'auto' (no distinct classifier variant).
    leaf_size : int, default=30
        Leaf size passed to the tree index. ABI-only: affects tree-build
        performance, not the result.
    p : float, default=2
        Power parameter for the Minkowski metric. Only ``p=2`` (Euclidean) is
        supported; any other value raises NotImplementedError (#876).
    metric : str, default='minkowski'
        Distance metric. Only 'minkowski' (with p=2) and 'euclidean' are
        supported; any other value raises NotImplementedError (#876).
    metric_params : dict or None, default=None
        Additional metric kwargs. Only ``None`` is supported (the Rust core has
        no custom metric); a non-None value raises NotImplementedError (#876).
    n_jobs : int or None, default=None
        Number of parallel jobs. ABI-only: a threading knob, does not affect
        the result.
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

    def fit(self, X, y):
        X, y = self._validate_data(X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = self._build_rs()
        _fit_rust(self._rs, X, y_encoded)
        self._store_training_data(X, y_encoded)
        return self

    def _build_rs(self):
        # Callable weights are NOT supported (the Rust core has only
        # Weights::{Uniform, Distance}); reject before the str-typed boundary
        # so the user sees a clear NotImplementedError, not a PyO3 TypeError
        # (NOT-STARTED #876).
        if callable(self.weights):
            raise NotImplementedError(
                "callable weights not supported (only 'uniform'/'distance'; "
                "NOT-STARTED #876)"
            )
        # metric_params is wrapper-validated: the Rust core has no custom
        # metric, so only None is supported (NOT-STARTED #876).
        if self.metric_params is not None:
            raise NotImplementedError(
                f"metric_params={self.metric_params!r} not supported "
                "(no custom metric; NOT-STARTED #876)"
            )
        return _RsKNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=float(self.p),
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def _rebuild_rs(self):
        self._rs = self._build_rs()
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))


class GaussianNB(_ClassifierPickleMixin, ClassifierMixin, BaseEstimator):
    """Gaussian Naive Bayes classifier backed by Rust.

    Parameters
    ----------
    var_smoothing : float, default=1e-9
        Variance smoothing parameter.
    """

    def __init__(self, *, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = self._validate_data(X, y, dtype="float64")
        _check_classification_target(y)
        X = _ensure_f64(X)
        y_encoded, self.classes_ = _encode_labels(y)
        self._rs = _RsGaussianNB(var_smoothing=self.var_smoothing)
        _fit_rust(self._rs, X, y_encoded)
        self.theta_ = np.array(self._rs.theta_)
        self.var_ = np.array(self._rs.var_)
        self.class_prior_ = np.array(self._rs.class_prior_)
        self.class_count_ = np.array(self._rs.class_count_)
        self.epsilon_ = float(self._rs.epsilon_)
        self._store_training_data(X, y_encoded)
        return self

    def _rebuild_rs(self):
        self._rs = _RsGaussianNB(var_smoothing=self.var_smoothing)
        self._rs.fit(self._fit_X, self._fit_y_encoded)

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        y_encoded = np.asarray(self._rs.predict(X))
        return _decode_labels(y_encoded, self.classes_)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.predict_proba(X))

    def predict_log_proba(self, X):
        # sklearn GaussianNB.predict_log_proba = jll - logsumexp(jll), which
        # equals log(predict_proba) where finite (sklearn/naive_bayes.py).
        return np.log(self.predict_proba(X))
