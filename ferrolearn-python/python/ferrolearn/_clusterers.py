"""sklearn-compatible wrappers for ferrolearn clustering models."""

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ferrolearn._ferrolearn_rs import _RsKMeans


def _ensure_f64(arr):
    """Ensure array is C-contiguous float64 (required by Rust bindings)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """K-Means clustering backed by Rust.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Relative tolerance for convergence.
    n_init : 'auto' or int, default='auto'
        Number of initializations to run. When ``n_init='auto'``, the number of
        runs is 1 for the default ``init='k-means++'``.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self, n_clusters=8, *, max_iter=300, tol=1e-4, n_init="auto", random_state=None
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X, y=None):
        X = self._validate_data(X, dtype="float64")
        # sklearn keeps n_init='auto' verbatim and resolves it at fit time
        # (sklearn/cluster/_kmeans.py:1240-1243): 'auto' -> 1 for the default
        # init='k-means++' (the only init ferrolearn_cluster supports).
        n_init = 1 if self.n_init == "auto" else self.n_init
        self._rs = _RsKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=n_init,
            random_state=self.random_state,
        )
        self._rs.fit(_ensure_f64(X))
        self.cluster_centers_ = np.array(self._rs.cluster_centers_)
        self.labels_ = np.asarray(self._rs.labels_)
        self.inertia_ = float(self._rs.inertia_)
        self.n_iter_ = int(self._rs.n_iter_)
        self._fit_X = X.copy()
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.asarray(self._rs.predict(X))

    def transform(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        if not hasattr(self, "_rs"):
            self._rebuild_rs()
        return np.array(self._rs.transform(X))

    def score(self, X, y=None, sample_weight=None):
        # sklearn KMeans.score = the negative inertia on X, i.e.
        # -sum_i sample_weight_i * min_k ||x_i - cluster_centers_[k]||^2
        # (sklearn/cluster/_kmeans.py:1156-1184, -_labels_inertia).
        # On the training data this equals -inertia_.
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype="float64")
        X = _ensure_f64(X)
        # Squared distance from each sample to each center, then min over centers.
        diff = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        min_sq = sq_dists.min(axis=1)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype="float64")
            return -float(np.sum(min_sq * sw))
        return -float(np.sum(min_sq))

    def get_feature_names_out(self, input_features=None):
        # KMeans is a transformer (distance-to-centroid); its output feature
        # names are `kmeans0..kmeans{n_clusters-1}`
        # (ClassNamePrefixFeaturesOutMixin, sklearn/cluster/_kmeans.py).
        check_is_fitted(self)
        n_out = self.cluster_centers_.shape[0]
        return np.asarray([f"kmeans{i}" for i in range(n_out)], dtype=object)

    def _rebuild_rs(self):
        n_init = 1 if self.n_init == "auto" else self.n_init
        self._rs = _RsKMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=n_init,
            random_state=self.random_state,
        )
        self._rs.fit(_ensure_f64(self._fit_X))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_rs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, "_fit_X"):
            self._rebuild_rs()
