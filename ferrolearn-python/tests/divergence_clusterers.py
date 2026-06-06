"""Divergence guards for ferrolearn-python KMeans binding (unit #2028).

Targets: `ferrolearn-python/src/clusterers.rs` (RsKMeans) + its Python wrapper
`ferrolearn-python/python/ferrolearn/_clusterers.py` (class KMeans), the
`ferrolearn.KMeans` binding, mirroring `sklearn.cluster.KMeans`
(`sklearn/cluster/_kmeans.py:1196`).

Verification model B (goal.md): every expected value is computed by the LIVE
sklearn 1.5.2 oracle in the same test and compared against `import ferrolearn`.
No expected value is literal-copied from the ferrolearn side (R-CHAR-3).

Contents:
  - TWO FAILING constructor-ABI pins (single-wrapper-fixable, R-DEV-2):
      * n_clusters positional   -> sklearn KMeans(2) works; ferrolearn raises TypeError
      * n_init default 'auto'   -> sklearn default is 'auto'; ferrolearn default is 10
  - PASSING API-conformance green guards (method/attribute surface + RNG-robust
    structural label parity on well-separated blobs).

The structural NOT-STARTED gaps (missing init/algorithm/copy_x/verbose params,
missing score method, RNG-blocked exact-value parity) are filed as -l blocker
crosslink issues, NOT pinned here as failing tests, because they cannot be
closed by a single-wrapper change THIS iteration (R-DEFER-3).
"""

import numpy as np

import ferrolearn as fl
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# Shared deterministic dataset: two far-apart blobs (RNG-robust partition).
# ---------------------------------------------------------------------------


def _two_blobs():
    rng = np.random.RandomState(42)
    a = rng.randn(20, 2) * 0.1 + np.array([0.0, 0.0])
    b = rng.randn(20, 2) * 0.1 + np.array([10.0, 10.0])
    return np.vstack([a, b]).astype(np.float64)


# ===========================================================================
# (1) FAILING constructor-ABI pins (R-DEV-2; single-wrapper-fixable)
# ===========================================================================


def test_n_clusters_positional_matches_sklearn():
    """Divergence: ferrolearn.KMeans makes n_clusters keyword-only.

    sklearn `KMeans.__init__(self, n_clusters=8, *, ...)`
    (`sklearn/cluster/_kmeans.py:1387-1399`) places `n_clusters` BEFORE the `*`,
    so `KMeans(2)` is valid positional-or-keyword.
    ferrolearn `_clusterers.py::__init__(self, *, n_clusters=8, ...)` makes every
    param keyword-only, so `ferrolearn.KMeans(2)` raises TypeError.

    Oracle: sklearn.cluster.KMeans(2).n_clusters -> 2.
    Tracking: #2029
    """
    expected = SkKMeans(2).n_clusters  # live sklearn oracle -> 2

    # FAILS today: ferrolearn.KMeans(2) raises
    #   TypeError: __init__() takes 1 positional argument but 2 were given.
    model = fl.KMeans(2)
    assert model.n_clusters == expected


def test_n_init_default_matches_sklearn():
    """Divergence: ferrolearn.KMeans default n_init differs from sklearn.

    sklearn `KMeans.__init__(..., n_init="auto", ...)`
    (`sklearn/cluster/_kmeans.py:1392`) defaults `n_init` to the string 'auto'
    (resolving to 1 for the default init='k-means++' at fit time).
    ferrolearn `_clusterers.py::__init__(..., n_init=10, ...)` defaults to int 10.

    Oracle: sklearn.cluster.KMeans().n_init -> 'auto'.
    Tracking: #2030
    """
    expected = SkKMeans().n_init  # live sklearn oracle -> 'auto'
    assert expected == "auto"  # confirm the oracle (sanity, not the assertion under test)

    # FAILS today: ferrolearn default is 10, not 'auto'.
    assert fl.KMeans().n_init == expected


# ===========================================================================
# (2) PASSING API-conformance green guards (must PASS)
# ===========================================================================


def test_fitted_attribute_surface_matches_sklearn_shapes():
    """ferrolearn.KMeans exposes the sklearn fitted-attribute surface with the
    same shapes/types/value-domains as sklearn (`_kmeans.py:1291-1311`).

    Exact centroids are NOT asserted (RNG-blocked); structure + domains are.
    """
    X = _two_blobs()
    n_samples, n_features = X.shape

    sk = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    fm = fl.KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    # cluster_centers_ : (n_clusters, n_features) — same shape as sklearn
    assert fm.cluster_centers_.shape == sk.cluster_centers_.shape == (2, n_features)

    # labels_ : (n_samples,), values in {0,1}, same domain as sklearn
    assert len(fm.labels_) == len(sk.labels_) == n_samples
    assert set(np.asarray(fm.labels_).tolist()) <= set(range(2))
    assert set(np.asarray(fm.labels_).tolist()) == set(np.asarray(sk.labels_).tolist())

    # inertia_ : float >= 0
    assert isinstance(fm.inertia_, float)
    assert fm.inertia_ >= 0.0

    # n_iter_ : int >= 1
    assert isinstance(fm.n_iter_, int)
    assert fm.n_iter_ >= 1

    # n_features_in_ : == n_features (== sklearn)
    assert fm.n_features_in_ == sk.n_features_in_ == n_features


def test_predict_transform_surface_matches_sklearn():
    """predict/transform produce the same shapes/domains as sklearn."""
    X = _two_blobs()
    n_samples = X.shape[0]

    sk = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    fm = fl.KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    fpred = np.asarray(fm.predict(X))
    spred = np.asarray(sk.predict(X))
    assert fpred.shape == spred.shape == (n_samples,)
    assert set(fpred.tolist()) <= set(range(2))

    ftr = np.asarray(fm.transform(X))
    str_ = np.asarray(sk.transform(X))
    assert ftr.shape == str_.shape == (n_samples, 2)

    # predict consistency with transform (nearest center), as in sklearn
    assert np.array_equal(fpred, ftr.argmin(axis=1))


def test_fit_predict_and_fit_transform_mixins_work():
    """fit_predict / fit_transform (from ClusterMixin / TransformerMixin) produce
    the same shapes/domains as sklearn."""
    X = _two_blobs()
    n_samples = X.shape[0]

    sk_fp = np.asarray(SkKMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X))
    fl_fp = np.asarray(fl.KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X))
    assert fl_fp.shape == sk_fp.shape == (n_samples,)
    assert set(fl_fp.tolist()) <= set(range(2))

    sk_ft = np.asarray(SkKMeans(n_clusters=2, n_init=10, random_state=0).fit_transform(X))
    fl_ft = np.asarray(fl.KMeans(n_clusters=2, n_init=10, random_state=0).fit_transform(X))
    assert fl_ft.shape == sk_ft.shape == (n_samples, 2)


def test_well_separated_partition_matches_sklearn():
    """On cleanly-separated blobs, ferrolearn's partition equals sklearn's up to
    label permutation (RNG-robust structural parity, R-DEV-1 contract).

    adjusted_rand_score == 1.0 iff the two clusterings agree exactly modulo
    label renaming. Exact centroids differ (StdRng != numpy RNG), but on
    well-separated data both estimators recover the same partition.
    """
    X = _two_blobs()

    sk = SkKMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    fm = fl.KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    ari = adjusted_rand_score(np.asarray(fm.labels_), np.asarray(sk.labels_))
    assert ari == 1.0

    # And each blob (first 20 / last 20 rows) is one pure cluster.
    labels = np.asarray(fm.labels_)
    assert len(set(labels[:20].tolist())) == 1
    assert len(set(labels[20:].tolist())) == 1
    assert labels[0] != labels[20]


def test_kmeans_score_equals_neg_inertia():
    """REQ #2032: ferrolearn.KMeans exposes `score` = the negative inertia.

    sklearn `KMeans.score(X)` returns `-_labels_inertia(...)` (the negative sum
    of squared distances to the nearest center), which on the training data
    equals `-inertia_` (`sklearn/cluster/_kmeans.py:1156-1184`). This contract
    is RNG-independent: it relates `score` to `inertia_` for the SAME fit.
    """
    X = np.array(
        [[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9], [10.0, 10.0], [9.9, 10.1]]
    )
    # sklearn's own contract: score(X) == -inertia_ on the training data.
    sk = SkKMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    assert np.isclose(sk.score(X), -sk.inertia_)

    # ferrolearn must expose the same `score` contract.
    fm = fl.KMeans(n_clusters=3, random_state=0).fit(X)
    assert np.isclose(fm.score(X), -fm.inertia_)

    # sample_weight weights each sample's squared distance.
    sw = np.array([1.0, 2.0, 1.0, 1.0, 3.0, 1.0])
    diff = X[:, np.newaxis, :] - np.asarray(fm.cluster_centers_)[np.newaxis, :, :]
    expected = -float(np.sum(np.sum(diff * diff, axis=2).min(axis=1) * sw))
    assert np.isclose(fm.score(X, sample_weight=sw), expected)
