"""Live-oracle parity guards for the `ferrolearn.DBSCAN` Python binding surface
(#2190): `sample_weight`, `core_sample_indices_`, `components_`.

Verification model B (goal.md R-CHAR-3): every expected value comes from the
LIVE installed `sklearn.cluster.DBSCAN` 1.5.2 oracle in the SAME process, never
copied from ferrolearn. We run `ferrolearn.DBSCAN` and `sklearn.cluster.DBSCAN`
on the SAME fixture and assert element-wise array equality of `labels_`,
`core_sample_indices_`, and `components_`.

DBSCAN is deterministic (no RNG, no iterative optimizer), so on the Euclidean /
in-range-`eps` path ferrolearn VALUE-matches sklearn EXACTLY; these guards pin
that the new binding surface (weighted core determination + the two fitted
attributes) rides that parity through the Python layer.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import DBSCAN as SkDBSCAN

import ferrolearn as fl


def _assert_dbscan_parity(X, eps, min_samples, sample_weight=None):
    """Fit both estimators and assert labels_/core_sample_indices_/components_
    match the live sklearn oracle EXACTLY."""
    f = fl.DBSCAN(eps, min_samples=min_samples).fit(X, sample_weight=sample_weight)
    s = SkDBSCAN(eps=eps, min_samples=min_samples).fit(X, sample_weight=sample_weight)

    np.testing.assert_array_equal(f.labels_, s.labels_)
    np.testing.assert_array_equal(
        f.core_sample_indices_, s.core_sample_indices_
    )
    assert f.components_.shape == s.components_.shape
    np.testing.assert_array_equal(f.components_, s.components_)
    return f, s


def _multi_cluster_noise_fixture():
    # Two well-separated dense clusters of 4 points each + one far outlier.
    return np.array(
        [
            [0.0, 0.0], [0.3, 0.0], [0.0, 0.3], [0.2, 0.2],
            [5.0, 5.0], [5.3, 5.0], [5.0, 5.3], [5.2, 5.2],
            [50.0, 50.0],
        ],
        dtype=float,
    )


def test_unweighted_matches_sklearn():
    """fl.DBSCAN(eps, min_samples=k).fit(X) (no weight): labels_,
    core_sample_indices_, components_ all match sklearn EXACTLY."""
    X = _multi_cluster_noise_fixture()
    f, s = _assert_dbscan_parity(X, eps=0.5, min_samples=3)
    # Sanity: two clusters + noise found (oracle-derived from sklearn above).
    assert set(np.unique(s.labels_)) == {-1, 0, 1}
    assert s.components_.shape == (8, 2)


def test_default_params_match_sklearn():
    """Default eps=0.5/min_samples=5 fit (no explicit args) matches sklearn."""
    X = _multi_cluster_noise_fixture()
    f = fl.DBSCAN().fit(X)
    s = SkDBSCAN().fit(X)
    np.testing.assert_array_equal(f.labels_, s.labels_)
    np.testing.assert_array_equal(
        f.core_sample_indices_, s.core_sample_indices_
    )
    np.testing.assert_array_equal(f.components_, s.components_)


def test_sample_weight_promotes_isolated_to_core():
    """A high-weight isolated point becomes its own core sample (>= min_samples)
    even though unweighted it is noise. labels_/core_sample_indices_/components_
    match the live sklearn oracle."""
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=float)
    w = np.array([5.0, 1.0, 1.0])
    # Oracle-anchored sanity: unweighted -> all noise; weighted -> idx0 a core.
    assert np.array_equal(
        SkDBSCAN(eps=0.5, min_samples=3).fit(X).labels_, [-1, -1, -1]
    )
    f, s = _assert_dbscan_parity(X, eps=0.5, min_samples=3, sample_weight=w)
    assert 0 in s.core_sample_indices_
    assert s.labels_[0] == 0


def test_sample_weight_demotes_cluster_to_noise():
    """A tight 4-point cluster with each weight 0.5 and min_samples=4 has
    neighborhood weight sum 2.0 < 4 -> all noise, vs unweighted one cluster.
    Matches the live sklearn oracle (incl. the empty (0, n_features)
    components_)."""
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]], dtype=float
    )
    w = np.full(4, 0.5)
    # Oracle-anchored sanity: unweighted -> one cluster.
    assert np.array_equal(
        SkDBSCAN(eps=0.5, min_samples=4).fit(X).labels_, [0, 0, 0, 0]
    )
    f, s = _assert_dbscan_parity(X, eps=0.5, min_samples=4, sample_weight=w)
    assert np.all(s.labels_ == -1)
    assert s.components_.shape == (0, 2)


def test_pairwise_weight_sum_boundary_2189():
    """#2189: ten co-located points each weight 0.1, min_samples=1. The
    neighborhood weight sum straddles the float `>=` core boundary: a SEQUENTIAL
    left-fold gives 0.9999999999999999 (< 1 -> would demote ALL to noise), but
    numpy's PAIRWISE summation gives exactly 1.0 (>= 1 -> all core). sklearn uses
    np.sum (pairwise), so all ten are core; the binding must reproduce that
    reduction order. Asserting parity here pins the pairwise-sum path THROUGH the
    binding."""
    X = np.zeros((10, 2), dtype=float)
    w = np.full(10, 0.1)
    # Document the reduction-order divergence the core must avoid.
    seq = 0.0
    for _ in range(10):
        seq += 0.1
    assert seq < 1.0  # a naive sequential core would demote everything
    assert np.sum(w) == 1.0  # numpy pairwise = the sklearn oracle's reduction
    f, s = _assert_dbscan_parity(X, eps=0.5, min_samples=1, sample_weight=w)
    # All ten are core (the pairwise-sum boundary), per the live oracle.
    assert len(s.core_sample_indices_) == 10
    np.testing.assert_array_equal(f.core_sample_indices_, np.arange(10))


def test_wrong_length_sample_weight_raises():
    """A wrong-length sample_weight raises ValueError (the Rust core's
    ShapeMismatch -> PyValueError, mirroring _check_sample_weight) — no
    panic/segfault (R-CODE-2)."""
    X = np.zeros((5, 2), dtype=float)
    with pytest.raises(ValueError):
        fl.DBSCAN(0.5, min_samples=2).fit(X, sample_weight=np.ones(3))


def test_fit_predict_threads_sample_weight():
    """fit_predict(X, sample_weight=w) routes through fit and matches sklearn's
    fit_predict (which threads sample_weight, _dbscan.py:450)."""
    X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]], dtype=float)
    w = np.array([5.0, 1.0, 1.0])
    fl_labels = fl.DBSCAN(0.5, min_samples=3).fit_predict(X, sample_weight=w)
    sk_labels = SkDBSCAN(eps=0.5, min_samples=3).fit_predict(X, sample_weight=w)
    np.testing.assert_array_equal(fl_labels, sk_labels)


def test_sklearn_compat_clone_get_set_params():
    """sklearn-estimator contract: default params, clone(), get_params/
    set_params round-trip, and the cloned estimator still fits + matches."""
    est = fl.DBSCAN()
    assert est.eps == 0.5
    assert est.min_samples == 5
    params = est.get_params()
    assert params == {"eps": 0.5, "min_samples": 5}

    cloned = clone(est)
    assert cloned.get_params() == params

    est.set_params(eps=1.0, min_samples=2)
    assert est.eps == 1.0 and est.min_samples == 2

    # A clone with custom params fits and matches the live oracle.
    X = _multi_cluster_noise_fixture()
    custom = clone(fl.DBSCAN(0.6, min_samples=2))
    f = custom.fit(X)
    s = SkDBSCAN(eps=0.6, min_samples=2).fit(X)
    np.testing.assert_array_equal(f.labels_, s.labels_)
    np.testing.assert_array_equal(
        f.core_sample_indices_, s.core_sample_indices_
    )


def test_negative_weight_inhibits_neighbor():
    """sklearn DBSCAN accepts negative weights (no only_non_negative); a negative
    weight can inhibit an eps-neighbor from being core (_dbscan.py:386-387). The
    binding must accept it and match the live oracle, not reject it."""
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]], dtype=float
    )
    w = np.array([-5.0, 1.0, 1.0, 1.0])
    _assert_dbscan_parity(X, eps=0.5, min_samples=2, sample_weight=w)
